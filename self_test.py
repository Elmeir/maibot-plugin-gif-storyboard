"""GIF 动图帧合成插件本地自检脚本。

不依赖宿主：用假的 maibot_sdk 模块加载 plugin.py，
覆盖多帧合成、单帧/非 GIF 放行、缓存命中、钩子改写路径、
表情包三种策略（bypass 旁路识别 / replace 直接替换 / off 不处理），
以及 after_process 回收与"多帧描述搬运"（含 database.query 故障告警、
超时兜底）逻辑。

用法：
    pip install Pillow
    python self_test.py
"""

import asyncio
import base64
import hashlib
import io
import sys
import types
from pathlib import Path

HERE = Path(__file__).parent


def install_fake_sdk() -> None:
    """注入一个最小化的 maibot_sdk 假实现，让 plugin.py 可独立导入。"""

    sdk = types.ModuleType("maibot_sdk")

    class MaiBotPlugin:
        def __init__(self) -> None:
            self.config = None
            self.ctx = None

    class PluginConfigBase:
        pass

    def Field(default=None, default_factory=None, **kwargs):  # noqa: ANN001
        return default_factory() if default_factory is not None else default

    class HookHandler:  # noqa: D401 装饰器占位：保留原函数即可
        def __init__(self, *args, **kwargs) -> None:
            self.hook = args[0] if args else kwargs.get("hook")

        def __call__(self, func):  # noqa: ANN001
            func._hook_name = self.hook
            return func

    sdk.MaiBotPlugin = MaiBotPlugin
    sdk.PluginConfigBase = PluginConfigBase
    sdk.Field = Field
    sdk.HookHandler = HookHandler

    sdk_types = types.ModuleType("maibot_sdk.types")

    class _Mode:
        def __init__(self, value: str) -> None:
            self.value = value

        def __repr__(self) -> str:  # pragma: no cover
            return self.value

    sdk_types.HookMode = types.SimpleNamespace(
        BLOCKING=_Mode("blocking"), OBSERVE=_Mode("observe")
    )
    sdk_types.HookOrder = types.SimpleNamespace(
        EARLY=_Mode("early"), NORMAL=_Mode("normal"), LATE=_Mode("late")
    )
    sdk_types.ErrorPolicy = types.SimpleNamespace(
        SKIP=_Mode("skip"), LOG=_Mode("log"), ABORT=_Mode("abort")
    )

    sys.modules["maibot_sdk"] = sdk
    sys.modules["maibot_sdk.types"] = sdk_types


class FakeLogger:
    """记录日志到内存，供断言检查。"""

    def __init__(self) -> None:
        self.records: list[tuple[str, str]] = []

    def _log(self, level: str, msg: str, *args) -> None:  # noqa: ANN001
        self.records.append((level, (msg % args) if args else str(msg)))

    def debug(self, msg: str, *args) -> None:  # noqa: ANN001
        self._log("debug", msg, *args)

    def info(self, msg: str, *args) -> None:  # noqa: ANN001
        self._log("info", msg, *args)

    def warning(self, msg: str, *args) -> None:  # noqa: ANN001
        self._log("warning", msg, *args)

    def error(self, msg: str, *args) -> None:  # noqa: ANN001
        self._log("error", msg, *args)

    def texts(self, level: str | None = None) -> list[str]:
        return [t for lv, t in self.records if level is None or lv == level]


class FakeDb:
    """内存版 Images 表，模拟宿主 database.query 能力的 get/update。"""

    def __init__(self) -> None:
        self.records: dict[tuple[str, str], dict] = {}
        self.raise_error: Exception | None = None
        self.updates: list[dict] = []

    def put(
        self,
        image_hash: str,
        image_type: str,
        description: str = "",
        vlm_processed: bool = False,
    ) -> None:
        self.records[(image_hash, image_type)] = {
            "image_hash": image_hash,
            "image_type": image_type,
            "description": description,
            "vlm_processed": vlm_processed,
        }

    async def query(  # noqa: ANN003
        self, *, model_name, query_type, filters, data=None, limit=None, single_result=False, **kwargs
    ):
        if self.raise_error is not None:
            raise self.raise_error
        if model_name != "Images":
            return {"success": False, "error": f"未知 model_name: {model_name}"}
        key = (filters.get("image_hash"), filters.get("image_type"))
        if query_type == "get":
            hit = self.records.get(key)
            if hit is None:
                return None
            return dict(hit)
        if query_type == "update":
            hit = self.records.get(key)
            if hit is None:
                return 0  # 与宿主 db_update 一致：返回受影响行数
            hit.update(data or {})
            self.updates.append({"filters": dict(filters), "data": dict(data or {})})
            return 1
        return {"success": False, "error": f"未知 query_type: {query_type}"}


class FakeMaisakaContext:
    """记录 maisaka.context.append 调用的假实现。"""

    def __init__(self) -> None:
        self.appends: list[dict] = []
        self.fail: Exception | None = None

    async def append(self, stream_id, segments, **kwargs):  # noqa: ANN001
        if self.fail is not None:
            raise self.fail
        self.appends.append({"stream_id": stream_id, "segments": segments, **kwargs})
        return {"success": True, "index": len(self.appends) - 1, "visible_text": kwargs.get("visible_text", "")}


class FakeMaisaka:
    def __init__(self) -> None:
        self.context = FakeMaisakaContext()


class FakeCtx:
    """最小化宿主上下文：logger + db + maisaka。"""

    def __init__(self) -> None:
        self.logger = FakeLogger()
        self.db = FakeDb()
        self.maisaka = FakeMaisaka()


def make_gif(frame_count: int, size: tuple = (120, 90)) -> bytes:
    """生成一个 frame_count 帧、每帧颜色渐变的测试 GIF。"""
    from PIL import Image

    frames = []
    for i in range(frame_count):
        color = (int(255 * i / max(1, frame_count - 1)), 80, 160)
        frames.append(Image.new("RGB", size, color))
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], duration=100)
    return buf.getvalue()


def make_png() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (10, 200, 10)).save(buf, format="PNG")
    return buf.getvalue()


def component_for(data: bytes, comp_type: str = "image") -> dict:
    return {
        "type": comp_type,
        "data": "",
        "hash": hashlib.sha256(data).hexdigest(),
        "binary_data_base64": base64.b64encode(data).decode("ascii"),
    }


def main() -> int:
    from PIL import Image

    install_fake_sdk()
    sys.path.insert(0, str(HERE))
    import plugin as plugin_module

    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        status = "PASS" if cond else "FAIL"
        print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
        if not cond:
            failures.append(name)

    # ── 构造插件实例 ──
    pl = plugin_module.create_plugin()
    pl.config = plugin_module.GifStoryboardConfig()
    pl.ctx = FakeCtx()

    # ── 1. 多帧 GIF：组件替换为合成图（识别窗口内），原图字节暂存 ──
    gif_12 = make_gif(12)
    comp = component_for(gif_12)
    msg1 = {"message_id": "t1", "session_id": "s1", "raw_message": [comp]}
    new_comp = asyncio.run(pl.handle_before_process(message=msg1))["modified_kwargs"]["message"]["raw_message"][0]
    check("多帧 GIF：组件被替换", new_comp is not comp and new_comp.get("type") == "image")
    merged_b64 = new_comp.get("binary_data_base64", "")
    merged = base64.b64decode(merged_b64)
    check("合成图不再是 GIF 魔数", not plugin_module.GifStoryboardPlugin._is_gif(merged))
    check("合成图是 JPEG（默认输出格式）", merged[:2] == b"\xff\xd8")
    check("hash 已同步为合成图摘要", new_comp.get("hash") == hashlib.sha256(merged).hexdigest())
    with Image.open(io.BytesIO(merged)) as out_img:
        w, h = out_img.size
        check("合成图尺寸在网格上限内", w <= 3 * 480 + 5 * 4 + 2 and h <= 3 * 480 + 5 * 4 + 2, f"{w}x{h}")
    check("data（content）保持空占位", new_comp.get("data", "") == "" and new_comp.get("type") == "image")
    check(
        "原图字节已进入暂存缓存",
        pl._original_bytes.get(comp["hash"]) == gif_12
        and plugin_module.GifStoryboardPlugin._is_gif(pl._original_bytes.get(comp["hash"], b"")),
    )
    check("替换按组件索引登记", pl._replaced_index.get(0) == (new_comp["hash"], comp["hash"]))
    check("搬运映射登记", pl._ghost_records.get(new_comp["hash"]) == ("image", comp["hash"]))

    # ── 2. 12 帧均匀抽样 9 帧：应含首尾帧 ──
    idx = plugin_module.GifStoryboardPlugin._sample_frame_indices(12, 9)
    check("均匀抽样包含首帧", idx[0] == 0, str(idx))
    check("均匀抽样包含尾帧", idx[-1] == 11, str(idx))
    check("抽样数量正确且无重复", len(idx) == 9 and len(set(idx)) == 9, str(idx))

    # ── 3. 单帧 GIF：原样放行 ──
    single = component_for(make_gif(1))
    same = asyncio.run(pl._process_component(single))
    check("单帧 GIF 原样放行", same == [single])

    # ── 4. 非 GIF 图片：原样放行 ──
    png_comp = component_for(make_png())
    same_png = asyncio.run(pl._process_component(png_comp))
    check("PNG 原样放行", same_png == [png_comp])

    # ── 5. 无二进制的图片组件：原样放行 ──
    empty_comp = {"type": "image", "data": "", "hash": "x"}
    same_empty = asyncio.run(pl._process_component(empty_comp))
    check("缺 binary_data_base64 时放行", same_empty == [empty_comp])

    # ── 6. emoji 组件三种策略 ──
    emoji_comp = component_for(gif_12, comp_type="emoji")

    # bypass（默认）：原表情保留 + 追加合成图 image 组件
    bypass = asyncio.run(pl._process_component(emoji_comp))
    check("bypass：原表情组件原样保留", len(bypass) == 2 and bypass[0] is emoji_comp)
    ghost = bypass[1]
    check("bypass：追加的是 image 组件且 content 为空", ghost.get("type") == "image" and ghost.get("data", "") == "")
    check("bypass：合成图与 image 链路结果一致", ghost.get("binary_data_base64") == merged_b64)
    check("bypass：追加组件带新 hash", ghost.get("hash") == hashlib.sha256(merged).hexdigest())

    # replace：直接替换表情二进制
    pl.config.merge.emoji_strategy = "replace"
    replaced = asyncio.run(pl._process_component(emoji_comp))
    check(
        "replace：表情二进制被替换",
        len(replaced) == 1 and replaced[0] is not emoji_comp and replaced[0].get("type") == "emoji",
    )
    check("replace：替换后不再是 GIF 魔数", not plugin_module.GifStoryboardPlugin._is_gif(base64.b64decode(replaced[0]["binary_data_base64"])))

    # off：不处理
    pl.config.merge.emoji_strategy = "off"
    off = asyncio.run(pl._process_component(emoji_comp))
    check("off：表情组件放行", off == [emoji_comp])
    pl.config.merge.emoji_strategy = "bypass"

    # ── 7. 缓存命中：同图二次合成直接复用 ──
    cache_size = len(pl._merge_cache)
    merged_again = asyncio.run(pl._process_component(component_for(gif_12)))[0]
    check(
        "同 GIF 二次处理命中缓存且结果一致",
        len(pl._merge_cache) == cache_size and merged_again.get("binary_data_base64") == merged_b64,
    )

    # ── 8. 钩子改写路径：含 GIF 图片 + GIF 表情 + PNG 的消息 ──
    message = {
        "message_id": "m1",
        "session_id": "s1",
        "raw_message": [
            {"type": "text", "data": "看这个"},
            component_for(gif_12),
            component_for(gif_12, comp_type="emoji"),
            png_comp,
        ],
    }
    result = asyncio.run(pl.handle_before_process(message=message))
    modified = result.get("modified_kwargs", {}).get("message", {})
    comps = modified.get("raw_message", [])
    check("钩子返回 modified_kwargs", bool(result.get("modified_kwargs")))
    check("图片替换、表情追加 ghost → 4+1=5", len(comps) == 5, f"实际 {len(comps)}")
    check("文本组件原样保留", comps[0] == message["raw_message"][0])
    check("GIF 图片组件被替换为合成图", (
        comps[1] is not message["raw_message"][1]
        and comps[1].get("hash") == new_comp["hash"]
        and comps[1].get("data", "") == ""
    ))
    check("emoji 原组件原样保留", comps[2] is message["raw_message"][2])
    check("emoji 之后是追加的 ghost 合成图", comps[3].get("hash") == new_comp["hash"])
    check("PNG 组件原样保留", comps[4] is message["raw_message"][3])
    check("替换按组件索引登记（idx=1）", pl._replaced_index.get(1) == (comps[1]["hash"], comp["hash"]))
    check("表情 ghost 按 hash 登记", pl._emoji_ghosts.get(comps[3]["hash"]) == comp["hash"])
    check("搬运映射登记（emoji）", pl._emoji_ghosts and pl._ghost_records.get(comps[3]["hash"]) == ("emoji", comp["hash"]))

    # ── 9. 纯文本消息：钩子直接放行 ──
    text_only = {"message_id": "m2", "session_id": "s1", "raw_message": [{"type": "text", "data": "hi"}]}
    res2 = asyncio.run(pl.handle_before_process(message=text_only))
    check("纯文本消息不触发改写", "modified_kwargs" not in res2)

    # ── 10. 总开关关闭：一律放行 ──
    pl.config.plugin.enabled = False
    res3 = asyncio.run(pl.handle_before_process(message=message))
    check("总开关关闭时不改写", "modified_kwargs" not in res3)
    pl.config.plugin.enabled = True

    # ── 11. 表情策略 off：含 emoji 的消息不再触发 emoji 改写 ──
    pl.config.merge.emoji_strategy = "off"
    emoji_only = {"message_id": "m3", "session_id": "s1", "raw_message": [component_for(gif_12, comp_type="emoji")]}
    res4 = asyncio.run(pl.handle_before_process(message=emoji_only))
    check("off 策略下纯 emoji 消息不触发改写", "modified_kwargs" not in res4)
    pl.config.merge.emoji_strategy = "bypass"

    # ── 12. 抽样帧数上限与参数指纹 ──
    fp_before = pl._settings_fingerprint()
    pl.config.merge.max_frames = 4
    check("参数指纹随配置变化", fp_before != pl._settings_fingerprint())
    comp_4 = asyncio.run(pl._process_component(component_for(make_gif(20))))[0]
    with Image.open(io.BytesIO(base64.b64decode(comp_4["binary_data_base64"]))):
        pass  # 能正常解码即可
    check("max_frames=4 时 20 帧 GIF 仍可合成", comp_4 is not None)
    pl.config.merge.max_frames = 9

    # ── 13. after_process：恢复原图组件 + 回收 ghost + 调度搬运 ──
    db = pl.ctx.db
    spawn_calls: list[str] = []
    pl._spawn_relocate_task = (
        lambda merged_hash, image_type, orig_hash: spawn_calls.append(merged_hash)
        if merged_hash not in spawn_calls
        else None
    )  # noqa: E731
    after_msg = {"message_id": "m5", "session_id": "s1", "raw_message": comps}
    after_res = asyncio.run(pl.handle_after_process(message=after_msg))
    kept = after_res.get("modified_kwargs", {}).get("message", {}).get("raw_message", [])
    check("after_process 返回 modified_kwargs", bool(after_res.get("modified_kwargs")))
    check("恢复原图 + 回收 ghost → 4 组件", len(kept) == 4, f"实际 {len(kept)}")
    check("组件已恢复为原图（二进制与 hash）", (
        kept[1].get("hash") == comp["hash"]
        and base64.b64decode(kept[1]["binary_data_base64"]) == gif_12
        and kept[1].get("data", "") == ""
    ))
    check("替换索引已消费", 1 not in pl._replaced_index)
    check("ghost 登记已消费", new_comp["hash"] not in pl._emoji_ghosts)
    check("恢复后原图字节缓存清理", comp["hash"] not in pl._original_bytes)
    check("after 后搬运映射已消费", new_comp["hash"] not in pl._ghost_records)
    check("恢复后调度了搬运任务（同 hash 去重）", set(spawn_calls) == {new_comp["hash"]}, str(spawn_calls))

    # ── 14. 描述搬运：合成图识别就绪且原图记录存在 → 覆盖写入 ──
    merged_hash = new_comp["hash"]
    orig_hash = str(comp["hash"])
    pl._ghost_records[merged_hash] = ("image", orig_hash)
    db.put(merged_hash, "image", description="一张多帧动画分镜……", vlm_processed=True)
    db.put(orig_hash, "image", description="", vlm_processed=False)  # 落库建的干净记录
    asyncio.run(pl._relocate_description(merged_hash, "image", orig_hash))
    check(
        "搬运成功：原图记录被覆盖为多帧描述",
        db.records[(orig_hash, "image")]["description"] == "一张多帧动画分镜……",
    )
    check("搬运后 vlm_processed 置 True", db.records[(orig_hash, "image")]["vlm_processed"] is True)
    check("搬运后任务标记被清理", merged_hash not in pl._relocate_tasks)

    # ── 15. 描述搬运：原图记录未建（落库前）→ 轮询等待 → 超时告警 ──
    db.records.clear()
    merged2 = hashlib.sha256(b"merged-2").hexdigest()
    orig2 = hashlib.sha256(b"orig-2").hexdigest()
    db.put(merged2, "image", description="多帧描述 B", vlm_processed=True)
    check(
        "原图记录缺失时不写入",
        asyncio.run(pl._try_relocate(merged2, orig2, "image")) is False,
    )
    db.put(orig2, "image", description="", vlm_processed=False)  # 落库完成
    check("原图记录就绪后写入多帧描述", (
        asyncio.run(pl._try_relocate(merged2, orig2, "image")) is True
        and db.records[(orig2, "image")]["description"] == "多帧描述 B"
    ))
    pl._ghost_records[merged2] = ("image", orig2)
    old_timeout, old_interval = (
        plugin_module.RELOCATE_TIMEOUT_SECONDS,
        plugin_module.RELOCATE_POLL_INTERVAL_SECONDS,
    )
    plugin_module.RELOCATE_TIMEOUT_SECONDS = 0.05
    plugin_module.RELOCATE_POLL_INTERVAL_SECONDS = 0.01
    try:
        db.records.pop((orig2, "image"), None)  # 模拟记录始终缺失
        asyncio.run(pl._relocate_description(merged2, "image", orig2))
        check("记录缺失时搬运超时告警", any("搬运超时放弃" in t for t in pl.ctx.logger.texts("warning")))
    finally:
        plugin_module.RELOCATE_TIMEOUT_SECONDS = old_timeout
        plugin_module.RELOCATE_POLL_INTERVAL_SECONDS = old_interval

    # ── 16. database.query 故障（如未声明能力被宿主拒绝）：只告警一次 ──
    db.records.clear()
    db.updates.clear()
    db.raise_error = RuntimeError("插件 github.elmeir.gif-storyboard 未获授权能力: database.query")
    pl._db_warned = False
    r1 = asyncio.run(pl._db_get_image_record("x" * 64, "image"))
    r2 = asyncio.run(pl._db_get_image_record("y" * 64, "image"))
    warn_once = [t for t in pl.ctx.logger.texts("warning") if "database.query 能力不可用" in t]
    check("db 故障时返回 None", r1 is None and r2 is None)
    check("db 故障告警去重（仅一次）", len(warn_once) == 1, f"实际 {len(warn_once)} 次")
    check("告警文案包含能力声明提示", bool(warn_once) and "capabilities" in warn_once[0])
    db.raise_error = None

    # ── 17. on_load 数据库能力自检 ──
    pl._db_warned = False
    asyncio.run(pl._check_db_access())
    check("db 可用时自检通过", any("自检通过" in t for t in pl.ctx.logger.texts("info")))
    db.raise_error = RuntimeError("未注册能力令牌")
    pl._db_warned = False
    asyncio.run(pl._check_db_access())
    check("db 不可用时自检告警", any("能力不可用" in t for t in pl.ctx.logger.texts("warning")))
    db.raise_error = None

    # ── 18. on_unload 清理运行态 ──
    pl._ghost_records["z" * 64] = "w" * 64
    pl._replaced_index[99] = ("y" * 64, "x" * 64)
    pl._emoji_ghosts["v" * 64] = "u" * 64
    asyncio.run(pl.on_unload())
    check(
        "卸载后运行态被清空",
        not pl._ghost_records
        and not pl._relocate_tasks
        and not pl._original_bytes
        and not pl._replaced_index
        and not pl._emoji_ghosts,
    )

    # ── 19. 重复 GIF 快速路径：多帧描述已就绪时直接写入组件文本 ──
    pl._ghost_records.clear()
    pl._relocate_tasks.clear()
    fast_gif = make_gif(6)
    fast_comp = component_for(fast_gif)
    fast_out = asyncio.run(pl._process_component(fast_comp))
    check(
        "首次出现走替换流程",
        len(fast_out) == 1 and fast_out[0].get("hash") != fast_comp["hash"],
    )
    merged_fast_hash = str(fast_out[0]["hash"])
    orig_fast_hash = str(fast_comp["hash"])
    db.put(merged_fast_hash, "image", description="多帧描述 D", vlm_processed=True)
    db.put(orig_fast_hash, "image", description="首帧描述 D", vlm_processed=True)
    fast_again = asyncio.run(pl._process_component(component_for(fast_gif)))
    check("重复 GIF 走快速路径（不替换二进制）", len(fast_again) == 1 and fast_again[0] is not fast_comp)
    check("快速路径写入多帧描述到 content", fast_again[0].get("data") == "[图片：多帧描述 D]")
    check(
        "快速路径保留原图二进制与 hash",
        fast_again[0].get("binary_data_base64") == fast_comp["binary_data_base64"]
        and fast_again[0].get("hash") == orig_fast_hash,
    )
    db.records.pop((orig_fast_hash, "image"), None)
    fallback = asyncio.run(pl._process_component(component_for(fast_gif)))
    check(
        "原图记录缺失时退回替换流程",
        len(fallback) == 1 and fallback[0].get("hash") != orig_fast_hash,
    )
    emoji_fast_comp = component_for(fast_gif, comp_type="emoji")
    db.put(orig_fast_hash, "emoji", description="", vlm_processed=True)
    emoji_fast = asyncio.run(pl._process_component(emoji_fast_comp))
    check(
        "emoji 快速路径写入多帧描述",
        len(emoji_fast) == 1 and emoji_fast[0].get("data") == "[表情包: 多帧描述 D]",
    )

    # ── 20. 自适应帧数 + 分段清晰度抽帧 ──
    from PIL import ImageDraw

    def solid_gif(frame_count: int, size=(80, 60), color=(120, 120, 120)) -> bytes:
        # 每帧 1px 微差：保证 Pillow 写出多帧 GIF，但感知哈希视之为"静止"
        frames = []
        for i in range(frame_count):
            img = Image.new("RGB", size, color)
            img.putpixel((i % size[0], 0), (color[0] + (i % 3), color[1], color[2]))
            frames.append(img)
        buf = io.BytesIO()
        frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], duration=100)
        return buf.getvalue()

    def checker_gif(frame_count: int, size=(80, 60)) -> bytes:
        frames = []
        for i in range(frame_count):
            img = Image.new("RGB", size, (0, 0, 0) if i % 2 == 0 else (255, 255, 255))
            ImageDraw.Draw(img).rectangle(
                (4, 4, size[0] - 5, size[1] - 5), fill=(200, 60, 60) if i % 2 else (60, 90, 200)
            )
            frames.append(img)
        buf = io.BytesIO()
        frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], duration=100)
        return buf.getvalue()

    with Image.open(io.BytesIO(solid_gif(20))) as im_static:
        count_static = pl._adaptive_frame_count(im_static, 20, 9)
    with Image.open(io.BytesIO(checker_gif(60))) as im_motion:
        count_motion = pl._adaptive_frame_count(im_motion, 60, 9)
    check("静止动画自适应取下限帧数", count_static == pl.config.merge.min_frames, str(count_static))
    check("高运动长动画自适应取上限帧数", count_motion == 9, str(count_motion))

    pl.config.merge.adaptive_frames = True
    pl.ctx.logger.records.clear()
    static_out = asyncio.run(pl._process_component(component_for(solid_gif(20))))
    static_grid = Image.open(io.BytesIO(base64.b64decode(static_out[0]["binary_data_base64"])))
    check(
        "合成入口：静止动画自适应降帧",
        bool(pl.ctx.logger.records) and " 2 帧" in " ".join(t for _, t in pl.ctx.logger.records),
        f"logs={pl.ctx.logger.records} grid={static_grid.size} hash={static_out[0].get('hash')}",
    )
    pl.config.merge.adaptive_frames = False
    pl.ctx.logger.records.clear()
    asyncio.run(pl._process_component(component_for(solid_gif(20))))
    check(
        "关闭自适应后固定抽满上限",
        not [t for _, t in pl.ctx.logger.records if "自适应帧数" in t],
    )
    pl.config.merge.adaptive_frames = True

    sharp_gif = make_gif(24)
    from PIL import ImageDraw

    img_a = Image.new("L", (64, 64), 0)
    ImageDraw.Draw(img_a).rectangle((0, 0, 31, 63), fill=255)  # 左亮右暗
    img_b = Image.new("L", (64, 64), 0)
    ImageDraw.Draw(img_b).rectangle((32, 0, 63, 63), fill=255)  # 左暗右亮
    d_a = plugin_module.GifStoryboardPlugin._dhash(img_a)
    d_b = plugin_module.GifStoryboardPlugin._dhash(img_b)
    check("dhash 基本性质（同图距离 0、异图有区分）", (
        plugin_module.GifStoryboardPlugin._hamming(d_a, d_a) == 0
        and plugin_module.GifStoryboardPlugin._hamming(d_a, d_b) > 0
    ))
    with Image.open(io.BytesIO(sharp_gif)) as im_sharp:
        picked = plugin_module.GifStoryboardPlugin._pick_frame_indices(im_sharp, 24, 9)
    check("清晰度抽帧数量与唯一性", len(picked) == 9 and len(set(picked)) == 9, str(picked))
    check("清晰度抽帧保持时序", picked == sorted(picked), str(picked))
    check("清晰度抽帧覆盖首尾", picked[0] == 0 and picked[-1] == 23, str(picked))

    print()
    if failures:
        print(f"自检未通过：{len(failures)} 项 — {failures}")
        return 1
    print("全部自检通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
