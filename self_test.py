"""GIF 动图帧合成插件本地自检脚本。

不依赖宿主：用假的 maibot_sdk 模块加载 plugin.py，
覆盖多帧合成、单帧/非 GIF 放行、缓存命中、钩子改写路径与
表情包三种策略（bypass 旁路识别 / replace 直接替换 / off 不处理）。

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

    # ── 1. 多帧 GIF：合成并替换 ──
    gif_12 = make_gif(12)
    comp = component_for(gif_12)
    new_comp = asyncio.run(pl._process_component(comp))[0]
    check("多帧 GIF 被替换为新组件", new_comp is not comp)
    merged_b64 = new_comp.get("binary_data_base64", "")
    merged = base64.b64decode(merged_b64)
    check("合成图不再是 GIF 魔数", not plugin_module.GifStoryboardPlugin._is_gif(merged))
    check("合成图是 JPEG（默认输出格式）", merged[:2] == b"\xff\xd8")
    check("hash 已同步为新字节摘要", new_comp.get("hash") == hashlib.sha256(merged).hexdigest())
    with Image.open(io.BytesIO(merged)) as out:
        w, h = out.size
        check("合成图尺寸在网格上限内", w <= 3 * 480 + 5 * 4 + 2 and h <= 3 * 480 + 5 * 4 + 2, f"{w}x{h}")
    check("data（content）字段未被改动", new_comp.get("data", "") == "" and new_comp.get("type") == "image")

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
    check("image 替换不增组件、emoji bypass 追加一个 → 4+1=5", len(comps) == 5, f"实际 {len(comps)}")
    check("文本组件原样保留", comps[0] == message["raw_message"][0])
    check("GIF 图片组件被替换", comps[1] is not message["raw_message"][1])
    check("emoji 原组件原样保留", comps[2] is message["raw_message"][2])
    check("emoji 之后是追加的合成图组件", comps[3].get("type") == "image" and comps[3].get("data", "") == "")
    check("PNG 组件原样保留", comps[4] is message["raw_message"][3])

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

    print()
    if failures:
        print(f"自检未通过：{len(failures)} 项 — {failures}")
        return 1
    print("全部自检通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
