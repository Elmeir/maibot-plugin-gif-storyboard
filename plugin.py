"""GIF 动图分镜插件（GIF Storyboard）

在麦麦把 GIF 动图交给视觉模型之前，把动画各帧按时序抽帧并合成到同一张
网格静态图上（带帧序号与说明条），让视觉模型看到动画的完整内容与时序，
而不是只看到第一帧。

实现（挂在宿主钩子上，不改宿主一行代码）：
- ``chat.receive.before_process``：检测 GIF 魔数，多帧动画——图片组件
  临时替换为合成图（原图字节暂存插件内部登记表，按"组件索引+hash"登记）；
  表情组件按 emoji_strategy 配置处理（bypass 追加 ghost 组件 / replace / off）；
- ``chat.receive.after_process``：落库前用暂存字节恢复被替换的组件、
  回收表情 ghost，并调度后台描述搬运任务；
- 描述搬运：等宿主 VLM 写出合成图描述后，用 ctx.db 写回原图 hash 的
  Images 记录，宿主视觉占位刷新器自动回填进麦麦上下文；
  同图重发走"已就绪描述快速路径"，宿主直接跳过识别。

只对 GIF 魔数动图动手；任何环节失败一律原样放行，绝不阻塞消息链。
"""

import asyncio
import base64
import hashlib
import io
import math
from collections import OrderedDict
from typing import Any, Dict, List, Literal, Optional

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageStat

from maibot_sdk import Field, HookHandler, MaiBotPlugin, PluginConfigBase
from maibot_sdk.types import ErrorPolicy, HookMode, HookOrder

SUPPORTED_CONFIG_VERSION = "1.0.0"

GIF_MAGICS = (b"GIF87a", b"GIF89a")
"""GIF 文件头魔数。"""

MANAGED_COMPONENT_TYPES = ("image", "emoji")
"""会被处理的组件类型（type 字段）。"""

MERGE_CACHE_MAX_ENTRIES = 64
"""合成结果内存缓存上限（键=原图哈希+参数指纹，超出按最旧淘汰）。"""

RELOCATE_TIMEOUT_SECONDS = 600.0
"""描述搬运任务的最长等待时间（宿主后台 VLM 识别合成图的宽限期）。"""

RELOCATE_POLL_INTERVAL_SECONDS = 5.0
"""描述搬运任务的轮询间隔。"""

ORIGINAL_BYTES_CACHE_MAX_ENTRIES = 16
"""替换窗口内暂存原图字节的缓存上限（key=orig_hash）。

before_process 把组件二进制替换为合成图（让宿主识别多帧），after_process
在落库前用这里暂存的原图字节恢复组件。条目生命周期只有一次消息处理
（几秒），并发 GIF 数量通常为个位数；超出上限时放弃替换（原样放行），
避免原图字节因缓存淘汰而丢失导致合成图落库。
"""

DHASH_DUPLICATE_DISTANCE = 6
"""感知哈希（64bit）汉明距离低于该值时，视为与上一选中帧"几乎相同"。"""


# ─── 配置模型 ────────────────────────────────────────────────────────────────


class PluginSectionConfig(PluginConfigBase):
    """插件基础配置。"""

    __ui_label__ = "插件"
    __ui_icon__ = "package"
    __ui_order__ = 0

    enabled: bool = Field(
        default=True,
        description="是否启用插件（总开关；关闭后 GIF 一律原样交给视觉模型）",
        json_schema_extra={
            "label": "插件总开关",
            "hint": "关闭后 GIF 一律原样交给视觉模型，钩子直接放行",
        },
    )
    config_version: str = Field(
        default=SUPPORTED_CONFIG_VERSION,
        description="配置版本（勿改）",
        json_schema_extra={"hidden": True, "disabled": True},
    )


class MergeSectionConfig(PluginConfigBase):
    """帧合成行为配置。"""

    __ui_label__ = "帧合成"
    __ui_icon__ = "film"
    __ui_order__ = 1

    process_image: bool = Field(
        default=True,
        description="处理普通图片组件中的 GIF 动图",
        json_schema_extra={
            "label": "处理图片",
            "hint": "普通图片（type=image）里的 GIF 动图会被合成多帧网格图",
        },
    )
    emoji_strategy: Literal["bypass", "replace", "off"] = Field(
        default="bypass",
        description=(
            "表情包组件（type=emoji）的处理策略。bypass（推荐）：原表情组件原样保留——表情库存原始"
            " GIF、可正常收藏与发送；另在消息里追加一个携带合成网格图的图片组件旁路识别，"
            "视觉模型通过它看全多帧，麦麦的上下文会同时出现 [表情包] 与 [图片: 多帧描述]。"
            "replace：直接替换表情二进制——识别最直接，但合成图会被宿主落盘表情库"
            "（data/emoji）甚至被收藏发出，原始 GIF 丢失。off：不处理表情包"
        ),
        json_schema_extra={
            "label": "表情包策略",
            "options": {
                "bypass": {"label": "旁路识别（推荐）", "description": "表情库存原始 GIF（保真），另用隐藏合成图让视觉模型看全多帧"},
                "replace": {"label": "直接替换", "description": "识别最直接，但表情库会存进合成图、可能被收藏发出，原始 GIF 丢失"},
                "off": {"label": "不处理", "description": "表情包一律原样，识别退回宿主原生行为（GIF 取首帧）"},
            },
            "hint": "bypass=保真+多帧识别（推荐）｜replace=会污染表情库｜off=不处理",
        },
    )
    max_frames: int = Field(
        default=9,
        ge=2,
        le=25,
        description="最多抽取的帧数（多帧动画均匀抽样；9=3x3 网格，兼顾信息量与视觉模型 token 开销）",
        json_schema_extra={
            "label": "最多帧数",
            "hint": "超出上限的动画按时序均匀抽样；9=3x3 网格，帧越多图越大、token 越贵",
        },
    )
    min_frames: int = Field(
        default=2,
        ge=2,
        le=10,
        description="动画至少包含多少帧才触发合成（单帧 GIF 等价静态图，无需处理）",
        json_schema_extra={
            "label": "最少帧数",
            "hint": "帧数不足该值的 GIF 视为静态图，原样放行",
        },
    )
    adaptive_frames: bool = Field(
        default=True,
        description=(
            "根据源动画的运动量与帧数，自动在『最少帧数~最多帧数』之间决定实际抽帧数："
            "画面几乎静止/循环重复的简单动画少抽帧（省 token），运动丰富、帧数多的动画多抽帧（保证证据）"
        ),
        json_schema_extra={
            "label": "自适应帧数",
            "hint": "按动画运动量自动调节实际帧数；关闭则始终抽『最多帧数』",
        },
    )
    draw_frame_index: bool = Field(
        default=True,
        description="在每帧左上角绘制序号，帮助视觉模型理解播放顺序",
        json_schema_extra={
            "label": "绘制帧序号",
            "hint": "每帧左上角标 1、2、3…，视觉模型更容易按时序描述动画",
        },
    )
    clean_ghost_components: bool = Field(
        default=True,
        description="识别完成后恢复原图组件（图片 GIF）并回收表情旁路的 ghost 组件，防止合成图进入聊天记录/WebUI",
        json_schema_extra={
            "label": "落库前恢复原图",
            "hint": "依赖 after_process 钩子 + 描述搬运；关闭后消息会以合成图落库并在 WebUI 中显示九宫格",
        },
    )
    storyboard_caption: bool = Field(
        default=True,
        description="在合成图顶部绘制一行说明文字，引导视觉模型把网格图理解为一段动画而不是一张拼图",
        json_schema_extra={
            "label": "动画分镜说明条",
            "hint": "修复视觉模型把合成图描述成『九宫格动漫截图』的问题；环境中无中文字体时自动改用英文说明",
        },
    )


class OutputSectionConfig(PluginConfigBase):
    """合成图输出配置。"""

    __ui_label__ = "输出图像"
    __ui_icon__ = "image"
    __ui_order__ = 2

    output_format: Literal["jpeg", "png"] = Field(
        default="jpeg",
        description="合成图编码格式：jpeg 体积小（推荐）；png 无损、体积较大",
        json_schema_extra={
            "label": "输出格式",
            "hint": "jpeg 体积小省 token（推荐）；png 无损但明显更大",
        },
    )
    jpeg_quality: int = Field(
        default=85,
        ge=50,
        le=95,
        description="JPEG 质量（仅输出格式为 jpeg 时生效）",
        json_schema_extra={
            "label": "JPEG 质量",
            "hint": "50~95，85 兼顾清晰度与体积；仅 jpeg 格式生效",
        },
    )
    max_cell_size: int = Field(
        default=480,
        ge=128,
        le=1024,
        description="网格中每帧格子的最长边像素上限（帧原图更大时等比缩小）",
        json_schema_extra={
            "label": "单帧格子边长上限",
            "hint": "帧原图超过该尺寸会等比缩小；480 在多数视觉模型上清晰度与开销均衡",
        },
    )
    grid_gap: int = Field(
        default=4,
        ge=0,
        le=32,
        description="帧与帧之间的间隔像素（白底）",
        json_schema_extra={
            "label": "帧间隔（像素）",
            "hint": "网格帧之间的白色间隔，0~32",
        },
    )


class GifStoryboardConfig(PluginConfigBase):
    """插件根配置。"""

    plugin: PluginSectionConfig = Field(default_factory=PluginSectionConfig, json_schema_extra={"label": "插件"})
    merge: MergeSectionConfig = Field(default_factory=MergeSectionConfig, json_schema_extra={"label": "帧合成"})
    output: OutputSectionConfig = Field(default_factory=OutputSectionConfig, json_schema_extra={"label": "输出图像"})


# ─── 主插件 ──────────────────────────────────────────────────────────────────


class GifStoryboardPlugin(MaiBotPlugin):
    """GIF 动图分镜：让视觉模型看见动画的每一帧。"""

    config_model = GifStoryboardConfig

    def __init__(self) -> None:
        super().__init__()
        # (原图 sha256 + 参数指纹) -> 合成图 bytes；OrderedDict 做 LRU 淘汰
        self._merge_cache: "OrderedDict[str, bytes]" = OrderedDict()
        # 替换窗口内暂存的原图字节（key=orig_hash），after_process 恢复组件后删除
        self._original_bytes: "OrderedDict[str, bytes]" = OrderedDict()
        # 组件索引 -> (合成图 hash, 原图 hash)：after_process 按"索引 + hash"
        # 精确恢复被替换的组件。不要把恢复信息写进组件本身——宿主钩子之间
        # 会做序列化往返，组件对象上的未知字段会被丢弃。
        self._replaced_index: "OrderedDict[int, tuple[str, str]]" = OrderedDict()
        # 表情旁路 ghost（key=合成图 hash -> 原表情 hash）：after_process 回收
        self._emoji_ghosts: "Dict[str, str]" = {}
        # 描述搬运映射（key=合成图 hash -> (image_type, 原图/原表情 hash)）
        self._ghost_records: "Dict[str, tuple[str, str]]" = {}
        # 已在跑的描述搬运任务（按合成图 hash 去重）
        self._relocate_tasks: "set[str]" = set()
        # 数据库能力不可用时的告警去重标志（避免每轮轮询刷屏）
        self._db_warned: bool = False

    @staticmethod
    def _b64_decode(data: Any) -> bytes:
        """组件二进制 base64 解码，失败返回空字节串。"""
        if not isinstance(data, str) or not data:
            return b""
        try:
            return base64.b64decode(data)
        except Exception:
            return b""

    # ── 配置读取 ────────────────────────────────────────────────────────

    def _opt(self, section: str, key: str, default: Any = None) -> Any:
        """安全读取插件配置项。"""
        try:
            return getattr(getattr(self.config, section, None), key, default)
        except Exception:
            return default

    def _enabled(self) -> bool:
        return bool(self._opt("plugin", "enabled", True))

    def _component_managed(self, comp_type: str) -> bool:
        """组件类型在当前配置下是否会被处理（image 默认开，emoji 默认 bypass）。"""
        if comp_type not in MANAGED_COMPONENT_TYPES:
            return False
        if comp_type == "image":
            return bool(self._opt("merge", "process_image", True))
        return str(self._opt("merge", "emoji_strategy", "bypass") or "bypass").lower() != "off"

    def _settings_fingerprint(self) -> str:
        """合成参数指纹，参与缓存键：参数变了缓存自动失效。"""
        return "|".join(
            str(
                self._opt(section, key, default)
            )
            for section, key, default in (
                ("merge", "max_frames", 9),
                ("merge", "min_frames", 2),
                ("merge", "adaptive_frames", True),
                ("merge", "draw_frame_index", True),
                ("merge", "storyboard_caption", True),
                ("output", "output_format", "jpeg"),
                ("output", "jpeg_quality", 85),
                ("output", "max_cell_size", 480),
                ("output", "grid_gap", 4),
            )
        )

    # ── 钩子：入站消息改写 ──────────────────────────────────────────────

    @HookHandler(
        "chat.receive.before_process",
        mode=HookMode.BLOCKING,
        order=HookOrder.LATE,
        name="gif_frame_merge",
        description="把 GIF 动图各帧合成网格图，以 ghost 组件形式交给视觉链路（原图保留）",
        timeout_ms=10000,
        error_policy=ErrorPolicy.SKIP,
    )
    async def handle_before_process(self, message: Any = None, **kwargs: Any) -> Dict[str, Any]:
        """入站钩子：消息里的 GIF 动图原位替换为多帧合成图。

        使用 LATE 槽位：排在消息防抖类插件（在 NORMAL 槽阻塞窗口并合并消息）
        之后，看到的是防抖放行后的最终消息——防抖窗口内并入的第 2..N 条消息
        的 GIF 也能被分镜处理，且被防抖 abort 的消息不会再触发本插件
        （避免登记永远不会被回收的 ghost）。
        """
        if not self._enabled() or not isinstance(message, dict):
            return {"action": "continue"}

        components = message.get("raw_message")
        if not isinstance(components, list) or not components:
            return {"action": "continue"}

        relevant = False
        for comp in components:
            if isinstance(comp, dict) and self._component_managed(str(comp.get("type") or "").strip().lower()):
                relevant = True
                break
        if not relevant:
            return {"action": "continue"}

        new_components: List[Any] = []
        changed = False
        for idx, comp in enumerate(components):
            results = await self._process_component(comp)
            if len(results) == 1 and results[0] is comp:
                new_components.append(comp)
                continue
            changed = True

            comp_hash = str(comp.get("hash") or "").strip()
            result = results[0]
            result_hash = str(result.get("hash") or "").strip()

            if len(results) == 1 and comp_hash and result_hash and result_hash != comp_hash:
                # image 替换模式：按组件索引登记，after_process 据此恢复原图。
                # 登记信息放在插件内部而非组件字段——宿主钩子间的序列化往返
                # 会丢弃组件对象上的未知字段。
                orig_bytes = self._b64_decode(comp.get("binary_data_base64"))
                if orig_bytes:
                    self._original_bytes[comp_hash] = orig_bytes
                    self._original_bytes.move_to_end(comp_hash)
                    while len(self._original_bytes) > ORIGINAL_BYTES_CACHE_MAX_ENTRIES:
                        self._original_bytes.popitem(last=False)
                self._replaced_index[idx] = (result_hash, comp_hash)
                self._ghost_records[result_hash] = ("image", comp_hash)
            elif len(results) == 2:
                # 表情 bypass：第二个组件是 ghost，after_process 阶段回收
                ghost_hash = str(results[1].get("hash") or "").strip()
                if ghost_hash:
                    self._emoji_ghosts[ghost_hash] = comp_hash
                    self._ghost_records[ghost_hash] = ("emoji", comp_hash)
            new_components.extend(results)

        # 防御：索引登记只服务于紧随其后的一条消息，避免异常路径残留累积
        while len(self._replaced_index) > ORIGINAL_BYTES_CACHE_MAX_ENTRIES * 4:
            self._replaced_index.popitem(last=False)

        if not changed:
            return {"action": "continue"}

        new_message = dict(message)
        new_message["raw_message"] = new_components
        return {"action": "continue", "modified_kwargs": {**kwargs, "message": new_message}}

    @HookHandler(
        "chat.receive.after_process",
        mode=HookMode.BLOCKING,
        order=HookOrder.LATE,
        name="gif_frame_ghost_cleanup",
        description="回收 GIF 分镜 ghost 组件，防止合成图进入聊天记录，并调度多帧描述搬运",
        timeout_ms=10000,
        error_policy=ErrorPolicy.SKIP,
    )
    async def handle_after_process(self, message: Any = None, **kwargs: Any) -> Dict[str, Any]:
        """after 钩子：宿主 process() 已完成识别调度，此刻做两件事——

        1. 替换模式：按"组件索引 + hash"精确匹配 before_process 替换过的组件，
           用暂存的原图字节恢复（二进制与 hash），随后的消息落库按原图 hash
           保存文件与记录，图片库/WebUI 全部保真；此时原图记录尚未被宿主识别
           （不存在首帧描述），搬运任务把合成图的多帧描述写到原图 hash 名下
           后，刷新器即可安全回填；
        2. 表情 bypass：按 hash 回收追加的 ghost 组件，多帧描述由后台任务搬到
           原表情 hash 名下。
        """
        if not isinstance(message, dict):
            return {"action": "continue"}
        if not bool(self._opt("merge", "clean_ghost_components", True)):
            self._replaced_index.clear()
            self._emoji_ghosts.clear()
            return {"action": "continue"}

        components = message.get("raw_message")
        if not isinstance(components, list) or not components:
            return {"action": "continue"}

        kept: List[Any] = []
        handled: List[tuple[str, str]] = []  # (合成图 hash, 原图/原表情 hash)
        for idx, comp in enumerate(components):
            comp_hash = str(comp.get("hash") or "").strip() if isinstance(comp, dict) else ""

            if idx in self._replaced_index:
                merged_hash, orig_hash = self._replaced_index[idx]
                if comp_hash != merged_hash:
                    # 组件列表被其他环节改动导致索引错位：放弃恢复（保持合成图），
                    # 避免把原图字节写错到别的组件上。
                    self.ctx.logger.warning(
                        "[GIF合成] 替换组件索引错位（hash 不匹配），放弃恢复：idx=%s hash=%s",
                        idx,
                        comp_hash[:12],
                    )
                    self._replaced_index.pop(idx, None)
                    kept.append(comp)
                    continue
                self._replaced_index.pop(idx, None)
                restored = self._restore_replaced_component(comp, orig_hash)
                # 恢复失败（暂存字节缺失）时组件保持合成图状态，但保留组件本体
                kept.append(restored if restored is not None else comp)
                handled.append((merged_hash, orig_hash))
                continue

            if comp_hash and comp_hash in self._emoji_ghosts:
                orig_hash = self._emoji_ghosts.pop(comp_hash, "")
                handled.append((comp_hash, orig_hash))
                continue

            kept.append(comp)

        if not handled:
            return {"action": "continue"}

        # 恢复/回收全部完成后统一清理暂存的原图字节（同 hash 多组件共用一份）
        for merged_hash, orig_hash in handled:
            image_type = str(self._ghost_records.pop(merged_hash, ("image", ""))[0] or "image")
            self._original_bytes.pop(orig_hash, None)
            self._spawn_relocate_task(merged_hash, image_type, orig_hash)
        self.ctx.logger.info(
            "[GIF合成] 已处理 %d 个组件（恢复原图/回收合成图），多帧描述将在后台搬运",
            len(handled),
        )

        new_message = dict(message)
        new_message["raw_message"] = kept
        return {"action": "continue", "modified_kwargs": {**kwargs, "message": new_message}}

    def _restore_replaced_component(self, comp: Dict[str, Any], orig_hash: str) -> Optional[Dict[str, Any]]:
        """把替换成合成图的组件恢复为原图（二进制与 hash），失败返回 None。"""
        original_bytes = self._original_bytes.get(orig_hash) if orig_hash else None
        if original_bytes is None:
            self.ctx.logger.warning(
                "[GIF合成] 暂存的原图字节缺失，组件将保持合成图状态 hash=%s",
                str(comp.get("hash") or "")[:12],
            )
            return None
        restored = dict(comp)
        restored["binary_data_base64"] = base64.b64encode(original_bytes).decode("ascii")
        restored["hash"] = orig_hash
        restored["data"] = ""  # 保持空占位，等待多帧描述回填
        return restored

    def _spawn_relocate_task(self, merged_hash: str, image_type: str, orig_hash: str) -> None:
        """按合成图 hash 去重后启动描述搬运后台任务。"""
        if merged_hash in self._relocate_tasks:
            return
        self._relocate_tasks.add(merged_hash)
        try:
            asyncio.get_running_loop().create_task(
                self._relocate_description(merged_hash, image_type, orig_hash)
            )
        except RuntimeError:
            self._relocate_tasks.discard(merged_hash)

    async def _relocate_description(self, merged_hash: str, image_type: str, orig_hash: str) -> None:
        """等待宿主 VLM 完成合成图识别，把多帧描述搬到原图/原表情 hash 名下。

        - image：等落库流程建好原图记录（替换模式原图不参与宿主识别，
          记录由 after_process 之后的落库创建）；
        - emoji：等 emoji_manager 写入原表情记录后覆盖为多帧描述；
        - 超时放弃时行为退化为宿主原生识别（原图/首帧描述），不影响消息链。
        """
        try:
            loop = asyncio.get_running_loop()
            deadline = loop.time() + RELOCATE_TIMEOUT_SECONDS
            while loop.time() < deadline:
                if await self._try_relocate(merged_hash, orig_hash, image_type):
                    return
                await asyncio.sleep(RELOCATE_POLL_INTERVAL_SECONDS)

            self.ctx.logger.warning(
                "[GIF合成] 描述搬运超时放弃（宿主识别未完成或记录缺失）：merged=%s orig=%s",
                merged_hash[:12],
                orig_hash[:12],
            )
        finally:
            self._relocate_tasks.discard(merged_hash)

    async def _try_relocate(self, merged_hash: str, orig_hash: str, image_type: str) -> bool:
        """尝试搬运一次多帧描述，成功（或已搬运过）返回 True。

        Args:
            merged_hash: 合成网格图 hash（描述来源）。
            orig_hash: 原图/原表情 hash（描述落点）。
            image_type: Images 记录类型（image / emoji）。
        """
        source = await self._db_get_image_record(merged_hash, "image")
        if not (source and source.get("vlm_processed")):
            return False
        source_desc = str(source.get("description") or "").strip()
        if not source_desc:
            return False

        target = await self._db_get_image_record(orig_hash, image_type)
        if target is None:
            return False  # 记录尚未由落库流程创建，下轮轮询再试
        if str(target.get("description") or "").strip() == source_desc:
            return True  # 已搬运过（重复轮询/重复触发时直接视为完成）

        await self._db_update_description(orig_hash, image_type, source_desc)
        self.ctx.logger.info(
            "[GIF合成] 多帧动画描述已搬运至原图记录 %s（%s）",
            orig_hash[:12],
            image_type,
        )
        return True

    def _warn_db_unavailable(self, reason: Any) -> None:
        """数据库能力不可用时只告警一次，避免轮询刷屏。"""
        if self._db_warned:
            return
        self._db_warned = True
        self.ctx.logger.warning(
            "[GIF合成] database.query 能力不可用（多帧描述无法搬运回原图，"
            "视觉识别会退化为只看首帧）：%s。请确认 _manifest.json 的 capabilities "
            '已声明 "database.query"，并重载插件',
            reason,
        )

    async def _db_get_image_record(self, image_hash: str, image_type: str) -> Optional[Dict[str, Any]]:
        """按 hash + 类型查询宿主 Images 表记录，兼容不同的返回包装结构。"""
        try:
            result = await self.ctx.db.query(
                model_name="Images",
                query_type="get",
                filters={"image_hash": image_hash, "image_type": image_type},
                limit=1,
                single_result=True,
            )
        except Exception as exc:  # noqa: BLE001 数据库不可达时静默放弃本轮
            self._warn_db_unavailable(exc)
            return None

        if not isinstance(result, dict):
            return None
        if result.get("success") is False:
            self._warn_db_unavailable(result.get("error") or "database.query 返回失败")
            return None
        if "description" in result:
            return result
        for key in ("data", "result", "items", "records"):
            inner = result.get(key)
            if isinstance(inner, dict) and "description" in inner:
                return inner
            if isinstance(inner, list) and inner and isinstance(inner[0], dict):
                return inner[0]
        return None

    async def _db_update_description(self, image_hash: str, image_type: str, description: str) -> None:
        """把多帧描述写入原图 hash 的 Images 记录（刷新器即可安全回填，无首帧固化竞态）。"""
        try:
            await self.ctx.db.query(
                model_name="Images",
                query_type="update",
                data={"description": description, "vlm_processed": True},
                filters={"image_hash": image_hash, "image_type": image_type},
            )
        except Exception as exc:  # noqa: BLE001 更新失败不影响消息链
            self.ctx.logger.warning("[GIF合成] 更新原图描述失败 hash=%s: %s", image_hash[:12], exc)

    async def _process_component(self, comp: Any) -> List[Any]:
        """处理单个消息组件，返回替换后的组件列表（bypass 策略可能追加组件）。

        组件 data（content）默认保持为空、交给宿主视觉链路；唯一例外是
        多帧描述已就绪的重复 GIF——把已就绪描述直接写入 data，宿主见
        content 非空跳过识别，planner 上下文立即拿到多帧描述。
        """
        if not isinstance(comp, dict):
            return [comp]

        comp_type = str(comp.get("type") or "").strip().lower()
        if not self._component_managed(comp_type):
            return [comp]

        b64_data = comp.get("binary_data_base64")
        if not isinstance(b64_data, str) or not b64_data:
            return [comp]

        try:
            raw = base64.b64decode(b64_data)
        except Exception:
            return [comp]
        if not self._is_gif(raw):
            return [comp]

        merged = await self._merge_with_cache(raw)
        if merged is None:
            return [comp]  # 单帧 GIF / 解码失败 / 合成异常：原样放行

        merged_b64 = base64.b64encode(merged).decode("ascii")
        merged_hash = hashlib.sha256(merged).hexdigest()
        orig_hash = str(comp.get("hash") or "").strip() or hashlib.sha256(raw).hexdigest()

        if comp_type == "emoji":
            strategy = str(self._opt("merge", "emoji_strategy", "bypass") or "bypass").lower()
            if strategy == "bypass":
                relocated = await self._lookup_ready_storyboard_description(orig_hash, merged_hash, "emoji")
                if relocated:
                    # 该 GIF 的多帧描述此前已搬运就绪：直接写入表情组件文本，
                    # 宿主对 content 非空的组件跳过识别——零额外识别，
                    # 也避免重发时缓存命中的旧首帧描述被固化进上下文。
                    new_comp = dict(comp)
                    new_comp["data"] = f"[表情包: {relocated}]"
                    self.ctx.logger.info(
                        "[GIF合成] 命中已就绪的分镜描述，直接写入表情组件文本 hash=%s", orig_hash[:12]
                    )
                    return [new_comp]
                # 原表情组件原样保留（表情库存原始 GIF，收藏/发送保真），
                # 追加一个携带合成网格图的 ghost image 组件旁路识别；该组件在
                # after_process 阶段被回收，多帧描述由后台任务搬到原表情 hash 名下。
                ghost = self._build_ghost(merged_hash, merged_b64)
                self.ctx.logger.info(
                    "[GIF合成] 表情包旁路识别：原表情保留，追加合成图 %.1fKB", len(merged) / 1024
                )
                return [comp, ghost]
            # replace：直接替换表情二进制（合成图会进表情库，由配置自担）
            new_comp = dict(comp)
            new_comp["binary_data_base64"] = merged_b64
            new_comp["hash"] = merged_hash
            return [new_comp]

        relocated = await self._lookup_ready_storyboard_description(orig_hash, merged_hash, "image")
        if relocated:
            # 该 GIF 的多帧描述此前已搬运就绪：直接写入图片组件文本，
            # 宿主对 content 非空的组件跳过识别——零额外识别，上下文
            # 立即拿到多帧描述（重发场景无延迟）。
            new_comp = dict(comp)
            new_comp["data"] = f"[图片：{relocated}]"
            self.ctx.logger.info(
                "[GIF合成] 命中已就绪的分镜描述，直接写入图片组件文本 hash=%s", orig_hash[:12]
            )
            return [new_comp]

        # image：替换模式——组件二进制与 hash 暂时替换为合成图，宿主识别的
        # 就是多帧网格图（识别次数 2→1，且原图记录不存在首帧描述，无固化
        # 竞态）；after_process 在落库前用暂存的原图字节恢复组件，原图文件
        # 与图片记录全部保真。替换登记（索引/暂存字节）由 handle_before_process
        # 的外层循环完成。
        new_comp = dict(comp)
        new_comp["binary_data_base64"] = merged_b64
        new_comp["hash"] = merged_hash
        self.ctx.logger.info(
            "[GIF合成] 图片组件已替换为合成图 %.1fKB（落库前恢复原图）", len(merged) / 1024
        )
        return [new_comp]

    async def _lookup_ready_storyboard_description(self, orig_hash: str, merged_hash: str, image_type: str) -> str:
        """若该 GIF 的多帧描述此前已搬运就绪，返回可直接写入组件文本的描述。

        命中条件（全部满足）：
        - 合成图记录（IMAGE 类型）已完成识别且有非空描述——搬运完成后
          merged_hash 名下的记录即持久化的多帧描述；同图重发时帧抽样与
          JPEG 编码确定，merged_hash 一致，可稳定命中；
        - 原图/原表情记录存在且 no_file_flag=False（文件在库中，跳过识别
          不会影响落库与展示）。

        未命中返回空字符串，走 ghost + 搬运流程。
        """
        merged = await self._db_get_image_record(merged_hash, "image")
        if not (merged and merged.get("vlm_processed")):
            return ""
        description = str(merged.get("description") or "").strip()
        if not description:
            return ""
        target = await self._db_get_image_record(orig_hash, image_type)
        if target is None or target.get("no_file_flag"):
            return ""
        return description

    @staticmethod
    def _build_ghost(merged_hash: str, merged_b64: str) -> Dict[str, str]:
        """构造携带合成网格图的 ghost 组件（content 留空交给宿主视觉链路）。"""
        return {
            "type": "image",
            "data": "",
            "hash": merged_hash,
            "binary_data_base64": merged_b64,
        }

    @staticmethod
    def _is_gif(data: bytes) -> bool:
        """按文件头魔数判断是否 GIF。"""
        return len(data) >= 6 and data[:6] in GIF_MAGICS

    # ── 合成（带缓存） ──────────────────────────────────────────────────

    async def _merge_with_cache(self, raw: bytes) -> Optional[bytes]:
        """合成 GIF 帧网格图，命中缓存时直接复用。失败返回 None（原样放行）。"""
        cache_key = f"{hashlib.sha256(raw).hexdigest()}|{self._settings_fingerprint()}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            merged = await asyncio.to_thread(self._merge_gif_frames, raw)
        except Exception as exc:  # noqa: BLE001 合成失败绝不能阻塞消息链
            self.ctx.logger.warning("[GIF合成] 帧合成失败，原图原样放行: %s", exc)
            return None

        if merged is None:
            return None

        self._cache_put(cache_key, merged)
        return merged

    def _cache_get(self, key: str) -> Optional[bytes]:
        cached = self._merge_cache.get(key)
        if cached is not None:
            self._merge_cache.move_to_end(key)  # 刷新热度
        return cached

    def _cache_put(self, key: str, value: bytes) -> None:
        self._merge_cache[key] = value
        while len(self._merge_cache) > MERGE_CACHE_MAX_ENTRIES:
            self._merge_cache.popitem(last=False)

    def _merge_gif_frames(self, data: bytes) -> Optional[bytes]:
        """把多帧 GIF 均匀抽样后拼成网格静态图。

        Returns:
            合成图 bytes；帧数不足 min_frames（视为静态图）时返回 None。
        Raises:
            解码失败等异常向上抛出，由调用方兜底。
        """
        max_frames = max(2, int(self._opt("merge", "max_frames", 9)))
        min_frames = max(2, int(self._opt("merge", "min_frames", 2)))
        draw_index = bool(self._opt("merge", "draw_frame_index", True))
        output_format = str(self._opt("output", "output_format", "jpeg") or "jpeg").lower()
        jpeg_quality = int(self._opt("output", "jpeg_quality", 85))
        max_cell = max(64, int(self._opt("output", "max_cell_size", 480)))
        gap = max(0, int(self._opt("output", "grid_gap", 4)))

        with Image.open(io.BytesIO(data)) as im:
            n_frames = int(getattr(im, "n_frames", 1) or 1)
            if n_frames < min_frames:
                return None

            count = min(n_frames, max_frames)
            if bool(self._opt("merge", "adaptive_frames", True)):
                adaptive = self._adaptive_frame_count(im, n_frames, count)
                if adaptive != count:
                    self.ctx.logger.info(
                        "[GIF合成] 自适应帧数: %d 帧（运动量与源帧数调节，上限 %d）", adaptive, count
                    )
                count = max(min_frames, min(count, adaptive))

            indices = self._pick_frame_indices(im, n_frames, count)

            frames = []
            for idx in indices:
                im.seek(idx)
                # 转 RGBA：保留透明通道，粘贴时按 alpha 混合到白底
                frames.append(im.convert("RGBA"))

        if not frames:
            return None

        # 网格布局：尽量接近正方形
        count = len(frames)
        cols = math.ceil(math.sqrt(count))
        rows = math.ceil(count / cols)

        # 单帧格子尺寸：取各帧最大宽高，超出上限等比缩小
        cell_w = max(frame.width for frame in frames)
        cell_h = max(frame.height for frame in frames)
        scale = min(1.0, max_cell / cell_w, max_cell / cell_h)
        if scale < 1.0:
            cell_w = max(1, round(cell_w * scale))
            cell_h = max(1, round(cell_h * scale))

        # 顶部说明条：极简文案 + 字号自适应，保证任何画布宽度都完整显示
        caption_text = ""
        caption_font = None
        caption_h = 0
        if bool(self._opt("merge", "storyboard_caption", True)):
            canvas_w = cols * cell_w + gap * (cols + 1)
            target_w = max(60, canvas_w - gap * 2 - 4)
            for size in range(18, 11, -2):
                cand_font, has_cjk = self._load_caption_font(size)
                cand_text = self._caption_text(count, has_cjk)
                caption_font, caption_text = cand_font, cand_text
                try:
                    bbox = cand_font.getbbox(cand_text)
                    if (bbox[2] - bbox[0]) <= target_w:
                        break  # 当前字号能放下，就用它
                except Exception:
                    break
            try:
                bbox = caption_font.getbbox(caption_text)
                caption_h = (bbox[3] - bbox[1]) + 10
            except Exception:
                caption_h = 0

        canvas = Image.new(
            "RGB",
            (cols * cell_w + gap * (cols + 1), caption_h + rows * cell_h + gap * (rows + 1)),
            (255, 255, 255),
        )
        draw = ImageDraw.Draw(canvas)
        if caption_h and caption_font is not None:
            draw.text((gap + 2, 4), caption_text, font=caption_font, fill=(17, 17, 17))
        font = self._load_caption_font(max(14, min(cell_w, cell_h) // 9))[0] if draw_index else None
        stroke_w = max(1, round(getattr(font, "size", 14) / 10)) if font is not None else 0

        for i, frame in enumerate(frames):
            row, col = divmod(i, cols)
            # 各帧尺寸可能略有差异：等比缩到格子内并居中
            fw, fh = frame.width, frame.height
            frame_scale = min(cell_w / fw, cell_h / fh, 1.0)
            if frame_scale < 1.0:
                frame = frame.resize((max(1, round(fw * frame_scale)), max(1, round(fh * frame_scale))), Image.LANCZOS)
            x = gap + col * (cell_w + gap) + (cell_w - frame.width) // 2
            y = caption_h + gap + row * (cell_h + gap) + (cell_h - frame.height) // 2

            canvas.paste(frame, (x, y), frame)  # 第三参数=alpha 遮罩
            if font is not None:
                # 黑字白描边：浅色/深色背景上都清晰
                draw.text(
                    (x + 4, y + 2),
                    str(i + 1),
                    font=font,
                    fill=(17, 17, 17),
                    stroke_width=stroke_w,
                    stroke_fill=(255, 255, 255),
                )

        buf = io.BytesIO()
        if output_format == "png":
            canvas.save(buf, "PNG")
        else:
            canvas.save(buf, "JPEG", quality=jpeg_quality)

        merged = buf.getvalue()
        self.ctx.logger.info(
            "[GIF合成] %d/%d 帧 -> %dx%d 网格图，%.1fKB -> %.1fKB（%s）",
            count,
            n_frames,
            canvas.width,
            canvas.height,
            len(data) / 1024,
            len(merged) / 1024,
            output_format,
        )
        return merged

    @staticmethod
    def _sample_frame_indices(n_frames: int, count: int) -> List[int]:
        """按时序均匀抽取 count 个帧索引（含首尾帧）。"""
        if count >= n_frames:
            return list(range(n_frames))
        if count <= 1:
            return [0]
        indices: List[int] = []
        for i in range(count):
            idx = round(i * (n_frames - 1) / (count - 1))
            if not indices or idx != indices[-1]:
                indices.append(idx)
        return indices

    @staticmethod
    def _frame_sharpness(frame: "Image.Image") -> float:
        """帧清晰度近似值：灰度边缘强度 RMS，运动模糊帧显著偏低。"""
        try:
            edges = frame.convert("L").filter(ImageFilter.FIND_EDGES)
            return float(sum(ImageStat.Stat(edges).rms))
        except Exception:
            return 0.0

    @staticmethod
    def _dhash(frame: "Image.Image") -> int:
        """感知哈希（dhash，8x8=64bit）：结构相似的画面之间距离很小。"""
        small = frame.convert("L").resize((9, 8), Image.LANCZOS)
        pixels = list(small.getdata())
        bits = 0
        for row in range(8):
            base = row * 9
            for col in range(8):
                bits = (bits << 1) | (1 if pixels[base + col] > pixels[base + col + 1] else 0)
        return bits

    @staticmethod
    def _hamming(a: int, b: int) -> int:
        """两个感知哈希的汉明距离。"""
        return bin(a ^ b).count("1")

    def _adaptive_frame_count(self, im: "Image.Image", n_frames: int, cap: int) -> int:
        """根据运动量自适应决定实际抽帧数（介于 min_frames 与 cap 之间）。

        粗抽 ≤24 个样本帧，以相邻帧感知哈希（dhash）的平均汉明距离衡量运动量：
        平均距离 ≤8 视为几乎静止/循环重复（取最少帧数），≥24 视为运动丰富
        （取最多帧数），中间线性插值。
        """

        def clamp01(v: float) -> float:
            return max(0.0, min(1.0, v))

        min_frames = max(2, int(self._opt("merge", "min_frames", 2)))
        if cap <= min_frames:
            return cap

        sample_idx = self._sample_frame_indices(n_frames, min(n_frames, 24))
        dists: List[int] = []
        prev_dhash = None
        for idx in sample_idx:
            try:
                im.seek(idx)
                dhash = self._dhash(im)
            except Exception:
                continue
            if prev_dhash is not None:
                dists.append(self._hamming(prev_dhash, dhash))
            prev_dhash = dhash

        avg_dist = sum(dists) / len(dists) if dists else 0.0
        motion = clamp01((avg_dist - 8.0) / 16.0)
        return min(cap, min_frames + round((cap - min_frames) * motion))

    @classmethod
    def _pick_frame_indices(cls, im: "Image.Image", n_frames: int, count: int) -> List[int]:
        """分段清晰度抽帧：首尾帧固定保留，中间时间轴均分 count-2 段。

        每段先取清晰度最高的帧；若它与上一选中帧几乎相同（感知哈希距离过小），
        改选段内与上一帧差异最大的一帧——让"新元素入画"（如手出现、姿态突变）
        这类变化帧有机会被保留，而不是连续选中相似姿态。

        等间隔抽帧容易命中快速运动的模糊中间帧，视觉模型对模糊帧的描述
        常与实际内容偏差很大；清晰度与差异度结合，既保持时序覆盖，
        又尽量覆盖动画中的每次变化。
        """
        if count >= n_frames:
            return list(range(n_frames))
        if count < 3:
            return cls._sample_frame_indices(n_frames, count)

        middle = [
            c
            for c in cls._sample_frame_indices(n_frames, min(n_frames - 2, (count - 2) * 3))
            if 0 < c < n_frames - 1
        ]
        if not middle:
            return cls._sample_frame_indices(n_frames, count)

        try:
            im.seek(0)
            prev_dhash = cls._dhash(im)
        except Exception:
            prev_dhash = 0

        picked = [0]
        step = len(middle) / (count - 2)
        for seg_i in range(count - 2):
            lo = min(int(round(seg_i * step)), len(middle) - 1)
            hi = max(min(int(round((seg_i + 1) * step)), len(middle)), lo + 1)
            best_idx, best_score, best_dhash = middle[lo], -1.0, 0
            far_idx, far_dist, far_dhash = middle[lo], -1, 0
            for cand in middle[lo:hi]:
                try:
                    im.seek(cand)
                    gray = im.convert("L")
                except Exception:
                    continue
                score = cls._frame_sharpness(gray)
                dhash = cls._dhash(gray)
                dist = cls._hamming(prev_dhash, dhash)
                if score > best_score:
                    best_idx, best_score, best_dhash = cand, score, dhash
                if dist > far_dist:
                    far_idx, far_dist, far_dhash = cand, dist, dhash

            chosen_idx, chosen_dhash = best_idx, best_dhash
            if cls._hamming(prev_dhash, best_dhash) < DHASH_DUPLICATE_DISTANCE:
                # 段内最清晰帧与上一选中帧几乎相同：改选与上一帧差异最大的帧
                chosen_idx, chosen_dhash = far_idx, far_dhash
            if chosen_idx != picked[-1]:
                picked.append(chosen_idx)
            prev_dhash = chosen_dhash
        picked.append(n_frames - 1)

        picked = sorted(set(picked))
        if len(picked) < count:
            for cand in cls._sample_frame_indices(n_frames, count):
                if cand not in picked:
                    picked.append(cand)
                    if len(picked) >= count:
                        break
            picked.sort()
        return picked[:count]

    @staticmethod
    def _load_font(size: int):
        """加载序号字体：Pillow>=10.1 可指定字号，旧版退回内置字体。"""
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()

    _CJK_FONT_CANDIDATES: "tuple[str, ...]" = (
        # Windows
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        # Linux（麦麦常见部署环境）
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        # macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
    )

    @classmethod
    def _load_caption_font(cls, size: int) -> "tuple[Any, bool]":
        """加载说明条字体：优先系统中文字体，找不到退回 Pillow 内置字体（仅拉丁）。

        Returns:
            (字体对象, 是否中文字体)。
        """
        for path in cls._CJK_FONT_CANDIDATES:
            try:
                return ImageFont.truetype(path, size=size), True
            except Exception:  # noqa: BLE001 字体缺失/格式不支持一律尝试下一个
                continue
        return cls._load_font(size), False

    @staticmethod
    def _caption_text(count: int, has_cjk: bool) -> str:
        """说明条文案：极简锚点 + 时序连播提示，帮助视觉模型把跨帧动作连成因果。"""
        if has_cjk:
            return f"动图分镜 · 共 {count} 帧（按序号连播）"
        return f"Storyboard: {count} frames, play in order"

    # ── 生命周期 ────────────────────────────────────────────────────────

    async def on_load(self) -> None:
        self.ctx.logger.info(
            "[GIF合成] 插件已加载 | 总开关: %s | 处理图片: %s | 表情包策略: %s | 最多 %s 帧 | 输出: %s",
            self._enabled(),
            bool(self._opt("merge", "process_image", True)),
            self._opt("merge", "emoji_strategy", "bypass"),
            self._opt("merge", "max_frames", 9),
            self._opt("output", "output_format", "jpeg"),
        )
        await self._check_db_access()

    async def _check_db_access(self) -> None:
        """启动自检：确认 manifest 的 capabilities 已包含 database.query。

        描述搬运依赖 ``ctx.db``；宿主 AuthorizationManager 只对 manifest 里
        声明过的能力签发令牌，未声明时所有 ``database.query`` 调用都会被拒绝，
        插件的视觉增强效果会在"描述回填"这一步静默失效。这里主动探一次，
        把问题在日志里暴露出来。
        """
        try:
            result = await self.ctx.db.query(
                model_name="Images",
                query_type="get",
                filters={"image_hash": "__gif_storyboard_probe__", "image_type": "image"},
                limit=1,
                single_result=True,
            )
        except Exception as exc:  # noqa: BLE001 仅做能力可用性提示
            self._warn_db_unavailable(exc)
            return
        if isinstance(result, dict) and result.get("success") is False:
            self._warn_db_unavailable(result.get("error") or "database.query 返回失败")
            return
        self.ctx.logger.info("[GIF合成] 数据库能力自检通过（database.query 可用，描述搬运就绪）")

    async def on_unload(self) -> None:
        self._merge_cache.clear()
        self._original_bytes.clear()
        self._replaced_index.clear()
        self._emoji_ghosts.clear()
        self._ghost_records.clear()
        self._relocate_tasks.clear()
        self._db_warned = False
        self.ctx.logger.info("[GIF合成] 插件已卸载")

    async def on_config_update(self, scope: str, config_data: Dict[str, Any], version: str) -> None:
        self._merge_cache.clear()
        self.ctx.logger.info("[GIF合成] 配置已更新（scope=%s version=%s），合成缓存已清空", scope, version)


def create_plugin() -> MaiBotPlugin:
    return GifStoryboardPlugin()
