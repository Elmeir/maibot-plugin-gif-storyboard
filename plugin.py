"""GIF 动图分镜插件（GIF Storyboard）

在麦麦把 GIF 动图交给视觉模型之前，把动画的各个帧合成到同一张静态图上
（均匀抽样 + 网格排布，可绘制帧序号），让视觉模型能"看到"动图的内容与时序——
否则大多数视觉模型遇到 image/gif 只能看到第一帧。

实现方式（宿主源码级挂钩，不改宿主一行代码）：
- 订阅宿主 ``chat.receive.before_process`` 钩子：该钩点允许改写序列化消息，
  且宿主会用改写后的字典重建消息对象（PluginMessageUtils 反序列化），
  因此后续的图片识别（ImageManager.get_image_description）与图片落库
  全部自动使用合成后的图片；
- 定位 image / emoji 组件的 ``binary_data_base64``，检测到 GIF 魔数
  （GIF87a / GIF89a）且为多帧动画时，用 Pillow 解帧、均匀抽样
  ``max_frames`` 帧、拼成网格静态图后原位替换；
- 只对 GIF 魔数动图动手，其余格式零接触；普通图片组件（type=image，
  群友发的 GIF 动图主要走这里）默认处理；表情包组件（type=emoji）默认
  走 bypass 旁路识别——宿主表情链路"识别与存储共用同一个字节流"，直接
  替换会让合成图落盘表情库（data/emoji）甚至被收藏发出，因此旁路模式下
  原表情组件原样保留（表情库存原始 GIF，收藏/发送完全保真），另在其后
  追加一个携带合成网格图的 image 组件，让视觉链路借道完成多帧识别；
- 替换后无需手工改 hash：宿主 ByteComponent 在二进制非空时会按新字节
  重算 SHA-256（插件同时主动写回，保持字典自洽）。
"""

import asyncio
import base64
import hashlib
import io
import logging
import math
from collections import OrderedDict
from typing import Any, Dict, List, Literal, Optional

from PIL import Image, ImageDraw, ImageFont

from maibot_sdk import Field, HookHandler, MaiBotPlugin, PluginConfigBase
from maibot_sdk.types import ErrorPolicy, HookMode, HookOrder

logger = logging.getLogger("plugin.gif_frames")

SUPPORTED_CONFIG_VERSION = "1.0.0"

GIF_MAGICS = (b"GIF87a", b"GIF89a")
"""GIF 文件头魔数。"""

MANAGED_COMPONENT_TYPES = ("image", "emoji")
"""会被处理的组件类型（type 字段）。"""

MERGE_CACHE_MAX_ENTRIES = 64
"""合成结果内存缓存上限（键=原图哈希+参数指纹，超出按最旧淘汰）。"""


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
    draw_frame_index: bool = Field(
        default=True,
        description="在每帧左上角绘制序号，帮助视觉模型理解播放顺序",
        json_schema_extra={
            "label": "绘制帧序号",
            "hint": "每帧左上角标 1、2、3…，视觉模型更容易按时序描述动画",
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
                ("merge", "draw_frame_index", True),
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
        order=HookOrder.EARLY,
        name="gif_frame_merge",
        description="把 GIF 动图的各帧合成一张网格静态图，替换原图交给视觉模型",
        timeout_ms=10000,
        error_policy=ErrorPolicy.SKIP,
    )
    async def handle_before_process(self, message: Any = None, **kwargs: Any) -> Dict[str, Any]:
        """入站钩子：消息里的 GIF 动图原位替换为多帧合成图。"""
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
        for comp in components:
            results = await self._process_component(comp)
            if len(results) != 1 or results[0] is not comp:
                changed = True
            new_components.extend(results)

        if not changed:
            return {"action": "continue"}

        new_message = dict(message)
        new_message["raw_message"] = new_components
        return {"action": "continue", "modified_kwargs": {**kwargs, "message": new_message}}

    async def _process_component(self, comp: Any) -> List[Any]:
        """处理单个消息组件，返回替换后的组件列表（bypass 策略可能追加组件）。

        注意：绝不改动组件的 data（content）字段——宿主以 content 非空
        判定"已识别过"，乱填会让图片跳过视觉模型识别。
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

        if comp_type == "emoji":
            strategy = str(self._opt("merge", "emoji_strategy", "bypass") or "bypass").lower()
            if strategy == "bypass":
                # 原表情组件原样保留（表情库存原始 GIF，收藏/发送保真），
                # 追加一个携带合成网格图的 image 组件旁路识别：视觉链路走
                # 图片描述（落 data/images，不碰表情库），麦麦上下文里会同时
                # 出现 [表情包] 与 [图片: 多帧动画描述]。
                ghost = {
                    "type": "image",
                    "data": "",  # content 留空，交给宿主视觉链路生成描述
                    "hash": merged_hash,
                    "binary_data_base64": merged_b64,
                }
                logger.info(
                    "[GIF合成] 表情包旁路识别：原表情保留，追加合成图 %.1fKB", len(merged) / 1024
                )
                return [comp, ghost]
            # replace：直接替换表情二进制（合成图会进表情库，由配置自担）
            new_comp = dict(comp)
            new_comp["binary_data_base64"] = merged_b64
            new_comp["hash"] = merged_hash
            return [new_comp]

        # image：原位替换二进制
        new_comp = dict(comp)
        new_comp["binary_data_base64"] = merged_b64
        new_comp["hash"] = merged_hash
        return [new_comp]

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
            logger.warning("[GIF合成] 帧合成失败，原图原样放行: %s", exc)
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
            indices = self._sample_frame_indices(n_frames, count)

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

        canvas = Image.new("RGB", (cols * cell_w + gap * (cols + 1), rows * cell_h + gap * (rows + 1)), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        font = self._load_font(max(12, min(cell_w, cell_h) // 10)) if draw_index else None

        for i, frame in enumerate(frames):
            row, col = divmod(i, cols)
            # 各帧尺寸可能略有差异：等比缩到格子内并居中
            fw, fh = frame.width, frame.height
            frame_scale = min(cell_w / fw, cell_h / fh, 1.0)
            if frame_scale < 1.0:
                frame = frame.resize((max(1, round(fw * frame_scale)), max(1, round(fh * frame_scale))), Image.LANCZOS)
            x = gap + col * (cell_w + gap) + (cell_w - frame.width) // 2
            y = gap + row * (cell_h + gap) + (cell_h - frame.height) // 2

            canvas.paste(frame, (x, y), frame)  # 第三参数=alpha 遮罩
            if font is not None:
                draw.text((x + 4, y + 2), str(i + 1), font=font, fill=(17, 17, 17))

        buf = io.BytesIO()
        if output_format == "png":
            canvas.save(buf, "PNG")
        else:
            canvas.save(buf, "JPEG", quality=jpeg_quality)

        merged = buf.getvalue()
        logger.info(
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
    def _load_font(size: int):
        """加载序号字体：Pillow>=10.1 可指定字号，旧版退回内置字体。"""
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()

    # ── 生命周期 ────────────────────────────────────────────────────────

    async def on_load(self) -> None:
        logger.info(
            "[GIF合成] 插件已加载 | 总开关: %s | 处理图片: %s | 表情包策略: %s | 最多 %s 帧 | 输出: %s",
            self._enabled(),
            bool(self._opt("merge", "process_image", True)),
            self._opt("merge", "emoji_strategy", "bypass"),
            self._opt("merge", "max_frames", 9),
            self._opt("output", "output_format", "jpeg"),
        )

    async def on_unload(self) -> None:
        self._merge_cache.clear()
        logger.info("[GIF合成] 插件已卸载")

    async def on_config_update(self, scope: str, config_data: Dict[str, Any], version: str) -> None:
        self._merge_cache.clear()
        logger.info("[GIF合成] 配置已更新（scope=%s version=%s），合成缓存已清空", scope, version)


def create_plugin() -> MaiBotPlugin:
    return GifStoryboardPlugin()
