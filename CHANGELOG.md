# 更新日志

## v1.0.0 (2026-09-11)

- 首个版本：GIF 动图分镜（视觉增强），插件 id `github.elmeir.gif-storyboard`。
- 只处理 GIF：仅对 `GIF87a` / `GIF89a` 魔数的多帧动图替换，PNG/JPEG/WebP 等其他格式零接触，单帧 GIF 视为静态图放行。
- 挂载宿主 `chat.receive.before_process` 钩子（BLOCKING / EARLY），把消息里 `image` 组件中的 GIF 动图
  原位替换为多帧合成网格图。
- **表情包组件三策略**（`emoji_strategy`，默认 `bypass` 旁路识别）：宿主表情链路"识别与存储共用同一个
  字节流"——直接替换会让 `emoji_manager` 把合成图落盘 `data/emoji`（Images 表 EMOJI 类型）甚至被收藏
  发出。`bypass`（默认）：原表情组件原样保留（表情库存原始 GIF，收藏/发送保真），另追加一个 content 为空
  的 image 组件携带合成网格图，视觉链路借道图片描述完成多帧识别，麦麦上下文同时出现 [表情包] 与
  [图片: 多帧描述]；`replace`：直接替换（识别最直接，代价是表情库污染）；`off`：不处理。
- 帧抽取：按时序均匀抽样（含首尾帧），`max_frames` 控制上限（默认 9 = 3x3 网格）；
  单帧 GIF 视为静态图直接放行（`min_frames` 可调）。
- 合成图：网格排布尽量接近正方形，帧间隔 `grid_gap`（白底）、单帧格子边长上限
  `max_cell_size`；每帧左上角可绘制序号（`draw_frame_index`）帮助视觉模型理解播放顺序；
  输出 jpeg（默认，体积小）/ png 可选。
- 替换后同步写回新字节的 SHA-256 hash；宿主 `ByteComponent` 在二进制非空时本就会
  重算 hash，双保险保持字典自洽，后续识别与落库自动使用合成图。
- 合成结果按「原图哈希 + 参数指纹」做内存 LRU 缓存（上限 64 条），重复出现的
  同一 GIF 不再重复解码；配置热更新时自动清空。
- 失败兜底：解码失败 / 合成异常 / 单帧 GIF 一律原样放行，绝不阻塞消息链
  （`ErrorPolicy.SKIP`，钩子超时 10s）。
- 不改动组件 `data`（content）字段——宿主以 content 非空判定"已识别过"，
  乱填会导致图片跳过视觉模型识别。
