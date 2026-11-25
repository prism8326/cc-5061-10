# 内容感知视频尺寸重构工具（时空接缝裁剪 / Spatiotemporal Seam Carving）

通过在「时间 × 行 × 列」三维体素上求解 **Boykov–Kolmogorov 最大流/最小割**，
删除或插入低代价接缝来改变视频宽高比。能量不只看单帧梯度，还融合
**光流运动、前后向遮挡一致性与时序连续约束**，因此相邻帧的接缝是联合求解、
真实关联的，而不是把上一帧坐标机械复制到下一帧。

## 目录结构

| 路径 | 作用 |
| --- | --- |
| `seamcarver/ext/bk_core.h` | 纯 C 的 BK 最大流核心（无 Python 依赖，可单独编译检查） |
| `seamcarver/ext/bkflow.c` | numpy 绑定，编译为 `seamcarver/bkflow.so` |
| `seamcarver/maxflow_wrapper.py` | 数组形状/类型校验的 Python 封装 |
| `seamcarver/seam_solver.py` | 终端容量、墙钉、时序容量与接缝提取（图的构造） |
| `seamcarver/energy.py` | 空间梯度 + 运动 + 遮挡能量、镜头切换检测、硬保护区域 |
| `seamcarver/flow.py` | OpenCV Farneback 光流，纯 numpy 块匹配回退 |
| `seamcarver/carve.py` | 接缝增删、前向/逆向逐行坐标映射（原始素材始终可恢复） |
| `seamcarver/engine.py` | 有状态引擎：求解、编辑接缝、保护区域、缩放、诊断告警 |
| `seamcarver/seam_store.py` | **分块、版本化**的能量/接缝存储：块 + 时间邻接边界记录 + 完整性校验 |
| `seamcarver/migration.py` | 旧扁平能量/路径表 → 分块存储的迁移器、作者锁保留/冲突、跨块重求自动路径 |
| `seamcarver/reconstruct.py` | 从分块存储**跨帧回溯**重建接缝：有限时域代价图、镜头切换截断、缺失邻帧上报、闪烁度量、lazy vs import-rebuild 对照 |
| `seamcarver/exporter.py` | 用重建路径导出，并明确标出须**回读原片**的片段（绝不静默继承旧能量） |
| `seamcarver/webapp.py` `web/` `serve.py` | 运动分析播放器（HTTP 接口 + 前端），图形入口驱动同一内核 |
| `seamcarver/synth.py` `evaluate.py` | 合成真值片段与量化评估 |
| `tests/test_core.py` | 含 2^N 暴力枚举 oracle 的 7 项核心正确性测试 |
| `tests/test_migration.py` | 17 项存储/迁移/重建契约测试（作者锁、跨块重求、缺失边界、同接缝保真、闪烁对照） |
| `tests/test_webapp.py` | 5 项播放器 HTTP 集成测试（状态、清空重建、编辑版本、导出回读、异常回显） |
| `scripts/run_evaluation.py` | 三场景 × 多缩放比的完整量化评估 + 迁移/重建证据 |
| `scripts/verify_migration.py` | **净目录**端到端验收：从像素生成旧表→迁移→重建→导出，记录接缝轨迹与画幅证据 |

## 1. 构建 C 扩展

求解器是 C 扩展，必须先编译：

```bash
bash scripts/build_ext.sh
```

脚本需要 gcc 与 numpy。若系统缺少 `Python.h`（无 python3-dev 且无 root），
可把匹配的 CPython 头目录放到 `PYHEADER_DIR`，脚本会在缺失时自动下载并
`./configure` 生成 `pyconfig.h`：

```bash
PYHEADER_DIR=/path/to/pyheaders bash scripts/build_ext.sh
```

成功后生成 `seamcarver/bkflow.so`，并打印 `import ok`。

依赖：`numpy`、`opencv-python`（光流与视频读写；缺失时回退到纯 numpy 块匹配，
速度明显更慢）。

## 2. 启动界面（运动分析播放器）

```bash
python -m seamcarver.serve --host 127.0.0.1 --port 8000
```

然后浏览器打开 <http://127.0.0.1:8000/>。可选 `--scenario`
取值 `moving_marker`、`fast_occlusion`、`shot_cut`。

界面左右两栏同时显示**接缝叠加图**与**真实重构后的视频帧**，支持：

- 播放/暂停、逐帧拖动、帧率调节，连续回放而非只展示最好看的一帧；
- 视图切换：接缝 / 重构结果 / 原始帧 / 能量图；
- **点击或拖动接缝顶点**编辑某一帧接缝，内核只硬锁该帧的垂直窗口，
  相邻帧通过时序容量重新关联（不是坐标复制）；
- **Shift+拖拽**框选硬保护区域，区域被赋予无穷 unary 屏障，接缝不可穿越；
- 单步删除/插入接缝、一键缩放到目标宽度、复位；
- 底部连续性曲线展示相邻帧平均接缝位移，红线标记镜头切换；
- 内核状态面板显示图割流量、求解耗时与告警；任何失败都以
  `{"ok": false, "status": ...}` 回传并显示，不会静默吞掉。

所有按钮都经 `/api/*` 调用同一个引擎内核，前端只渲染状态，保证所见即所得。

新增的接缝数据转换/重建控件（第二行工具栏）：

- **按分块存储重建接缝**（`/api/reconstruct`）：从版本化分块存储跨帧回溯重建路径，
  返回闪烁指标与须回读帧；
- **清空接缝并播放重建**（`/api/clear_reconstruct`）：清掉缓存的接缝路径，再从存储
  重建并立即播放——播放中看到的连续性来自**实际重建路径**，不是缓存的旧解；
- **校验重构片段与边界**（`/api/verify`）：用真实重构路径逐帧删缝，核对帧数/宽度
  一致与相邻帧边界位移，失败原因回显到状态区；
- **导出（标注须回读段）**（`/api/export`）：用重建路径导出，并列出哪些帧因
  时间边界缺失/块过期须**回读原片**。

> 图形界面是本工具的主入口；命令行仅作辅助（见下）。

## 2.1 接缝数据转换与分块存储（迁移契约）

旧的每帧能量表是整段扁平数组，没有块、版本或时间邻接概念。迁移到
`SeamStore` 后：

- **分块读取**：片段按 `chunk_size` 帧切块，每块与每条时间边界记录都在
  `manifest.json` 里带内容哈希；读到缺失/过期块时在 `BlockRead.missing` 里
  **上报缺口**，绝不补零（补零会让接缝偏爱缺口、并悄悄切断帧间对应）。
- **版本化**：`manifest.revision` 每次写盘自增，导出器会拒绝用旧版本重建的路径
  导出（`export revision != reconstruction revision` 直接报错）。
- **时间邻接 + 对象保护字段**：每个边界记录存前向光流幅值与镜头切换标志；每帧
  记录存 `protected`（对象保护）与逐行 `lock`（`-1` 自动、`>=0` 作者锁定），
  以及 `origin`/`provenance`（作者/自动/导入、重算/继承）。
- **迁移器**（`migrate_legacy`）：从像素**重算**能量并重建时间边界；作者锁
  （`lock>=0`）保留为 `ORIGIN_AUTHOR` 且后续改帧时**报告新冲突**而不静默重解；
  自动/缓存路径对被改能量**跨块重求**。没有像素时能量标记为“继承”，且所有
  时间缺口都列为缺失（不伪造零代价边）。
- **重建策略对照**（`compare_strategies`）：`lazy`（逐块求解、块边界无时间耦合）
  会在块边产生接缝闪烁；`import_rebuild`（导入时读全有限时域代价图一次求解）
  保持跨块平滑。`strategy="auto"` 按该工程的几何/分块/边界结构实测两种实现的
  闪烁并选更平滑者（结果按结构指纹缓存，交互编辑不会每次重测）；也可显式指定。
  评估中三场景 `lazy` 的块边跳变均达 26–31px，`import_rebuild` 为 0，故后者胜出。
- **时域范围**（`horizon`）：默认 `0` = 整段有限片段，依赖**只在镜头切换处**截断
  （符合迁移契约）。正的 `horizon` 会每 N 帧额外截断一次，属显式取舍，会如实计入
  闪烁，绝不用来掩盖不连续。图结构两种取值下都已按整段分配，故有限 `horizon`
  不省内存、只会引入伪跳变，因此默认整段。
- **镜头切换 ≠ 缺失邻帧**：切点处时间依赖被**截断**（两侧独立求解，合法）；而
  缺失边界必须上报为须回读，二者语义严格区分。
- **中途切镜截断依赖**：重建在切点断开，量出的闪烁指标（`flicker_index`）即导出
  影片的接缝闪烁。

## 3. 运行测试

```bash
python tests/test_core.py        # 7 项核心正确性测试
python tests/test_migration.py   # 17 项存储/迁移/重建契约测试
python tests/test_webapp.py      # 5 项播放器 HTTP 集成测试
```

`test_grid_mincut_matches_bruteforce` 在 2×2×3 小图上枚举全部 2^N 标注，
逐一核对割代价与流量；其余覆盖硬保护屏障、8 连通接缝合法性、静态场景时序
一致性、快速运动下的接缝换边、像素级坐标往返、插入增宽与镜头切换切断时序边。

`test_migration.py` 覆盖迁移契约：分块往返与版本化、缺失块上报且不补零、
作者锁保留并报告新冲突、自动路径跨块重求、import-rebuild 在块边优于 lazy、
缺失边界须回读且不当作零代价（且只标记两侧邻帧、不误报整块）、镜头切换合法截断、
闪烁度量、转换后重建**同一接缝**、auto 策略按结构指纹缓存、导出使用新路径并
拒绝旧版本，以及无像素旧表全部缺口上报。

### 净目录验收（执行证据）

```bash
python scripts/verify_migration.py --scenario moving_marker --out /tmp/seam_clean
```

脚本拒绝写入非空目录（除非 `--force`），从**像素现场生成**旧扁平表、迁移、
跨帧重建、导出，全程不读手改数据。它把每帧的接缝轨迹（列均值/极值/内容哈希）
与原始/导出画幅写入 `evidence.json`，并断言：能量确为重算而非继承、每条时间
邻接都被重建、转换后重建的接缝与旧解**逐行相同**、导出宽度 = 原宽 − 1、
删掉一条边界后导出器把**该缺口两侧**标为须回读。任一断言失败即非零退出。

## 3.1 命令行（辅助）

图形播放器为主入口；CLI 只镜像同一内核，供批处理：

```bash
# 启动图形界面（主入口）
python -m seamcarver serve --scenario moving_marker

# 旧扁平表 -> 分块存储（--frames 可省略：自动读取旧表旁的 .frames.npz 边车）
python -m seamcarver migrate --legacy old.npz --target stores/clip --chunk-size 8

# 从存储重建并比较策略 / 校验完整性
python -m seamcarver reconstruct --store stores/clip --horizon 12 --compare
python -m seamcarver verify --store stores/clip

# 内置迁移/重建演示（含 lazy vs import-rebuild 闪烁对照）
python -m seamcarver demo --out /tmp/seam_demo

# 净目录端到端验收（从像素生成 → 迁移 → 重建 → 导出，断言同接缝与回读标记）
python scripts/verify_migration.py --scenario moving_marker --out /tmp/seam_clean
```

## 4. 运行量化评估

```bash
python scripts/run_evaluation.py --flow farneback --out outputs/eval
```

对移动标记、快速遮挡、镜头切换三个合成片段，分别在 0.85 / 0.75 / 0.65 缩放比下
测量：目标物体相对宽度、轨迹二阶差分抖动、接缝换边次数、镜头切换数、
求解与总耗时，并给出**等比缩放对照**（仅作对照，不参与核心处理）。
产物写入 `outputs/eval/`：`report.json`、`report.tsv`，以及原始、重构、
接缝叠加、等比缩放四组 mp4；`migration.json` 记录每个场景的分块存储版本、
重建闪烁、策略对照与导出回读图。

### 当前评估结论（难例刻意保留）

- `moving_marker`、`shot_cut`：物体宽度保持在真值 ±12% 内、抖动 ≤ 1.5px，
  判定 `acceptable=True`；镜头切换处时序容量被置零，接缝允许在切点重新决策。
- `fast_occlusion`（高速 + 遮挡条）：在大幅缩放下物体宽度被压缩、抖动超阈值，
  判定 `acceptable=False`。这是真实难例，**不隐藏、不挑帧**，用于标注该算法
  当前可接受的缩放范围；等比缩放对照同步保留。

## 关键设计与可恢复性

- 原始片段在 `CarveProject` 中永不修改；每步记录 `SeamStep`，前向/逆向逐行
  坐标映射由步骤日志重建，任意中间几何与原始素材都可恢复（测试含像素级往返）。
- 无向 Potts 边用一对互为 XOR 配对的弧（双向各容量 c）表示；终端墙钉把接缝
  限制在合法列区间。孤儿节点入栈即清 parent，收养失败则离开搜索树并退出活跃
  环形队列，这些不变量保证 BK 收敛且割与暴力枚举一致。
- 光流在原始几何上计算一次，之后通过逆向映射重采样到每个缩放几何，避免逐步
  重跑光流；纯 Python 块匹配仅在无 OpenCV 时使用，速度远慢于 Farneback。

## 已知边界

逐接缝（seam-by-seam）缩放每删一条缝都重新构图求解，大幅缩放（如 0.65）在
24 帧片段上需要数十次图割，是当前主要耗时；评估全量运行约数分钟到十分钟量级。
后续可改为一次求解并行移除多条互不相交接缝（multi-seam）来成比例减少图割次数。
