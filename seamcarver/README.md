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
| `seamcarver/webapp.py` `web/` `serve.py` | 运动分析播放器（HTTP 接口 + 前端） |
| `seamcarver/synth.py` `evaluate.py` | 合成真值片段与量化评估 |
| `tests/test_core.py` | 含 2^N 暴力枚举 oracle 的 7 项正确性测试 |
| `scripts/run_evaluation.py` | 三场景 × 多缩放比的完整量化评估 |

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

## 3. 运行测试

```bash
python tests/test_core.py
```

其中 `test_grid_mincut_matches_bruteforce` 在 2×2×3 小图上枚举全部 2^N 标注，
逐一核对割代价与流量；其余覆盖硬保护屏障、8 连通接缝合法性、静态场景时序
一致性、快速运动下的接缝换边、像素级坐标往返、插入增宽与镜头切换切断时序边。

## 4. 运行量化评估

```bash
python scripts/run_evaluation.py --flow farneback --out outputs/eval
```

对移动标记、快速遮挡、镜头切换三个合成片段，分别在 0.85 / 0.75 / 0.65 缩放比下
测量：目标物体相对宽度、轨迹二阶差分抖动、接缝换边次数、镜头切换数、
求解与总耗时，并给出**等比缩放对照**（仅作对照，不参与核心处理）。
产物写入 `outputs/eval/`：`report.json`、`report.tsv`，以及原始、重构、
接缝叠加、等比缩放四组 mp4。

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
