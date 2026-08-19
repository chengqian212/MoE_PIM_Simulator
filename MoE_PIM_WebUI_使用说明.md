# MoE-PIM Simulator Web 使用说明

> 本文档用于说明 MoE-PIM Simulator Web 可视化页面的启动方式、页面结构、主要功能与常见问题。  
> 文档中的路径均使用相对路径或通用占位写法，不包含个人电脑用户名、绝对目录、设备信息或私人环境名称。

---

# 1. Web 页面用途

该网页用于展示 MoE-PIM Simulator 中已经生成的：

- 硬件空间划分结果
- Weight-Cube / Plane 的物理映射结果
- 单个真实 Token 的 MoE 路由与执行过程
- 单层调度时间线
- 16 个 Sub-Cube 的 3D 执行状态
- 一个完整 Token 的 58 层推理过程
- 多 Token Workload 延迟统计

整体可以理解为：

```text
空间规划
   ↓
静态 Mapping
   ↓
真实 Token 路由
   ↓
单层调度
   ↓
58 层完整 Token 调度
   ↓
多 Token 结果分析
```

---

# 2. 项目目录要求

启动网页前，项目中至少需要存在以下内容：

```text
MoE_PIM_Simulator/
├─ webui/
│  ├─ backend/
│  │  ├─ main.py
│  │  ├─ trace_api.py
│  │  ├─ schedule_api.py
│  │  ├─ token_schedule_api.py
│  │  └─ workload_api.py
│  │
│  └─ frontend/
│     ├─ package.json
│     └─ src/
│        ├─ App.jsx
│        └─ components/
│           ├─ TokenSimulator.jsx
│           ├─ ActiveMappingPanel.jsx
│           ├─ LayerTimeline.jsx
│           ├─ ExecutionCube3D.jsx
│           ├─ FullTokenPlaybackPanel.jsx
│           └─ FullTokenRunner.jsx
│
├─ results/
│  └─ mappings/
│     └─ <mapping_result>.json
│
└─ deepseek_r1_trace/
   └─ ...
      └─ Chinese-SimpleQA/
```

其中：

- `results/mappings/`：保存当前静态 Mapping 结果。
- `Chinese-SimpleQA/`：保存真实 MoE Router Trace。
- `webui/backend/`：FastAPI 后端。
- `webui/frontend/`：React + Vite 前端。

---

# 3. 启动后端

先进入项目根目录：

```text
MoE_PIM_Simulator/
```

运行：

```bash
python -m uvicorn webui.backend.main:app --reload --port 8000
```

启动成功后，终端通常会显示：

```text
Uvicorn running on http://127.0.0.1:8000
```

可以打开：

```text
http://127.0.0.1:8000/docs
```

如果能够看到 FastAPI Swagger 页面，说明后端已经正常启动。

---

# 4. 启动前端

保持后端终端不关闭，再打开一个新的终端。

进入：

```bash
cd webui/frontend
```

如果是第一次运行前端，先安装依赖：

```bash
npm install
```

然后启动：

```bash
npm run dev
```

终端通常会显示：

```text
Local: http://localhost:5173/
```

在浏览器打开：

```text
http://localhost:5173/
```

即可进入网页。

---

# 5. 推荐启动顺序

建议每次按下面顺序启动：

```text
1. 进入项目根目录
2. 启动 FastAPI 后端
3. 保持后端运行
4. 打开第二个终端
5. 进入 webui/frontend
6. 启动 Vite 前端
7. 浏览器访问 localhost:5173
```

对应命令：

```bash
# 终端 1
python -m uvicorn webui.backend.main:app --reload --port 8000
```

```bash
# 终端 2
cd webui/frontend
npm run dev
```

---

# 6. 页面结构

当前 Web 页面主要分为三个模块：

```text
01 Cube 总览
02 Token 模拟
03 结果分析
```

---

# 7. Cube 总览

点击左侧：

```text
01 Cube 总览
```

该页面用于查看整个硬件空间的划分和静态 Mapping。

当前硬件中：

```text
N = 4
Sub-Cube 数量 = 4 × 4 = 16
```

因此页面中的 16 个 3D 柱体分别表示：

```text
SC-0
SC-1
...
SC-15
```

注意：

```text
N = 4
```

并不是表示存在 `4 × 4 × 4 = 64` 个 Sub-Cube。

这里的 Sub-Cube 数量为：

```text
N² = 16
```

Sub-Cube 的竖直方向表示 Depth。

---

# 8. Cube 总览操作

常见交互方式：

```text
鼠标左键拖动：旋转
鼠标滚轮：缩放
点击 Sub-Cube：选中
```

选中某个 Sub-Cube 后，可以查看该 Sub-Cube 的基本信息，例如：

```text
Used Planes
Depth
Empty Planes
Weight-Cubes
Shared Weight
Gate
Up
Down
```

---

# 9. 查看 Sub-Cube 内部 Plane

进入某个 Sub-Cube 后，可以通过 `z` 选择具体深度：

```text
z = 0
z = 1
...
z = D - 1
```

每个深度对应一个 Physical Plane。

页面可用于查看：

```text
当前 Plane 上放了哪些 Weight-Cube
属于哪个 Layer
属于哪个 Expert
矩阵类型是 gate / up / down
所在 Slot
是否旋转
```

不同矩阵使用不同颜色区分。

---

# 10. Token 模拟页面

点击左侧：

```text
02 Token 模拟
```

该页面用于模拟一个真实 Token 从 Router Trace 到实际硬件调度的全过程。

当前页面经过精简后，主要保留：

```text
Token 选择
Layer 选择
Current Route
Physical Mapping
Layer Timeline
3D Execution
Full Token Execution
```

---

# 11. 选择真实 Token

页面从 Chinese-SimpleQA Trace 中读取真实 Token。

可以选择：

```text
Dataset
Category
Random Real Token
```

Category 可根据当前 Trace 数据中的分类进行选择。

点击：

```text
Random Real Token
```

后，会随机读取一个真实 Token。

页面会显示简化后的来源信息，例如：

```text
Source File
Segment
Token Index
```

---

# 12. Layer 选择

一个完整 Token 包含：

```text
58 个 MoE Layer
```

网页中的项目层编号为：

```text
L0 ~ L57
```

可以使用：

```text
上一层
下一层
Layer Slider
```

切换当前查看的 Layer。

如果页面同时显示 Trace Layer，则：

```text
Project Layer 0 ~ 57
```

对应原始 Trace 中的：

```text
Trace Layer 3 ~ 60
```

---

# 13. Current Route

当前 Layer 会显示：

```text
Top-8 Routed Experts
+
Shared Expert E256
```

例如：

```text
E17
E21
E100
E103
E118
E169
E250
E24
+
E256 Shared
```

其中：

```text
Top-8 Routed Expert
```

来自真实 Router Trace。

Shared Expert：

```text
E256
```

不参与 Top-8 Router 选择，但当前层始终参与计算。

因此每层实际参与：

```text
8 个 Routed Expert
+
1 个 Shared Expert
=
9 个 Expert
```

每个 Expert 包含：

```text
gate
up
down
```

因此每层共有：

```text
9 × 3 = 27 个 Weight-Cube Task
```

---

# 14. Full 58-Layer Route

完整的 58 层 Router Expert 列表默认折叠。

需要查看时点击：

```text
View Full 58-Layer Route
```

可以查看：

```text
L0
L1
...
L57
```

每层的 Top-8 Expert 和 Shared Expert。

由于该信息较长，默认不展开，以避免影响主页面阅读。

---

# 15. Physical Mapping

Physical Mapping 默认可以折叠查看。

页面会以紧凑表格形式显示：

```text
Expert
Gate
Up
Down
Check
```

例如：

```text
Expert   Gate          Up            Down
E17      SC-10/z2      SC-6/z935     SC-10/z2
E21      SC-6/z0       SC-12/z936    SC-6/z0
...
E256     SC-0/z0       SC-14/z1395   SC-0/z0
```

每个矩阵还会显示：

```text
Physical Plane
Slot
```

例如：

```text
P17 · Slot 34
```

---

# 16. Mapping Check

Physical Mapping 中还会检查当前 Expert 的基本 Mapping 约束。

其中：

```text
GD
```

表示：

```text
gate / down 是否位于同一个 Sub-Cube
```

```text
GU
```

表示：

```text
gate / up 是否分散到不同 Sub-Cube
```

如果当前 9 个 Expert 全部满足检查，会显示：

```text
Mapping Check ✓ 9/9
```

---

# 17. Layer Timeline

Layer Timeline 是 Token Simulation 页面中的核心部分之一。

它展示当前 Layer 在 16 个 Sub-Cube 上的精确执行过程。

纵轴：

```text
SC-0
SC-1
...
SC-15
```

横轴：

```text
Cycle 0
Cycle 1
Cycle 2
...
```

---

# 18. Timeline 状态含义

时间线中：

```text
S = Switch
G = gate
U = up
D = down
空白 = Idle
```

颜色与 3D Cube 中的状态保持一致。

当前 Cycle 使用深色边框标出。

---

# 19. Layer Timeline 播放

Timeline 上方提供播放控制：

```text
回到第一个 Cycle
上一个 Cycle
播放 / 暂停
下一个 Cycle
跳到最后一个 Cycle
Speed
```

页面还会显示当前 Layer 的：

```text
Cycles
Tasks
Active SC
```

例如：

```text
Cycles = 6
Tasks = 27
Active SC = 13
```

---

# 20. Current Cycle 摘要

为了减少页面重复信息，当前 Cycle 不再显示 16 个大状态卡片。

只保留一行摘要：

```text
Cycle 0
Switch 13
Gate 0
Up 0
Down 0
Idle 3
```

具体哪个 Sub-Cube 在做什么，则直接从 Timeline 中查看。

---

# 21. 调度规则

当前网页中的单层调度基于项目 Scheduler。

主要规则包括：

```text
不同 Sub-Cube 可以并行执行

同一 Sub-Cube 同一时刻只能执行一个 Weight-Cube

切换 Weight-Cube 会产生 Switch 周期

gate 与 up 可以并行

down 需要等待本 Expert 的 gate 和 up 完成

当前跨 Sub-Cube 通信开销按项目既定设定处理
```

---

# 22. 3D Execution

Timeline 旁边显示唯一一个 3D Execution Cube。

该 Cube 由：

```text
16 个 Sub-Cube
```

组成。

3D Cube 与 Timeline 显示的是同一时刻的硬件状态。

状态颜色表示：

```text
Idle
Switch
gate
up
down
Shared
```

---

# 23. 3D Execution 两种模式

当前页面只保留一个 3D Cube，但它可以接收两种状态源。

## 单 Layer 模式

当用户播放 Layer Timeline 时：

```text
LayerTimeline
    ↓
3D Cube
```

3D Cube 显示当前 Layer 的当前 Cycle。

---

## Full Token 模式

当用户运行完整 Token 时：

```text
Full Token Player
    ↓
3D Cube
```

此时同一个 Cube 会显示完整 Token 当前 Global Cycle 下的状态。

这样可以避免页面同时出现两个重复的 3D Cube。

---

# 24. Full Token Execution

页面下方保留：

```text
FULL TOKEN EXECUTION
58-Layer Token Schedule
```

点击：

```text
Run Full Token
```

后端会对当前真实 Token 的：

```text
58 个 Layer
```

进行完整调度。

---

# 25. Full Token Task 数量

每层：

```text
9 Expert × 3 Matrix = 27 Task
```

因此完整 Token：

```text
58 × 27 = 1566 Task
```

Full Token 调度会计算整个 Token 从：

```text
Layer 0
```

到：

```text
Layer 57
```

的总推理周期。

---

# 26. Full Token 播放

运行完成后，可以播放完整 Token。

页面显示：

```text
Total Cycles
Global Cycle
Current Layer
Local Cycle
Active SC
```

例如：

```text
Total = 522 cycles
Global = 127
Layer = L14
Local = 4
Active SC = 9
```

表示：

```text
当前整个 Token 已运行到第 127 个全局周期，
当前正在执行 Layer 14，
并处于该 Layer 的局部第 4 个周期。
```

---

# 27. Full Token 播放控制

支持：

```text
第一个 Cycle
上一个 Cycle
播放 / 暂停
下一个 Cycle
最后一个 Cycle
Speed
```

播放时：

```text
L0
→ L1
→ L2
→ ...
→ L57
```

3D Cube 会同步显示当前执行状态。

---

# 28. 58-Layer Cycle Overview

Full Token 区域保留 58 层周期图。

每个格子表示：

```text
Layer ID
+
该 Layer 的执行周期数
```

例如：

```text
L0  8
L1  10
L2  8
...
```

当前正在执行的 Layer 会高亮。

点击某个 Layer 格子，也可以快速跳到该层。

---

# 29. 结果分析页面

点击左侧：

```text
03 结果分析
```

进入多 Token Workload 评估页面。

该页面主要用于回答：

```text
当前 Mapping 在真实 Trace 下的总体性能怎么样？
```

而不是展示单个 Token 的详细执行过程。

---

# 30. Workload 参数

可以选择：

```text
Trace Category
Token Count
```

常用 Token 数量：

```text
10
100
1000
```

建议调试时：

```text
先运行 10 Tokens
```

确认结果正常后，再运行更大的样本。

---

# 31. Workload 评估流程

点击：

```text
Run Workload
```

系统会：

```text
读取真实 Trace Token
        ↓
获得每个 Token 的 58 层 Top-8 Route
        ↓
加入 Shared Expert
        ↓
执行完整 Token Scheduler
        ↓
统计多个 Token 的延迟
```

当前多个 Token 主要按独立 Token 方式评估。

---

# 32. Workload 延迟指标

结果页面会显示：

```text
Mean
Min
P50
P95
P99
Max
```

单位为：

```text
cycles / token
```

---

# 33. Latency Distribution

Token Latency Distribution 用于显示不同 Token 延迟的分布。

横轴：

```text
Latency 区间
```

纵向柱高：

```text
处于该区间的 Token 数量
```

---

# 34. 58-Layer Mean Latency

页面会统计：

```text
L0 ~ L57
```

在多个真实 Token 上的平均执行周期。

可用于观察：

```text
哪些 Layer 长期更慢
```

---

# 35. Critical Sub-Cube

Critical Sub-Cube 表示：

```text
某个 Layer 中最晚完成、决定该 Layer 最终结束时间的 Sub-Cube
```

结果页面会统计：

```text
SC-0 ~ SC-15
```

成为 Critical Sub-Cube 的频率。

如果某个 Sub-Cube 明显偏高，通常说明：

```text
当前 Mapping 在该 Sub-Cube 上存在更严重的任务集中或冲突
```

---

# 36. Slowest Tokens

结果页面还会保留延迟最高的一些真实 Token。

通常包含：

```text
Category
File
Segment
Token Index
Latency
```

便于进一步回到 Trace 中检查该 Token 的路由模式。

---

# 37. 当前结果的正确理解

当前网页使用的是：

```text
已经生成好的当前 Mapping
```

因此当前 Workload 结果回答的是：

```text
当前 Mapping 在真实 Router Trace 下需要多少推理周期？
```

它暂时不能直接证明：

```text
当前 Mapping 是最优 Mapping
```

如果后续加入新的 Mapping 优化算法，需要：

```text
不同 Mapping 策略
        ↓
分别生成 Mapping Result
        ↓
使用相同 Trace
        ↓
使用相同 Scheduler
        ↓
比较 Mean / P95 / P99 / Max
```

这样才能进行严格的 Mapping 策略对比。

---

# 38. FastAPI 常用检查地址

后端启动后，可以打开：

```text
http://127.0.0.1:8000/docs
```

检查 API。

常见接口包括：

```text
System Summary
Sub-Cube
Layer Mapping
Trace
Layer Schedule
Full Token Schedule
Workload
```

---

# 39. 常见问题：前端无法启动

确认当前终端位于：

```text
webui/frontend
```

然后运行：

```bash
npm run dev
```

不要在项目根目录直接运行：

```bash
npm run dev
```

否则可能出现：

```text
Missing script: "dev"
```

---

# 40. 常见问题：第一次运行缺少依赖

在：

```text
webui/frontend
```

运行：

```bash
npm install
```

之后再执行：

```bash
npm run dev
```

---

# 41. 常见问题：后端连接失败

确认后端仍然运行：

```bash
python -m uvicorn webui.backend.main:app --reload --port 8000
```

然后打开：

```text
http://127.0.0.1:8000/docs
```

检查是否能够正常访问。

---

# 42. 常见问题：Physical Mapping 出现 Not Found

首先检查：

```text
/api/layers/{layer_id}
```

是否能够返回当前 Layer 的 Weight-Cube Mapping。

正常情况下，当前 Token Layer 应该能够找到：

```text
9 Expert × 3 Matrix = 27 Matrix
```

如果缺失，需要检查：

```text
API 返回字段
expert_id
matrix_name
subcube_id
z
physical_plane_id
slot_id
```

---

# 43. 常见问题：浏览器出现翻译插件报错

例如：

```text
Immersive Translate ERROR
dynamic-i18n version mismatch
```

这通常来自浏览器翻译插件，不属于本项目代码错误。

调试网页时建议关闭自动翻译功能。

---

# 44. 常见问题：THREE.Clock deprecated

浏览器控制台可能出现：

```text
THREE.Clock: This module has been deprecated
```

这是 Three.js / React Three Fiber 相关依赖的兼容警告。

如果 3D 页面仍然可以正常显示和操作，可以暂时忽略。

---

# 45. 常见问题：WebGL Context Lost

如果偶尔出现：

```text
THREE.WebGLRenderer: Context Lost
```

但页面切换后 3D Cube 可以恢复，一般不影响使用。

如果持续黑屏，则需要检查：

```text
是否重复创建多个 Canvas
3D 组件是否被频繁卸载
浏览器 WebGL / GPU 状态
```

当前 Token 页面应该只保留一个主要 Execution 3D Cube。

---

# 46. 停止网页

停止前端：

```text
在前端终端按 Ctrl + C
```

停止后端：

```text
在后端终端按 Ctrl + C
```

---

# 47. 推荐演示流程

如果需要快速完整展示系统，可以按下面顺序操作：

```text
1. 启动后端
2. 启动前端
3. 打开 Cube 总览
4. 查看 16 个 Sub-Cube
5. 点击一个 Sub-Cube 查看某个 Plane
6. 进入 Token 模拟
7. 随机选择一个真实 Token
8. 查看当前 Layer 的 Top-8 + Shared Expert
9. 展开 Physical Mapping
10. 查看 gate / up / down 的真实位置
11. 播放 Layer Timeline
12. 观察 3D Cube 同步变化
13. 点击 Run Full Token
14. 播放完整 58 层 Token
15. 观察 Global Cycle、Current Layer 和 3D Cube
16. 进入结果分析
17. 先运行 10 个 Token
18. 查看 Mean / P50 / P95 / P99 / Max
```

---

# 48. 各页面与项目步骤的对应关系

```text
前三步：空间规划
    ↓
Cube 总览
Sub-Cube / Plane 查看


第四步：静态 Mapping
    ↓
Physical Mapping


第五步：精确调度
    ↓
Layer Timeline
3D Execution
Full Token Execution


多 Token Trace 评估
    ↓
结果分析


后续 Mapping 优化
    ↓
不同 Mapping 策略对比
消融实验
最终性能评估
```

---

# 49. 当前 Web 页面的主要作用

当前网页主要承担两个任务：

## 1. 验证 Mapping 是否正确落到硬件空间

通过：

```text
Cube
Sub-Cube
Plane
Physical Mapping
```

检查权重的真实位置。

## 2. 展示 Mapping 如何影响推理过程

通过：

```text
Layer Timeline
3D Execution
Full Token Execution
Workload Analysis
```

观察：

```text
Sub-Cube 并行
Weight-Cube Switch
任务冲突
Critical Sub-Cube
Layer Latency
Token Latency
```

为后续 Mapping 优化和消融实验提供直观依据。

---

# 50. 隐私说明

本文档没有包含：

```text
个人姓名
电脑用户名
个人绝对路径
本地硬盘目录
Conda 环境名称
设备编号
私人账户
学校或个人身份信息
```

实际部署或分享项目时，建议继续使用：

```text
相对路径
通用端口
通用项目目录
```

避免将本地开发机信息写入公开文档。
