# MoE-PIM Simulator Web 使用说明

> 面向已经了解 MoE / PIM 映射、调度和实验评估流程的用户。  
> 本文只说明网页怎么使用、每一栏展示什么、结果看哪里。

---

# 1. 启动网页

## 后端

在项目根目录运行：

```bash
python -m uvicorn webui.backend.main:app --reload --port 8000
```

后端接口：

```text
http://127.0.0.1:8000/docs
```

## 前端

新开一个终端：

```bash
cd webui/frontend
npm install
npm run dev
```

浏览器打开：

```text
http://localhost:5173/
```

---

# 2. 页面结构

当前页面共四栏：

```text
01 总览
02 映射空间
03 策略对比
04 实验结果
```

推荐使用顺序：

```text
01 看最终方案和核心性能
→
02 看权重实际放置和 UP-UP Pairing
→
03 实时比较 Mapping / Prefill / Decode 策略
→
04 看正式 Held-out 实验结论
```

---

# 3. 01 总览

点击：

```text
01 总览
```

该页只看最终方案和核心性能。

主要内容：

```text
最终静态 Mapping
Prefill 调度策略
Decode 调度策略
Prefill 核心指标
Decode 核心指标
数据划分
```

## Prefill

重点看：

```text
Mean Latency
Cycles / Input Token
P95
评估 Batch / Input Token 数
```

当前运行策略：

```text
Aggressive-Reuse
```

## Decode

重点看：

```text
Mean Cycles / Token
P95
Decode Token 数
```

当前实际调度：

```text
Greedy
```

CP-SAT 只作为最优性验证结果查看。

## 数据划分

页面中的：

```text
Profile 1,616 文件
Evaluation 404 文件
seed=42
```

表示文件级数据划分。

Prefill 和 Decode 都使用 Evaluation 集，统计单位分别为：

```text
Prefill → Batch / Input Token
Decode  → Decode Token
```

---

# 4. 02 映射空间

点击：

```text
02 映射空间
```

该页分为：

```text
当前映射
UP 配对策略
```

---

# 5. 02 - 当前映射

点击：

```text
当前映射
```

用于查看最终静态 Mapping 的物理位置。

## 3D Cube

页面显示：

```text
SC-0 ~ SC-15
```

操作：

```text
左键拖动：旋转
滚轮：缩放
点击 Sub-Cube：进入内部
```

## Sub-Cube

进入某个 Sub-Cube 后查看：

```text
Plane / z
Weight-Cube
Layer
Expert
Matrix Type
Physical Slot
```

矩阵类型：

```text
gate
up
down
```

## Plane

选择具体 `z` 后查看：

```text
该 Plane 上有哪些 Weight-Cube
每个 WC 的 Layer / Expert
Matrix Type
Slot
物理位置
```

## Mapping Locator

使用 Layer / Expert / Matrix 定位后，直接查看：

```text
Sub-Cube
z
Plane
Slot
```

---

# 6. 02 - UP 配对策略

点击：

```text
UP 配对策略
```

用于实时比较两个 Expert 的 up 如何组成同一个 Plane。

可选策略：

```text
Sequential
Random
Frequency-aware
Coactivation Greedy
Greedy + Local Search
Optimal Matching
```

## 使用方法

选择：

```text
Phase：Decode / Prefill
方案 A
方案 B
```

点击：

```text
随机真实请求
```

A/B 使用同一个真实请求，只改变 UP-UP Pairing。

## 页面结果

先看当前请求：

```text
Decode Token
或
Prefill Batch
```

再看：

```text
当前 Layer
当前激活 Expert
方案 A 的 UP Pair
方案 B 的 UP Pair
```

Pair 图直接显示：

```text
E12 ─ E13
E27 ─ E84
...
```

当前请求中同时激活、又被配到同一 Plane 的 Pair 会突出显示。

## 58 Layer Pair 冲突

页面显示：

```text
L0 ~ L57
```

每层 A/B 的 Pair 冲突情况。

点击任意 Layer 查看该层具体 Pair。

主要看：

```text
A/B Pair 具体怎么组成
当前请求命中了多少冲突 Pair
同一请求下两种 Pairing 的 Layer Cycles
```

---

# 7. 03 策略对比

点击：

```text
03 策略对比
```

页面分为三个 Tab：

```text
Mapping
Prefill 调度
Decode 最优性
```

所有实时 A/B 都使用同一个请求。

---

# 8. 03 - Mapping

点击：

```text
Mapping
```

用于比较 Plane → Sub-Cube Mapping。

可选：

```text
Round-Robin
Least-Loaded
Frequency-aware
Trace-aware
```

## 使用方法

选择：

```text
Phase：Decode / Prefill
方案 A
方案 B
```

运行当前随机实例。

## 结果区

先看：

```text
方案 A Total Cycles
方案 B Total Cycles
绝对差值
百分比差值
```

再看：

```text
58 Layer Cycles
16 个 Sub-Cube 负载
当前选中 Layer
```

点击某个 Layer 后，下方 Timeline 和调度动画切换到该层。

---

# 9. 03 - Prefill 调度

点击：

```text
Prefill 调度
```

固定 Mapping，只比较 Prefill Scheduler。

可选：

```text
No-Reuse
Switch-Aware
Aggressive-Reuse
Largest-Batch-Reuse
```

## 使用方法

选择同一个真实 Prefill Batch：

```text
方案 A
方案 B
```

运行后查看：

```text
Total Cycles
Cycles / Input Token
Switch Count
58 Layer Cycles
16-SC 负载
```

重点看：

```text
A/B 总周期
WC Switch 数
不同 Layer 的周期差
```

---

# 10. 03 - Decode 最优性

点击：

```text
Decode 最优性
```

比较：

```text
Greedy
vs
CP-SAT Optimal
```

## 使用方法

选择：

```text
真实 Decode Token
Layer L0 ~ L57
```

运行后查看：

```text
Greedy Layer Cycles
CP-SAT Optimal Cycles
Gap
任务时间线
```

该页只比较当前 `Token × Layer`。

---

# 11. 03 - 58 Layer 对比

Mapping 和 Prefill 对比运行后，会显示：

```text
L0 ~ L57
```

每层 A/B 周期。

操作：

```text
点击某一 Layer
→
下方 Timeline 和调度动画切换到该 Layer
```

主要看：

```text
A/B 差距大的 Layer
慢 Layer
当前选中 Layer
```

---

# 12. 03 - Lxx A/B Timeline

展开：

```text
Lxx A/B Timeline
```

查看当前 Layer 的：

```text
SC-0 ~ SC-15
```

精确任务时间线。

横轴为 Cycle。

A/B 使用同一个绝对时间尺度。

Timeline 中查看：

```text
Switch
Gate
Up
Down
Idle
```

该模块可以折叠。

---

# 13. 03 - 调度动画

展开：

```text
调度动画
```

查看 A/B 两个 3D Cube。

控制：

```text
播放
暂停
单步
重置
速度
Cycle Slider
```

播放时同步：

```text
A 3D Cube
B 3D Cube
Timeline 游标
16-SC 当前状态
Current Cycle
```

3D Cube 中查看：

```text
Idle
Switch
Gate
Up
Down
```

A/B 使用同一个 Cycle。

该模块可以折叠。

---

# 14. 03 - 16-SC 当前状态

调度动画下方查看：

```text
SC-0 ~ SC-15
```

当前 Cycle 的状态。

每个 Sub-Cube 显示：

```text
Running
Waiting
Idle
```

运行任务时还能看到：

```text
Expert
Matrix Type
Switch / Compute
```

拖动 Cycle Slider 可以检查任意时刻。

---

# 15. 04 实验结果

点击：

```text
04 实验结果
```

该页只看正式 Held-out 实验，不进行实时随机测试。

主要分为：

```text
Mapping Baseline
Pairing × Mapping 2×2 消融
Prefill Scheduler
Decode 最优性
Expert Replication
```

---

# 16. 04 - Mapping Baseline

查看：

```text
Random
Round-Robin
Least-Loaded
Frequency-aware
Trace-aware
```

主要指标：

```text
Mapping Conflict
Prefill Mean
Decode Mean
Decode P95
```

---

# 17. 04 - Pairing × Mapping 2×2

查看：

```text
Naive
Pairing Only
Mapping Only
Full
```

主要看：

```text
Prefill
Decode
P95
```

---

# 18. 04 - Prefill Scheduler

查看：

```text
No-Reuse
Switch-Aware
Aggressive-Reuse
Largest-Batch-Reuse
```

主要指标：

```text
Mean Prefill
Cycles / Input Token
P95
WC Switch
```

---

# 19. 04 - Decode 最优性

查看 Greedy 与 CP-SAT 的正式验证结果。

主要看：

```text
CP-SAT Proven OPTIMAL 数量
Greedy Already Optimal 比例
Mean Gap
Max Improvement
```

---

# 20. 04 - Expert Replication

查看：

```text
Baseline
Balanced Replication
Oracle
```

主要看：

```text
Mean Prefill
P95
Improved / Equal Batch
Oracle 上限
```

---

# 21. 推荐使用流程

```text
1. 进入 01 总览
   看最终 Prefill / Decode 核心指标

2. 进入 02 当前映射
   查看 16 个 Sub-Cube
   点开一个 Sub-Cube / Plane
   检查 Weight-Cube 物理位置

3. 进入 02 UP 配对策略
   选择 Sequential vs Greedy + Local Search
   随机真实请求
   查看当前 Layer Pair 差异

4. 进入 03 Mapping
   选择 Round-Robin vs Trace-aware
   运行同一真实请求
   查看 Total Cycles 和 58 Layer 差异

5. 点击差距明显的 Layer
   展开 Lxx A/B Timeline

6. 展开调度动画
   播放 A/B 3D Cube
   查看同 Cycle 下的 16-SC 状态

7. 进入 03 Prefill 调度
   选择 No-Reuse vs Aggressive-Reuse
   查看总周期和 Switch 数

8. 进入 03 Decode 最优性
   选择 Token + Layer
   查看 Greedy vs CP-SAT

9. 进入 04 实验结果
   查看正式 Mapping / Prefill / Decode / Replication 结果
```

---

# 22. 实时结果和正式结果

## 实时随机实例

位于：

```text
02 UP 配对策略
03 策略对比
```

表示当前随机：

```text
Request
Batch
Token
Layer
```

## 正式实验结果

位于：

```text
01 总览
04 实验结果
```

表示 Evaluation 集上的汇总结果。

---

# 23. 常用排查

## 后端接口

```text
http://127.0.0.1:8000/docs
```

## 前端无法启动

确认目录：

```text
webui/frontend
```

运行：

```bash
npm run dev
```

## 第一次启动

```bash
npm install
```

## 页面没有数据

检查：

```text
后端是否运行
Mapping JSON 是否存在
Trace 是否可读取
results/experiments 下正式实验结果是否存在
```

## 3D 页面异常

刷新页面后重新进入对应模块。

---

# 24. 停止网页

前端终端：

```text
Ctrl + C
```

后端终端：

```text
Ctrl + C
```
