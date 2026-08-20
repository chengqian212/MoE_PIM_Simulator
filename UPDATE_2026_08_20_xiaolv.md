# UPDATE_2026_08_20 — Prefill / Decode 模拟器运行效率优化

## 1. 本次修改目标

本次更新只优化 **MoE Prefill / Decode 推理周期评估程序本身的运行效率**。

核心原则：

> **同一份 Mapping、同一份 Trace、同一套调度规则，优化前后得到的推理周期必须一致。**

本次没有修改：

- 58 个 MoE Layer 的执行顺序；
- Routed Top-8 + Shared Expert 的激活规则；
- `gate || up -> down` 的依赖关系；
- 不同 Sub-Cube 并行、同一 Sub-Cube 串行的规则；
- `compute = 1 cycle`；
- `switch = 1 cycle`；
- 跨 Sub-Cube 开销为 0；
- Prefill 中同一 Weight-Cube 连续处理多个 Token 时只切换一次、但每个 Token 仍独立计算的规则；
- Mapping、Plane Pairing、Sub-Cube Placement 等静态映射结果。

因此，本次更新属于：

```text
Simulator Runtime Optimization
```

不是新的 Mapping 算法，也不是新的 Scheduling 策略。

---

# 2. 原代码中影响运行速度的主要因素

## 2.1 Prefill 会创建大量完整 Task 对象

原来的 `prefill_layer_scheduler.py` 是完整事件调度器。

对于一个 Batch 中的每个：

```text
Token × Expert × gate/up/down
```

都会构造对应 Task，并记录 ready、dispatch、finish、wait、switch 等完整信息。

当前每个 Token 每层实际有：

```text
8 Routed Expert + 1 Shared Expert = 9 Expert
9 × 3 Matrix = 27 Tasks
```

因此一个 Batch 大小为 `B` 时，58 层一共需要处理：

```text
B × 58 × 27
```

个任务。

完整 Task 对象适合调试和可视化，但全量 Prefill 评估时会产生大量：

- Python 对象创建；
- dataclass 属性访问；
- List / Tuple 中间对象；
- 垃圾回收开销。

---

## 2.2 Prefill Ready Queue 会反复扫描

原调度器在同一个 Sub-Cube 中选择下一个任务时，需要反复执行：

```text
扫描 Queue
↓
找 ready_time <= current_time 的任务
↓
创建 candidates
↓
min(candidates)
↓
list.remove()
```

Prefill Batch 中任务数量很多，这种重复 List 扫描会被放大。

---

## 2.3 Prefill 使用较重的 Event Loop

原 Prefill Scheduler 会维护：

```text
running heap
ready queue
current time
running task
gate finish
up finish
down ready
```

任务完成后还需要持续：

```text
弹出事件
→ 更新状态
→ 检查依赖
→ 生成 down
→ 再寻找下一任务
```

该实现适合作为 EXACT Scheduler，但大规模评估时 Python 控制流开销较高。

---

## 2.4 Prefill 原先会解析大量不需要的 Decode Segment

Chinese-SimpleQA 当前已经确认：

```text
segment0  = Prefill
segment1+ = Decode
```

但是原 Prefill workload 使用通用 Segment Reader，会遍历一个 JSON 中的后续 Decode Segment。

全量数据只有：

```text
2020 个 Prefill Batch
```

但总 Segment 超过 25 万。

因此 Prefill 路径原先会花时间解析大量最终不会参与 Prefill 计算的 Decode Segment。

---

## 2.5 Decode 存在重复 Segment 解析

原 `decode_workload.py` 中，一个有效 Decode Segment 会先调用：

```text
collect_segment_routes()
```

检查合法性。

之后进入 Token 构造阶段时，又可能再次收集同一个 Segment 的 Layer Route。

全量 Decode Token 数量很大，这种重复扫描累计后会产生明显开销。

---

## 2.6 Decode Route 存在重复合法性检查

Workload Reader 已经检查过：

- 58 层完整；
- 每层恰好 Top-8；
- Expert ID 合法；
- Top-8 无重复。

进入 Fast Scheduler 后又逐层执行一次 Router Route 验证。

对于已经经过 Workload Reader 的正式评估路径，这属于重复工作。

---

## 2.7 原 Decode Fast 在 Cache Miss 时仍使用轻量 Event Loop

原 `decode_fast_evaluator.py` 已经比完整 EXACT Scheduler 快，但 Cache Miss 时仍维护：

```text
queue
running task
finish time
gate/up finish
down created
```

仍然需要进行较多 Python List / Tuple / 分支操作。

---

## 2.8 全量评估主要还是单进程

即使单个 Token 或 Batch 已经进行 Fast 化，全量：

```text
2020 Prefill Batch
+
约 25 万 Decode Token
```

如果仍然由一个 Python 进程顺序执行，就无法充分利用多核 CPU。

---

# 3. 第一次修改：Fast 内核优化

第一次修改主要解决：

> **单个 Token / Batch 本身算得太慢。**

---

## 3.1 新增 Fast Prefill Evaluator

新增：

```text
scheduling/prefill_fast_evaluator.py
```

原来的：

```text
prefill_layer_scheduler.py
prefill_scheduler.py
```

继续保留，作为 EXACT 参考实现。

新的 Fast Prefill 用于正式大规模评估。

---

## 3.2 Prefill 减少完整 Task 对象创建

Fast Prefill 不再为每个：

```text
Token × Expert × Matrix
```

都创建完整 `ScheduledPrefillTask`。

改为主要维护紧凑运行状态：

```text
SC 当前时间
SC 当前 active Weight-Cube
gate finish
up finish
down ready
Switch
Wait
Busy Cycles
```

减少 Python 对象分配和属性访问。

---

## 3.3 Prefill 针对固定调度规则进行直接状态计算

当前执行规则已经固定：

```text
gate / up 初始 ready

gate(token,e) ─┐
                ├→ down(token,e)
up(token,e) ───┘

不同 SC 并行
同一 SC 串行

compute = 1
switch = 1
```

因此 Fast 版本可以直接维护对应状态，而不必为所有步骤建立通用事件对象。

这只是更换程序实现方式，不改变任务依赖和执行顺序。

---

## 3.4 Prefill Workload 只读取 segment0

修改：

```text
scheduling/prefill_workload.py
```

正式 Prefill 评估现在对每个 JSON：

```text
读取 JSON
↓
只取 segment0
↓
验证 Prefill Route
↓
运行 Prefill Fast Scheduler
```

不再继续解析该请求的 `segment1+`。

---

## 3.5 Decode Fast 内核进一步简化

修改：

```text
scheduling/decode_fast_evaluator.py
```

Cache Miss 时不再使用轻量 Event Loop。

当前 Mapping 和调度规则固定后，可以直接根据：

```text
gate 所在 SC
up 所在 SC
↓
gate/up finish
↓
down ready
↓
down 串行时间
↓
Layer finish
```

计算 Layer 周期。

---

## 3.6 Decode Workload 去掉重复 Segment 收集

修改：

```text
scheduling/decode_workload.py
```

已经完成一次 `collect_segment_routes()` 后，后续 Token 构造直接复用结果，不再对同一个 Segment 再扫描一次。

---

## 3.7 Fast 正式热路径减少重复 Route 校验

已经由 Trace / Workload Reader 验证过的 Route，在正式 Fast Evaluator 热路径中不再每个 Token、每个 Layer 重复验证。

独立调用 Scheduler 时仍保留必要的合法性检查能力。

---

## 3.8 增加 FAST == EXACT 回归

新增：

```text
tests/test_runtime_fast_path.py
```

Fast Scheduler 不是近似算法。

运行正式评估时，可以先用少量样本同时计算：

```text
FAST
vs
EXACT
```

要求：

```text
Total Cycles 完全一致
58 Layer Cycles 完全一致
```

Prefill 还会对 Switch、Wait、最终 SC active state 等核心结果进行一致性验证。

只要不一致就立即报错。

---

# 4. 第二次修改：Fast 内核上叠加多进程

第二次修改主要解决：

> **单个 Token / Batch 已经快了，但全量请求仍然一个个算。**

因此在 Fast 内核上增加：

```text
ProcessPoolExecutor
```

进行多进程并行。

---

## 4.1 使用多进程而不是多线程

该评估主要是 Python CPU 计算。

为了避免 Python GIL 对线程并行的限制，使用：

```text
ProcessPoolExecutor
```

让多个 Python 进程真正利用多个 CPU 核心。

---

## 4.2 使用 JSON 文件级粗粒度并行

没有采用：

```text
主进程读取每个 Token
→ 一个 Token 一个 Token 发给 Worker
```

因为 Windows 多进程需要 pickle 数据，大量 Token Route 跨进程传输会产生很高 IPC 开销。

第二次修改采用：

> **按 JSON 文件分片。**

---

## 4.3 Prefill 并行方式

多个 Worker 分别处理不同 JSON：

```text
Worker
↓
读取自己的 JSON 文件
↓
取 segment0
↓
解析 Prefill Route
↓
运行 Fast Prefill
↓
返回局部结果
```

不同请求之间本来就互相独立，因此可以安全并行。

---

## 4.4 Decode 并行方式

Decode 同样按 JSON 文件分片。

每个 Worker：

```text
读取自己的 JSON
↓
处理 segment1+
↓
运行 Fast Decode
↓
本地聚合统计
↓
返回紧凑结果
```

避免主进程向 Worker 传输大量 `58 × 8` Token Route。

---

## 4.5 Decode Cache 按 Worker 分摊

多进程后，每个 Worker 都有自己的进程内 LRU Cache。

如果每个 Worker 都使用原来的完整 Cache Size，会导致内存占用成倍增加。

因此第二次修改中：

```text
总 Cache Budget
↓
按照 Worker 数进行近似分摊
```

该改动只影响：

- Simulator 内存；
- Cache Hit / Miss；
- Simulator 实际运行时间。

不会影响推理周期。

Cache Miss 时仍然重新执行同一个 Fast Scheduler。

---

## 4.6 `run_phase_evaluation.py` 增加 `--workers`

修改：

```text
scheduling/run_phase_evaluation.py
```

现在支持：

```bash
--workers 1
```

表示单进程。

```bash
--workers 4
```

表示 4 个进程。

```bash
--workers 8
```

表示 8 个进程。

```bash
--workers 0
```

表示自动选择 Worker 数量。

`workers=0` 不是单核。

---

## 4.7 小规模 Smoke Test 不强制多进程

多进程也存在固定启动成本：

```text
创建子进程
加载 Python 模块
加载 Mapping
建立 RuntimeIndex
建立 Fast Table / Cache
```

因此：

```bash
python -m scheduling.run_phase_evaluation --smoke
```

这类小规模测试优先单进程 Fast，避免进程启动成本反而超过实际计算成本。

正式全量评估再使用：

```bash
python -m scheduling.run_phase_evaluation --workers 0
```

---

## 4.8 增加单进程 / 多进程一致性测试

新增：

```text
tests/test_parallel_fast_path.py
```

检查：

```text
Single Process
vs
Multi Process
```

在同一批数据上得到一致的：

- Prefill / Decode Cycles；
- Layer 结果；
- 最终 Summary。

允许不同的只有：

- Simulator 实际墙钟运行时间；
- Worker 执行顺序；
- Cache Hit / Miss 数量。

---

# 5. 本次涉及的主要文件

## 新增

```text
scheduling/prefill_fast_evaluator.py

tests/test_runtime_fast_path.py
tests/test_parallel_fast_path.py
```

## 修改

```text
scheduling/prefill_workload.py
scheduling/decode_workload.py
scheduling/decode_fast_evaluator.py
scheduling/run_phase_evaluation.py
```

## 保留不删除的 EXACT 实现

```text
scheduling/prefill_layer_scheduler.py
scheduling/prefill_scheduler.py
scheduling/layer_scheduler.py
scheduling/token_scheduler.py
```

这些文件继续承担：

```text
正确性基准
调试
详细事件时间线
FAST / EXACT 回归
```

---

# 6. 为什么这次修改不会改变推理周期

需要区分两种并行。

## PIM 模拟中的 Sub-Cube 并行

这是被模拟硬件的一部分：

```text
不同 Sub-Cube 可以同时工作
```

它会影响最终推理周期。

## 电脑 CPU 上的多进程

这是 Simulator 自己的程序执行方式：

```text
CPU Process-0 → 计算 Request-A
CPU Process-1 → 计算 Request-B
CPU Process-2 → 计算 Request-C
```

它只让电脑更快地求出多个 Request 的周期。

例如单进程得到：

```text
A = 3308 cycles
B = 2914 cycles
C = 4012 cycles
```

改成多进程后仍然必须得到：

```text
A = 3308 cycles
B = 2914 cycles
C = 4012 cycles
```

改变的是程序跑完这三个请求需要的真实时间，而不是模拟出来的 PIM cycles。

---

# 7. 推荐测试命令

先运行 Fast 内核测试：

```bash
python -m pytest tests/test_runtime_fast_path.py -v
```

再运行多进程一致性测试：

```bash
python -m pytest tests/test_parallel_fast_path.py -v
```

再运行原 Prefill / Decode 回归：

```bash
python -m pytest tests/test_phase_evaluation.py -v
```

---

# 8. 推荐运行命令

## Smoke Test

```bash
python -m scheduling.run_phase_evaluation --smoke
```

## 正式全量，自动选择 CPU Worker

```bash
python -m scheduling.run_phase_evaluation --workers 0
```

## 手动 4 Worker

```bash
python -m scheduling.run_phase_evaluation --workers 4
```

## 单进程对照

```bash
python -m scheduling.run_phase_evaluation --workers 1
```

## 已确认 FAST / EXACT 一致后，后续批量实验关闭重复 Exact Check

```bash
python -m scheduling.run_phase_evaluation --workers 0 --prefill-exact-check 0 --exact-check 0
```

---

# 9. 本次更新总结

本次运行效率优化分两轮完成。

第一次主要优化：

```text
单个 Prefill Batch / Decode Token 怎么算得更快
```

包括：

- Fast Prefill；
- Decode Fast 进一步简化；
- 减少 Task 对象；
- 减少 Event Loop；
- Prefill 只读取 segment0；
- Decode 去掉重复 Segment 解析；
- 减少重复 Route 校验；
- 保留 FAST == EXACT 回归。

第二次主要优化：

```text
大量请求怎么利用 CPU 多核同时算
```

包括：

- `ProcessPoolExecutor`；
- JSON 文件级粗粒度并行；
- Windows `spawn` 兼容；
- Worker 本地读取 Trace；
- Worker 本地执行 Fast Scheduler；
- Cache 按 Worker 分摊；
- 新增 `--workers`；
- 新增单进程 / 多进程一致性回归。

最终形成：

```text
Fast Scheduler
+
减少重复数据处理
+
文件级多进程并行
```

整个优化过程始终保持：

> **不修改推理周期的计算语义，只缩短 Simulator 自己求解推理周期所需要的运行时间。**
