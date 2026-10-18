# 天空掩码学生模型 · 执行方案 (v4)

> 本文件是 v4 路线的**执行手册**：需求规范 → 分阶段步骤 → 验证 → 风险。
> 每执行一步对照本文，判定门槛不达标就走失败回路，不要加容量硬扛。
> 研究背景与设计论证见 `SKY_TRAINING.md`；与 v3（`README.md`）冲突时以本文件为准。

---

## 0. 适用范围与角色

- 范围：单通道 Y 平面蒸馏学生模型 → INT8 RKNN → 板端检测级验收。
- 不在范围：教师模型自身的再训练、人工真值标注（除非 §7 风险触发）。
- 角色：本文件不依赖 `ultralytics` 内部 API；新建 `skytrain/` 子包独立实现训练循环。

---

## 1. 需求规范

### 1.1 接口契约（输入 / 输出，长期稳定，不可违反）

| 项 | 规定 |
|---|---|
| 输入张量 | `float32`，`1×1×288×384`（H=288 × W=384） |
| 输入内容 | NV12 的 **Y 平面**（Rec.601 亮度），由 RGA 从 640×480 缩到 384×288 |
| 输入归一化 | `(Y/255 − 0.5) / 0.5 → [-1, 1]`；板端等价 `mean_values=[[127.5]]`、`std_values=[[127.5]]`，uint8 直喂 NPU |
| 通道顺序 | 单通道，不存在通道顺序问题 |
| 输出张量 | `float32`，`1×1×72×96`（C1）或 `1×1×144×192`（C2/C3），**Sigmoid 后的软 alpha ∈ [0, 1]** |
| INT8 输出 | RKNN 用 uint8 0–255（scale=1/255），板端不还原浮点 |
| 语义 | `1` = 天空（保留），`0` = 地面（抑制） |
| 板端后处理 | RGA 双线性放大回 640×480，不额外羽化；随后由调用方自行叠加到 RGB 或 Y |
| 静态形状 | opset 11，`dynamic_axes={}`，batch=1；改尺寸必须重导出 |

**为什么是 384×288 而不是 640×480 / 320×240**：
- 640×480 多花 2.8 倍算力只为省一次 RGA 缩放（< 1 ms），不值。
- 320×240 实测非法：YOLOv8 系最大 stride 32，240/32 = 7.5，前向直接崩。
- 4:3 零补边的合法档位只有 `k·128 × k·96`：128×96 / 256×192 / 384×288 / 640×480。

### 1.2 性能门槛

| 指标 | 门槛 | 性质 |
|---|---|---|
| `leakage`（地面残留在天空里） | **≤ 1%** | 硬约束 |
| `erosion`（天空被啃） | 在硬约束下最小化 | 软目标，初始门槛 ≤ 5% |
| INT8 vs fp32 `leakage` | 不劣化（仍 ≤ 1%） | 量化门槛 |
| INT8 `erosion` 增幅 | ≤ 0.5 个百分点 | 量化门槛 |
| 单核延迟（与检测并发） | ≥ 30 fps | 部署门槛 |
| INT8 静态形状 | 必须是常量列表 | 导出门槛 |
| 时序抖动 `jitter.p95` | 直观可接受（具体上限由检测级验收决定） | 失效模式门槛 |

**`leakage` 和 `erosion` 都必须在双线性放大回 640×480 之后计算**，不在网格上算——分辨率损失必须计入。对称 IoU 不作为主指标。

### 1.3 数据规模与排除规则

- 全库 2630 帧 640×480 RGB，手持扫拍，3 个域（`1/`、`2/`、`3/`）。
- 排除 3 帧：`3/3_000001.png`、`3/3_000002.png`、`3/3_000003.png`（教师在此处对 RGB 自身不稳定，亮度反而更接近正确）。**训练库 2627 帧**。
- 划分：每域内按帧号排序，40 帧一块，块序号 `i % 10 == 9` 进验证集，三域独立分块。
- INT8 校准集：64 张必须**从验证块取**，取训练集会系统性低估量化漂移。

### 1.4 不在范围

- ❌ 重训教师模型（44M 参数，已稳定）
- ❌ 人工标注真值（除非 §7 触发，预算 100 帧 / 2 小时）
- ❌ 修改 ultralytics 内部 trainer（v4 完全脱离 ultralytics trainer）
- ❌ 多尺度监督、CRF、边界 loss（先验证简单组合的上限）
- ❌ 把 `max_det`、`conf` 等部署侧优化塞进训练侧

---

## 2. 资产现状与可复用项

实测清点（截至 2026-10-18）：

| 资产 | 状态 | v4 用途 |
|---|---|---|
| `sky_workflow/dataset/images/{train,val}/*.png` | 2455 + 172 = 2627 | 输入图源（直接复用） |
| `sky_workflow/work/prob/{1,2,3}/*.npy` | 2630 个 float32，1.2 MB/帧 | **软标签种子**（免跑教师推理） |
| `sky_workflow/work/mask_clean/{1,2,3}/*.png` | 11 M | 不用于 v4（二值硬标签会废掉软 alpha） |
| `sky_workflow/work/excluded.txt` | 3 帧 | 排除规则一致 |
| `sky_workflow/work/qc_frames.csv` | 邻帧跳变统计 | 时序抖动基线 |
| `sky_workflow/models/skyseg_teacher.onnx` | 168 M，单输出 | 不再使用（v3 用过的教师，v4 用 .npy 软标签即可） |
| `sky_workflow/models/yolov8n-sky1-seg.yaml` | v3 学生 yaml | v4 不使用（v4 自定义 SkyHead） |
| `sky_workflow/dataset/data.yaml` | v3 数据配置 | v4 不使用（自写 Dataset） |

**简化点 1**：`SKY_TRAINING.md` §4.3 原本要跑 OpenVINO Arc GPU 60 s 重新产 16-bit PNG；现有 `work/prob/*.npy` 就是教师原概率图，**直接 `.npy → uint16 PNG` 即可**，跳过教师推理这一步。

**简化点 2**：v3 已跑了 03 时序质检、08 教师往返校验（min=0.9566 是细长锯齿带帧，非标签错误），结果可直接采信。

---

## 3. 技术方案概述

### 3.1 总体路线

```
[已完成 v3]                                [v4 待执行]
input PNG ──┐                         ┌── SkyHead(C1) / TinyNet(C2/C3)
            │                         │
            ├─ 教师推理 (v3 已跑)      │
            v                         v
   work/prob/*.npy (3.1 GB软标签) ──> Phase 0 转换 ──> 16-bit PNG
            │                                              │
            │                                              v
        mask_clean                                 YSkyDataset (单通道 + 范围抖动 + 同步几何增广)
                         │                                │
                         v                                v
                  v3 二值多边形标签              Phase 1 gamma 标定(C2)
                  (本方案不再用)                          │
                                                          v
                                                  Phase 2 三候选横向训练
                                                          │
                                                          v
                                                  Phase 3 输出分辨率消融
                                                          │
                                                          v
                                                  Phase 4 RKNN INT8 + 上板 bench
                                                          │
                                                          v
                                                  Phase 5 板端微调 + 检测级验收
                                                          │
                                                          v
                                                  Phase 6 交付
```

### 3.2 三个候选架构（@ `1×1×288×384`，纯卷积 MAC）

| # | 架构 | 参数 | INT8 | GFLOPs | 输出 | 预训练 | RKNN 风险 |
|---|---|---|---|---|---|---|---|
| **C1** | YOLOv8n backbone+neck(0–21) + `SkyHead` | 3.09 M | 3.09 MB | 2.26 | 96×72 (stride 4) | COCO 可迁移 | 低；neck 的 nearest 上采样要验 |
| **C2** | Tiny 普通卷积 U-Net 16/32/64 | 95 K | 95 KB | 1.03 | 192×144 (stride 2) | 无 | **最低**，全稠密卷积 |
| **C3** | Tiny depthwise U-Net 32/64/128 | 49 K | 50 KB | 0.46 | 192×144 (stride 2) | 无 | **未知**，depthwise 在 RK3588 MAC 阵列利用率差 |

核心矛盾三角（PC 无法判定、必须上板）：
- 权重流量（6 MB SRAM） → 偏爱少参数：C3 > C2 >> C1
- 算力总量 → 偏爱 depthwise：C3 最优
- NPU 三核 MAC 阵列利用率 → 偏爱稠密卷积：C1/C2 优于 C3

→ **三候选都训、都转、都上板测**，不在 PC 上押注。

### 3.3 关键设计决策（一句话总览，论证见 SKY_TRAINING.md）

1. **单通道 Y 输入**：教师 RGB vs 亮度 300 帧 IoU 0.995，色度几乎无损；省部署内存拷贝。
2. **软 alpha 不二值**：硬阶跃被双线性放大成锯齿，会被帧差/梯度类下游算法当真实边缘，产生假目标。
3. **蒸馏不是监督**：用教师概率图作软标签，专门培养"软 alpha"的渐变性质；二值硬标签会废掉这个设计。
4. **`p^γ` 偏向项**：γ≈1.3 是**唯一**偏向超参；把天空置信度整体压低，对应你定的"宁可啃天空，不能留地面"。
5. **分块按 40 帧不按帧随机**：相邻帧近重复，随机划分会让 IoU 虚高到失去意义。
6. **几何增广图与概率图同一套仿射**：错误不会崩，只会安静学废，要单测守。
7. **范围抖动吃掉 full/limited range 未知**：50% 抖动，让量化不被未知阻塞。
8. **天花板是教师**：无人工真值时学生只能逼近不可能可靠超过；突破唯一出路是标 100 帧。

---

## 4. 分阶段执行步骤

> 每个 Phase 必须满足"判定门槛"才能进下一个。门槛不达标走"失败回路"，不能跳步硬扛。

### Phase 0：教师软标签 + 数据基础设施

**目标**：把 `.npy` 概率图转为 16-bit PNG 软标签，搭好 Dataset / engine 骨架 / 实现正确性单测。

**前置**：v3 的 `work/prob/*.npy` 全部存在（已验证 2630 个）。

**步骤**：

1. **打包转换器** `skytrain/labels.py`：
   - `prob_to_png16(npypath) -> png16path`：`uint16 = round(prob * 65535)`，存为 16-bit PNG。
   - 选 16-bit 而非 8-bit：8-bit 在 0.1–0.9 边界带量化太粗。
   - 输出至 `output/teacher_prob16/{1,2,3}/*.png`。
2. **分块脚本** `skytrain/splits.py`：
   - 按 §1.3 规则生成 `output/split/blocks.csv`，列：`domain, block_idx, frame_stem, split`。
   - 排除三帧。
   - 预期约 65 训练块 / 7 验证块。
3. **Dataset** `skytrain/data.py`：
   - `YSkyDataset(frames, net_hw=(288,384), training, gamma, prob_dir, img_dir)`。
   - `y_plane()`：BGR→RGB→GRAY(Rec.601)，与 ISP 写 Y 权重一致；50% limited-range 抖动。
   - 几何增广：水平翻转 / ±3° 旋转 / 0.9–1.1 缩放中心裁回；**对图与概率图用同一仿射矩阵 `m`**。
   - 概率图插值**必须用 `INTER_LINEAR`**，绝不 nearest。
   - 归一化：`x = (y/127.5 - 1.0)[None]`，`t = p[None] ** gamma`。
   - 标签降采样到输出尺寸用 bilinear，损失定义域与输出域一致。
4. **engine 骨架** `skytrain/engine.py`：
   - train_loop(model, loader, optimizer, scheduler, epochs, ...) —— 三候选共用。
   - AdamW lr=3e-4 cosine→1e-5 wd=1e-4 warmup 3 epoch；EMA；AMP。
   - 每 epoch 算 `leakage`/`erosion`，按 §6.1 checkpoint 选择规则存 best。
5. **正确性单测** `tests/test_skytrain_sanity.py`（或 `tools/sanity_geo.py`）：
   - 输出形状：`1×1×288×384 → 1×1×72×96`（C1）或 `1×1×144×192`（C2/C3）。
   - stride 整除：288、384 都被 32 整除。
   - **增广对齐**：构造对角梯度概率图，`mean(|p_warped − warp(p_original)|) < 1e-6`。
   - 归一化往返：Y=0/127.5/255 处 = -1/0/+1。
   - γ 单调：`p=0.5` 时 `p^1.3 < 0.5`；`p=0`、`p=1` 不动点。
   - Sigmoid 单调：logit 单调增时 alpha 单调增。

**产出**：
- `output/teacher_prob16/{1,2,3}/*.png`（2627 张）
- `output/split/blocks.csv`
- `skytrain/{__init__.py, arch.py, labels.py, splits.py, data.py, losses.py, engine.py, export.py, dump_teacher.py}`
- `tools/{eval_checkpoint.py, sanity_geo.py}`

**判定门槛**：
- 一个 batch 前向反向跑通；
- `device='xpu'` 不报缺算子（如缺则退 `cpu`，16 核）；
- **增广对齐单测**通过（这层最重要的一个）；
- 输出形状无余数。

**失败回路**：
- XPU 缺算子 → 不强行换算子，直接退 CPU；C2/C3 结构简单，CPU 也能在 20–40 min 跑完。
- 增广对齐失败 → 重写 `geometry_matrix()` 同步路径，不存在脱敏修复。

---

### Phase 1：gamma 标定（关键路径）

**目标**：定出满足 `leakage ≤ 1%` 的最小 `erosion` 点，**冻结 γ**。

**前置**：Phase 0 通过。

**为什么先于训练**：γ 定了三候选才能在同一工作点上比较，否则"哪个更好"退化成"哪个偏向调得更巧"。

**步骤**：

1. 固定 C2（参数量适中、无预训练干扰、最稳定）。
2. 扫 `γ ∈ {1.0, 1.3, 1.6, 2.0}`，各跑短训（30–40 epoch 即可看清趋势）。
3. 评测：在验证块上画 `leakage–erosion` 曲线。
4. 选满足 `leakage ≤ 1%` 的最小 `erosion` 点对应的 γ。

**产出**：
- `runs/phase1_gamma_sweep/{gamma_1.0, gamma_1.3, gamma_1.6, gamma_2.0}/val.csv`
- `output/gamma_curve.png`

**判定门槛**：
- 至少一个 γ 满足 `leakage ≤ 1%`；
- 选出γ冻结，后续 Phase 2 不得再调。

**失败回路**：
- 全部 γ 都 `leakage > 1%` → **不是训练失败，是 γ 太小**；扩 γ 上限到 2.5、3.0 再扫，**不要加容量或加 epoch**。
- 若 γ→∞ 仍不达标 → 退到 §7「教师天花板」风险，标 100 帧人工真值。

---

### Phase 2：三候选横向训练

**目标**：在冻结 γ、固定 splits 下，三候选各训一轮，PC 侧定泄漏/侵蚀工作点。

**前置**：Phase 1 通过，γ 冻结。

**步骤**：

1. 三候选共用同一 `engine.py`、同一 `data.py`、同一 γ、同一 splits、同一 epoch/batch。差异只在 `arch.py`。
2. 训练超参：`batch=32, epochs=120 (C1) / 80 (C2/C3), device='xpu' or 'cpu'`。
3. C1 权重迁移：层 0 按输入通道求和 `Conv2d(3,64,3,2) → Conv2d(1,64,3,2)`，层 1–21 原样搬 COCO 预训练。
4. 每 epoch 在验证块算 `leakage`/`erosion`（双线性放大回 640×480 再算）。
5. 按 §6.1 checkpoint 选择规则存 best.pt + EMA 副本。

**产出**：
- `runs/{C1, C2, C3}/best.pt`
- `runs/{C1, C2, C3}/val.csv`（每 epoch 的 leakage/erosion/F1/IoU）
- `runs/{C1, C2, C3}/curve.png`

**判定门槛**：
- 至少一个候选 `leakage ≤ 1%` 且 `erosion ≤ 5%`；
- 全部不达标 → 回 Phase 1 重标 γ。

**失败回路**：
- 单候选不达标 → 不碍事，继续训另两个，三候选都跑完再排。
- 三候选全不达标 → 回 Phase 1。
- C1 缺算子 → 退 CPU；CPU 仍崩 → 该候选淘汰（不算失败，是数据）。

---

### Phase 3：输出分辨率消融

**目标**：验证 C1 在 stride 4 (96×72) 是否够；若草叶帧敏感，提分辨率而非换骨干。

**前置**：Phase 2 至少一候选达标。

**步骤**：

1. 对达标候选补 stride 2 / 8 两档训练（小规模调，~30 epoch 看趋势）。
2. 在验证块上专门看草叶剪影帧的 `erosion`。

**判定门槛**：
- 输出保真度预测（SKY_TRAINING.md §1.2 表）得到验证：96×72 的 min IoU ≈ 0.84，主要损失在草叶帧；
- 若草叶帧的 `erosion` 在 stride 2 下显著改善 → 提输出到 stride 2；否则保持 stride 4。

**失败回路**：
- 提分辨率导致 RKNN 漂移超门槛 → 退回 stride 4，接受草叶帧损失。

---

### Phase 4：RKNN INT8 转换 + 上板 bench

**目标**：三候选全转 RKNN INT8，**与检测器并发**地测延迟、漂移、掉帧。

**前置**：Phase 2 通过；Linux/WSL 环境可用；RK3588 板可访问。

**步骤**：

1. PC 侧导出 ONNX（按 SKY_TRAINING.md §2.4 顺序）：
   - BN 折叠（`torch.nn.utils.fuse_conv_bn_eval`）；
   - 末端烘进 `Sigmoid`；
   - opset 11，`dynamic_axes={}`，batch=1，静态形状；
   - 删除常量折叠（防 OpenVINO 风格的 reshape 失败）；
   - 落档上采样算子清单（nearest / bilinear）。
2. 验 PC 侧对拍：PyTorch vs ONNX Runtime，逐像素 `max abs diff < 1e-3`；上采样落点单独核。
3. Linux 侧 `tools/convert_rknn.py --onnx <x>.onnx --target rk3588 --calib <val_block_64> --quant int8`。
   - 校准集**必须 64 张来自验证块**。
4. 板端 `board/bench_rknn.py`：
   - `core_mask=NPU_CORE_0`，另两核留给现有 YOLO；
   - 单核 fps、INT8 vs fp32 在验证块上的 `leakage`/`erosion` 漂移；
   - **关键测**：与现有 YOLO 检测器**并发**时的掉帧（独立跑 NPU 延迟对本项目无意义）。

**产出**：
- `runs/{C1, C2, C3}/onnx/<name>.onnx`
- `runs/{C1, C2, C3}/rknn/<name>.rknn`
- `runs/{C1, C2, C3}/board_bench.csv`（latency / core / drift / concurrent_fps）

**判定门槛**：
- INT8 `leakage` 不劣化（仍 ≤1%）；
- `erosion` 增幅 ≤ 0.5 个百分点；
- 单核 ≥ 30 fps；
- 与检测并发时无明显掉帧。

**失败回路**（按代价从小到大，不跳步）：
```
per-channel 量化  →  输出降到 stride 8  →  末端 Sigmoid 移到板端 CPU  →  QAT
```

---

### Phase 5：板端范围微调 + 检测级验收

**目标**：固定板端真实 NV12 范围末轮微调；用现有 YOLO 检测器做**终点指标**验收。

**前置**：Phase 4 通过；板端 NV12 范围（full 0–255 / limited 16–235）已确认。

**步骤**：

1. 末轮微调：固定真实范围，再训 5–10 epoch 微调权重。
2. 检测级验收（**唯一终点指标**）：
   - A 组：原图 640×480 → 现有 YOLO 检测结果 A；
   - B 组：叠加 alpha（地面被抑制）→ 同一 YOLO 检测结果 B；
   - 统计：
     - 地面区域假阳性下降幅度（项目立项理由，不降则无收益）；
     - 天空真目标漏检增加幅度（≤ 业务上限，上限在方案里留空，由业务定）。
3. 时序抖动测试：验证序列逐帧算 `jitter[t] = |mean(alpha[t]) − mean(alpha[t-1])|`，取 p50/p95/max。

**判定门槛**：
- 假阳性显著下降；
- 真目标漏检 ≤ 业务上限；
- `jitter.p95` 在可接受范围。

**失败回路**：
- 假阳性没降 → 不是模型问题，项目立项失败，需重新评估；
- 真目标丢了 → 回 Phase 1 调 γ，**不要调网络**；
- 软 alpha 半透明带让传统算法出伪影 → 改硬阈值边界（代价是阶梯）；
- `jitter.p95` 过大 → 板端对 alpha 做时域一阶低通（一行代码，不改模型）。

---

### Phase 6：交付

**目标**：一条命令可复现训练；文档、脚本整理。

**步骤**：
1. 把分阶段命令封装为 `skytrain/train.py --arch {C1,C2,C3}` 一键串起来。
2. 整理 `runs/<arch>/` 目录结构、`docs/` 增补 v4 入口。
3. 形成 `reproduce.md`：一条命令重训。

**产出**：
- 一个 RKNN INT8 模型 + 一条可复现训练命令 + 本文件 + SKY_TRAINING.md。

**判定门槛**：
- `python skytrain/train.py --arch <selected>` 从零开始能在一天内复现到验收门槛。

---

## 5. 代码结构

```
skytrain/
  __init__.py
  arch.py          # block/blk/dw_blk/SkyHead/TinyNet + build(name) -> (model, output_shape)
  labels.py        # .npy -> uint16 PNG 转换 + 读回
  splits.py        # 40 帧分块，输出 blocks.csv
  data.py          # YSkyDataset：Y 提取 + 范围抖动 + 同步几何增广
  losses.py        # BCE(soft, gamma) + soft_dice + leakage/erosion 度量
  engine.py        # 训练循环（三候选共用）
  export.py        # BN 折叠 + 烘 Sigmoid + ONNX 导出 + PyTorch 对拍
  dump_teacher.py  # 备用：教师推理入口（v4 实际不需要，因 .npy 已存在）
  train.py         # Phase 6 的一键入口
tools/
  eval_checkpoint.py   # 读 CSV，出 leakage/erosion/F1 报告 + 三联图
  sanity_geo.py        # 增广对齐单元测试
tests/
  test_skytrain_sanity.py
```

数据落盘：
```
output/teacher_prob16/{1,2,3}/*.png    # uint16 = prob*65535，640x480
output/split/blocks.csv                 # domain, block_idx, frame_stem, split
runs/<candidate>/best.pt
runs/<candidate>/onnx/<candidate>.onnx
runs/<candidate>/rknn/<candidate>.rknn
runs/<candidate>/board_bench.csv
```

---

## 6. 验证与测试方案（每层有明确 pass / fail）

### 6.1 第 1 层：实现正确性（秒级，每次改代码都跑）

| 测试 | 判据 |
|---|---|
| 输出形状 | `1×1×288×384 → 1×1×72×96`（C1）或 `1×1×144×192`（C2/C3），精确无余数 |
| stride 整除 | 288、384 都被 32 整除 |
| **增广对齐** | 对角梯度概率图，`mean(\|p_warped − warp(p_original)\|) < 1e-6` |
| 归一化往返 | Y=0/127.5/255 处 = -1/0/+1 |
| γ 单调 | `p=0.5` 时 `p^1.3 < 0.5`；`p=0`、`p=1` 不动点 |
| Sigmoid 单调 | logit 增时 alpha 增 |

### 6.2 第 2 层：PC 侧对拍（毫秒级）

| 测试 | 判据 |
|---|---|
| PyTorch vs ONNX Runtime | 同输入，`max abs diff < 1e-3` |
| 上采样落点 | nearest（C1 两处）、bilinear（C2/C3 两处）像素索引单独核 |
| 端到端叠加 | 640×480 → RGA 模拟缩放 → 模型 → alpha → 双线性放大 → 叠加 与 教师直接放大 比 IoU ≥ 0.98 |
| 静态形状 | `sess.get_inputs()[0].shape` 是常量列表 |

### 6.3 第 3 层：RKNN 量化漂移（Linux/WSL + 板子）

| 测试 | 判据 |
|---|---|
| INT8 vs fp32 掩码一致性 | `leakage` 不劣化、`erosion` 增幅 ≤ 0.5pp |
| 校准集来源 | 必须 64 张来自验证块 |
| 上板延迟 | 单核 ≥ 30 fps，`core_mask=NPU_CORE_0` |
| 并发掉帧 | 与检测并发时的真实工况 |
| 三候选横向 | 延迟 / 单核占用 / 并发掉帧 / 漂移 同时采 |

不达标顺序：per-channel → 降输出分辨率 → Sigmoid 移板端 → QAT。

### 6.4 第 4 层：检测级验收（终点指标）

| 数 | 期望 |
|---|---|
| 地面假阳性下降 | 显著（不降则项目无收益） |
| 天空真目标漏检增加 | ≤ 业务上限（待定） |

### 6.5 时序抖动（独立失效模式）

逐帧 `jitter[t] = |mean(alpha[t]) − mean(alpha[t-1])|`，取 p50/p95/max。
掩码闪动会被帧差类算法当成运动目标，即使每帧掩码单独看都是对的。这条必须单独出数。

### 6.6 回归基线（每阶段末重跑，防倒退）

| 基线 | 实测值 | 用途 |
|---|---|---|
| 教师 RGB vs 亮度 300 帧 | IoU mean 0.995 / median 1.000 | 单通道输入合法性依据 |
| 掩码降到 96×72 再放大 | IoU mean 0.9864 / p10 0.9797 / min 0.8440 | 输出分辨率档位选择依据 |
| 3ch vs 1ch @ 1×1×288×384 | 2.3041→2.2882 GFLOPs(-0.69%) | 防止再以为单通道省算力 |
| C1/C2/C3 参数量·GFLOPs | 见 §3.2 | 选型对照 |
| 标注吞吐 | 22 ms/帧 (OpenVINO Arc GPU) | 后端选择依据 |

---

## 7. 风险与退路

| 风险 | 严重度 | 退路 |
|---|---|---|
| 天花板是教师，无人工真值 | 高 | 人工标 100 帧作真值锚点（约 2h），若部署质量高于教师这是唯一出路 |
| 域覆盖只 3 段扫拍 | 高 | `leakage ≤ 1%` 必须在板子实际场景重测，本机数字不算数 |
| depthwise 是纯未知项（C3 PC 最优） | 中 | 不提前押注，Phase 4 三个都转都测 |
| 软 alpha 与检测器交互未验证 | 中 | 改硬阈值边界（代价阶梯），已为此保留 §6.4 测试 |
| HuggingFace 被墙 | 低 | `hf-mirror.com` |
| Arc XPU 缺算子 | 低 | 退 `cpu`（16 核）；C2/C3 结构简单风险主要在 C1 的 C2f |
| onnxruntime 静默退 CPU | 中 | 一律用 `make_session(device=...)`，启动打印实际后端，不能"没报错就是快了" |

---

## 8. 失败回路速查

```
Phase 0  XPU 缺算子           → 退 CPU
         增广对齐失败          → 重写同步路径,不要脱敏修复

Phase 1  全部 γ leakage>1%     → 扩 γ 上限,不加容量/epoch
         γ→∞ 仍不达标          → 标 100 帧真值 (§7)

Phase 2  三候选全不达标         → 回 Phase 1 重标 γ
         C1 缺算子             → 退 CPU;CPU 崩 → 该候选淘汰(是数据)

Phase 3  提分辨率 RKNN 漂移超阈 → 退回 stride 4,接受草叶帧损失

Phase 4  INT8 漂移超阈         → per-channel → 降分辨率 → Sigmoid 移板端 → QAT
         并发掉帧               → 看 §6.4 是否软 alpha 新伪影,否则硬阈值边界

Phase 5  假阳性没降             → 项目立项失败,重评
         真目标丢               → 回 Phase 1 调 γ,不调网络
         软 alpha 出伪影        → 改硬阈值(代价阶梯)
         jitter.p95 过大       → alpha 时域一阶低通(板端一行)
```

---

## 9. 交付物清单

- [ ] `output/teacher_prob16/{1,2,3}/*.png`（2627 张 16-bit 软标签）
- [ ] `output/split/blocks.csv`（分块清单）
- [ ] `skytrain/` 子包（arch/labels/splits/data/losses/engine/export/train）
- [ ] `tools/{eval_checkpoint.py, sanity_geo.py}`
- [ ] `runs/{C1,C2,C3}/best.pt`、`val.csv`、`curve.png`
- [ ] `runs/<selected>/onnx/<name>.onnx` + PyTorch 对拍报告
- [ ] `runs/<selected>/rknn/<name>.rknn` + 板端 bench 报告
- [ ] Phase 5 检测级验收报告（A/B 两组对比）
- [ ] `reproduce.md`（一条命令重训）
- [ ] 本文件 + `SKY_TRAINING.md`

---

## 10. 命令速查

```bash
PY=C:/Users/Xiao/miniconda3/envs/yolo/python.exe  # 用环境里实际可用的解释器

# Phase 0：软标签转换（无需重跑教师,.npy 已存在）
$PY skytrain/labels.py --npy-dir sky_workflow/work/prob --out output/teacher_prob16

# Phase 0：分块
$PY skytrain/splits.py --input sky_workflow/dataset --out output/split/blocks.csv

# Phase 0：实现正确性单测
$PY tools/sanity_geo.py
$PY -m pytest tests/test_skytrain_sanity.py -v

# Phase 1：gamma 标定（C2,短训扫 4 个 γ）
$PY skytrain/engine.py --arch C2 --sweep-gamma 1.0,1.3,1.6,2.0 --epochs 40

# Phase 2：三候选训练（γ 已冻结）
$PY skytrain/engine.py --arch C1 --epochs 120 --batch 32 --device xpu --gamma <frozen>
$PY skytrain/engine.py --arch C2 --epochs 80  --batch 32 --device xpu --gamma <frozen>
$PY skytrain/engine.py --arch C3 --epochs 80  --batch 32 --device xpu --gamma <frozen>

# 评测 + 逐帧看
$PY tools/eval_checkpoint.py --arch C2 --ckpt runs/C2/best.pt --csv runs/C2/val.csv
$PY tools/eval_checkpoint.py --arch C2 --ckpt runs/C2/best.pt --show

# Phase 4：PC 侧导出 ONNX + 对拍
$PY skytrain/export.py --arch C2 --ckpt runs/C2/best.pt --onnx runs/C2/onnx/sky_c2.onnx

# Phase 4：Linux/WSL 上转 RKNN
python3 tools/convert_rknn.py --onnx sky_c2.onnx --target rk3588 \
       --calib runs/C2/val_blocks_64 --quant int8

# Phase 5：板端 bench
python3 board/bench_rknn.py --rknn sky_c2.rknn --core 0 --concurrent-yolo
```

**全量跑由用户执行，不强求我代跑。** 脚本默认 `--limit 0`（全库），`--limit N` 只做抽样检查。

---

## 11. 选型规则（Phase 4 结束后立即执行，不留讨论）

```
1. 淘汰所有 leakage > 1% 的候选。
2. 剩余候选里,先按"与检测并发时的 fps"排,不按 GFLOPs 排。
3. 前两名并列时,按 erosion 小者胜。
4. 都淘汰 -> 不是重训,是回 Phase 1 重标 γ。
```