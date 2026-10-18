# 天空掩码模型 · 训练技术方案（独立版）

> 本文自包含，不依赖其它文档即可执行。范围：接口契约 → 模型修改 → 训练方法 → 验证 → 测试 → 判定。
> 标注为「实测」的有数据支撑，标注为「预估」的尚未验证，不要当结论用。

---

## 一、接口契约（输入与输出）

这是模型与板端之间唯一需要长期稳定的东西，先定死，后面所有改动都不能违反它。

### 1.1 输入

| 项 | 规定 |
|---|---|
| 张量 | `float32`，`1×1×288×384`（H×W = 288 行 × 384 列） |
| 内容 | NV12 的 **Y 平面**（Rec.601 全分辨率亮度），由 RGA 从 640×480 缩到 384×288 |
| 归一化 | `(Y/255 − 0.5) / 0.5` → `[-1, 1]`。**不要**用 ImageNet mean/std，那是给 RGB 三通道的 |
| 板端等价写法 | `mean_values=[[127.5]]`、`std_values=[[127.5]]`，直接喂 uint8 Y，归一化折进 NPU |
| 通道顺序 | 不存在（单通道）。这是单通道输入顺带消灭的一类部署 bug |

**为什么是 384×288 而不是 640×480**：天空/地面是低频大区域，原生分辨率多花 2.8 倍算力只为省一次 RGA 缩放（<1 ms、不占 CPU）。
**为什么不是 320×240**：实测非法。YOLOv8 系最大 stride 32，240/32 = 7.5，前向直接崩、GFLOPs 返回 0。4:3 零补边的合法档位只有 `k·128 × k·96`：128×96 / 256×192 / 384×288 / 640×480。

**依据（实测）**：教师模型吃 RGB vs 吃「亮度复制成三通道」，300 帧 IoU mean = 0.995 / median = 1.000；全库 2630 帧里唯一分歧集中在 `input/3/3_000001`–`3_000003` 三帧（过曝草地特写，教师自己在 RGB 上也是错的）。所以**丢弃色度在本数据上几乎无损**。

**但要说清楚单通道省的是什么**（实测）：3ch→1ch，GFLOPs 2.3041→2.2882，只降 **0.69%**；输入张量 324 KB→108 KB。只有 stem 第一层看得见输入通道数。真正的收益在部署侧：Y 平面本身就是连续内存，可以零拷贝喂 NPU，不必连 UV 一起搬再做交织转换。

### 1.2 输出

| 项 | 规定 |
|---|---|
| 张量 | `float32`，`1×1×72×96`（C1）或 `1×1×144×192`（C2/C3），**Sigmoid 后的软 alpha ∈ [0,1]** |
| INT8 落地 | RKNN 输出用 `0–255` uint8 量化（scale=1/255），板端不还原浮点 |
| 语义 | `1` = 天空（保留），`0` = 地面（抑制） |
| 放大 | 板端 **RGA 双线性**拉回 640×480，**不额外羽化**（双线性本身带 1–2 px 过渡） |
| 叠加 | 模型不参与。你自行 `imblend` 到 RGB 或 Y 上 |

**为什么是软 alpha 不是二值掩码**（你的选择）：硬边界在 96×72 网格上是阶梯，叠加后被梯度类传统算法当成**真实边缘**，会直接产出假目标。软值让边界带是渐变。
**代价（实测）**：软 alpha 与检测器的交互未验证，半透明带可能产生新伪影。已保留退路：改回硬阈值，代价是阶梯。这条只能在第六节的检测级测试里判定。

**输出分辨率保真度（实测，200 帧：降采样再放大回 640×480 后的 IoU）**

```
输出尺寸    像素占比   IoU mean     p10      min
320x240      25.00%     0.9971   0.9957   0.9653
160x120       6.25%     0.9918   0.9876   0.9067
 96x72        2.25%     0.9864   0.9797   0.8440   <- C1 档位
 48x36        0.56%     0.9758   0.9634   0.7575
```

min=0.844 出现在草叶剪影帧。**因为下游是检测不是给人看**，锯齿不致命；但若检测级测试对这几帧敏感，**第一优先的调参是提输出分辨率，不是换骨干**。

---

## 二、模型修改方向

### 2.1 从 YOLOv8-seg 出发要改什么

先说清楚为什么不能直接拿 YOLOv8n-seg 训：

| 现状 | 问题 | 修改 |
|---|---|---|
| `Segment` 头（层 22）输出 `no = 4*reg_max + nc = 65` 通道 | 单类别**区域**任务里，64/65 个通道在做边界框回归。框对"天空"这个非连通、无边界的东西没有意义 | **整层删掉**，接自定义 `SkyHead` |
| `Proto` 的 mask 合成（proto × 系数 + resize）在 **Python 后处理**里 | 这部分不在 ONNX 图内 → 不进 NPU，板端要自己用 CPU 做矩阵乘，等于把最贵的逐像素操作放在最慢的核上 | 用 `SkyHead` 直接在图内输出完整分辨率掩码，**零后处理** |
| stem 是 `Conv2d(3, 64, 3, 2)` | 输入是单通道 | 权重沿输入通道**求和**变 `Conv2d(1, 64, 3, 2)`，其余层 1–21 原样搬 COCO 预训练 |
| 有 P3/P4/P5 三尺度，max stride 32 | 尺寸必须整除 32 | 见 §1.1 的 `k·128 × k·96` 合法档位 |
| `Proto` 上采样用 nearest 到 stride 4 | — | `SkyHead` 沿用 stride 4 输出，但需实测 RKNN 对 nearest 的支持 |

一句话：**保留 backbone + neck（层 0–21，这部分是 COCO 预训练的资产），把整个检测头换成一个 33K 参数的掩码头。**

### 2.2 三个候选（@ `1×1×288×384`，纯卷积 MAC 计数，实测）

| # | 架构 | 参数量 | INT8 权重 | GFLOPs | 预训练 | 输出 | RKNN 风险 |
|---|---|---|---|---|---|---|---|
| **C1** | YOLOv8n backbone+neck(0–21) + `SkyHead` | 3.088 M | 3.09 MB | 2.257 | 可迁移 COCO | 96×72 (stride 4) | 低；neck 的 nearest 上采样要验 |
| **C2** | Tiny 普通卷积 16/32/64 | 0.095 M | **95 KB** | 1.028 | 无 | 192×144 (stride 2) | **最低**，全稠密卷积 |
| **C3** | Tiny depthwise 32/64/128 | 0.049 M | **50 KB** | 0.458 | 无 | 192×144 (stride 2) | **未知**，depthwise 效率存疑 |

**反直觉但已实测的一点，它决定了本方案必须上板选型**：普通卷积的轻量 U-Net **省参数不省算力**——`Tiny 24/48/96` 参数量只有 yolov8n 的 1/14，GFLOPs 反而略高（2.307 vs 2.257），因为解码器在 stride-2 高分辨率上跑拼接卷积；yolov8n 靠 C2f 的 1×1 降维压住了 FLOPs。

于是 RK3588 上有三个互相拉扯的方向：

```
权重流量（6 MB 片上 SRAM，权重要流式载入）  ->  偏爱少参数：C3 > C2 >> C1
算力总量                                    ->  偏爱 depthwise：C3 最优
NPU 三核 MAC 阵列利用率                      ->  偏爱稠密卷积；depthwise 每核只做
                                                1×1×C 乘法，MAC 阵列大量空转
```

**这个三角在 PC 上无法判定**，因为 PC 的 FLOPs/参数量都不代表 NPU 的周期数。所以本方案不押注，而是三个都训、都转 RKNN、上板实测。

### 2.3 模型定义

```python
def block(ci, co, k=3):
    return [nn.Conv2d(ci, co, k, padding=k // 2, bias=False), nn.BatchNorm2d(co), nn.SiLU()]


class SkyHead(nn.Module):
    """P3/P4/P5 (stride 8/16/32) -> 单通道软 alpha @ stride 4。

    实测：输入 1x1x288x384 时输出 (1,1,72,96)，精确 stride 4；反向可通；33,025 参数。
    """

    def __init__(self, ch=(64, 128, 256), hidden=32):
        super().__init__()
        self.lat = nn.ModuleList([nn.Conv2d(c, hidden, 1) for c in ch])
        self.fuse = nn.Sequential(*block(hidden, hidden))
        self.ref = nn.Sequential(*block(hidden, hidden))
        self.logit = nn.Conv2d(hidden, 1, 1)

    def forward(self, p3, p4, p5):
        size = p3.shape[-2:]
        x = (self.lat[0](p3)
             + F.interpolate(self.lat[1](p4), size, mode="nearest")
             + F.interpolate(self.lat[2](p5), size, mode="nearest"))
        x = self.ref(self.fuse(x))
        return self.logit(F.interpolate(x, scale_factor=2, mode="nearest"))
```

```python
def blk(ci, co, s=1):
    return [nn.Conv2d(ci, co, 3, s, 1, bias=False), nn.BatchNorm2d(co), nn.ReLU6(inplace=True)]


def dw_blk(ci, co, s=1):
    return [nn.Conv2d(ci, ci, 3, s, 1, groups=ci, bias=False), nn.BatchNorm2d(ci), nn.ReLU6(True),
            nn.Conv2d(ci, co, 1, bias=False), nn.BatchNorm2d(co), nn.ReLU6(True)]


class TinyNet(nn.Module):
    """三级编码器 + 跳连解码器。无 FPN、无注意力。dw=True 即 C3。"""

    def __init__(self, ch=(16, 32, 64), dw=False):
        super().__init__()
        mk = dw_blk if dw else blk
        self.e1 = nn.Sequential(*mk(1, ch[0], 2)); self.e2 = nn.Sequential(*mk(ch[0], ch[1], 2))
        self.e3 = nn.Sequential(*mk(ch[1], ch[2], 2)); self.b = nn.Sequential(*mk(ch[2], ch[2]))
        self.d2 = nn.Sequential(*mk(ch[2] + ch[1], ch[1])); self.d1 = nn.Sequential(*mk(ch[1] + ch[0], ch[0]))
        self.out = nn.Conv2d(ch[0], 1, 1)

    def up(self, x, ref):
        return F.interpolate(x, ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        a = self.e1(x); b = self.e2(a); c = self.e3(b); d = self.b(c)
        e = self.d2(torch.cat([self.up(d, b), b], 1))
        return self.out(self.d1(torch.cat([self.up(e, a), a], 1)))
```

C2/C3 用 `ReLU6` 而非 SiLU：SiLU 的 sigmoid 在 NPU 上要额外算子且非线性区间对 INT8 校准不友好；ReLU6 是 RKNN 的一等公民，且天然有界，量化 scale 更好定。C1 保留 SiLU 是因为要继承预训练权重，改了会破坏迁移价值。

### 2.4 导出前的必做修改

按顺序，每一步都可能影响 INT8 精度：

1. **BN 折叠**（`torch.nn.utils.fuse_conv_bn_eval`）—— 少一层算子，且让量化 scale 直接落在 conv 权重上。
2. **末端烘进 `Sigmoid`** —— 图输出直接是 alpha，板端零后处理。若 §7 的漂移门槛不达标，第四顺位退路是把 Sigmoid 挪回板端 CPU 做。
3. **opset 11、`dynamic_axes={}`、batch=1** —— 静态图是 RKNN 最省事的情况。
4. **删掉所有 `torch.Size` / reshape 常量折叠**。教训来自教师模型：`tools/make_rknn_onnx.py` 的常量折叠把静态形状烘进了图，导致 OpenVINO 想 reshape 到 160 直接 `Check 'TRShape::merge_into' failed`。学生模型导出后第一件事是确认能按别的尺寸重推。
5. **上采样算子清单落档** —— 记录每个候选用了 nearest / bilinear，转换失败时这是第一排查项。

---

## 三、训练方法的修改方向

这一节写的是「与常规 YOLO 训练不一样的地方」，每一处都给出为什么。

### 3.1 不是监督训练，是蒸馏

```
常规分割：二值人工标注  ->  硬标签  ->  模型学阶跃函数
本方案：  教师概率图    ->  软标签  ->  模型学渐变函数  ->  这正是"软 alpha"要的
```

**必须新产出教师概率图**，不能用现成的 `output/label/mask_clean/`（那是二值掩码）。二值标签会把模型训成阶跃函数，双线性放大后仍是硬边，等于把 §1.2 的软 alpha 设计从根上废掉。

产出方式（实测）：`sky_only.py --device GPU` 走 OpenVINO Arc iGPU，**22 ms/帧**，全库 2630 帧约 60 秒。存 16-bit PNG：`uint16 = round(prob * 65535)`。选 16-bit 不选 8-bit 是因为 8-bit 在低概率区量化太粗，而边界带恰好集中在 0.1–0.9。

> **天花板是教师**。无人工真值时学生只能逼近、不可能可靠超过。这是本方案最大的方法论风险，见 §8。

### 3.2 损失函数：加一个偏向项

```
L = 1.0 * BCEWithLogits(logit, p ** gamma)          <- 主项，软标签
  + 0.5 * soft_dice(sigmoid(logit), p)              <- 区域项，抗前景/背景失衡
```

`p` 是教师概率，`p' = p ** gamma` 且 `gamma ≈ 1.3`：

- **为什么这样做**：你定了「宁可啃掉天空，不能留地面」。`p**gamma` 整体压低天空置信度、把天空区域往回收，误差偏向"多切天空"那一侧。
- **为什么不用类别权重 / focal**：那两个改变的是梯度分配，不改变决策阈值语义；`p**gamma` 直接改标签本身，等价于告诉模型"边界往里缩一点"，且不破坏软标签的连续性。
- `gamma` 是**唯一**的偏向超参，用 §5.2 的 leakage–erosion 曲线标定到目标点，不要同时调别的。
- 不加边界 loss、不加 CRF、不加多尺度监督。先验证简单组合的上限。

### 3.3 划分：按连续块，绝不按帧随机

手持扫拍的相邻帧近乎重复。随机划分会让验证集里满是训练帧的邻居，**指标虚高到失去意义**（这不是理论担忧，是这类数据的典型失效方式）。

- 每域内按帧号排序，40 帧一块，块序号 `i % 10 == 9` 进验证集；三个域各自独立分块。
- 预期约 65 训练块 / 7 验证块。
- INT8 校准集 **64 张必须从验证块取**（取训练集会低估量化漂移）。

### 3.4 增广：一半是禁用的

| 允许 | 禁止 | 原因 |
|---|---|---|
| 水平翻转 | 垂直翻转 | 天空在上是这个任务的先验，翻上去等于教坏 |
| ±3° 旋转 | elastic / 透视 | 地平线不该弯 |
| 随机缩放 0.9–1.1 后中心裁回 | mosaic / copy-paste | 拼接缝会被学成边界 |
| 亮度 / 对比度 / gamma 抖动 | 色彩抖动 | 输入是单通道，色彩增广无意义 |
| NV12 范围抖动（下条） | | |

**几何增广必须用同一套仿射参数同时作用于图像和概率图**，否则标签错位——蒸馏会立刻学废。这条要有对应的单元测试（§6.1）。

**范围抖动解决一个未知项**：板子 ISP 输出的是 full range (0–255) 还是 limited/video range (16–235) 目前未确认。训练时 50% 概率在两者间随机映射，把这个不确定性用增广吃掉；板端范围确认后，在末轮固定真实范围微调。**这样量化不被这个未知项阻塞。**

### 3.5 训练循环：脱离 ultralytics

**ultralytics 的 trainer 硬绑 3 通道输入**，单通道必须自己写。三个候选因此共用同一套循环（约 150 行），差异只在模型定义文件——这是横向比较成立的前提。

代价：增广、AMP、早停、EMA 都要自己实现，且**三个候选必须用完全相同的实现**，否则 §7 的对比无意义。

| 项 | 值 |
|---|---|
| 设备 | `device='xpu'`（Intel Arc 140T；已实测 ultralytics 8.4.115 认这个值，conv 前向+反向可跑通）。报缺算子则退 `cpu`（16 核） |
| 优化器 | AdamW，lr 3e-4，cosine→1e-5，wd 1e-4，warmup 3 epoch |
| batch / epoch | 32 / 120（C2/C3 收敛更快，80 可） |
| 权重迁移 | 仅 C1：层 0 按输入通道求和，层 1–21 原样搬 |
| 保存 | 按 §5.2 门槛（leakage ≤ 1% 前提下最优 F1）存 checkpoint，保留 EMA |
| 耗时 | 2627 帧 / 32 ≈ 82 iter/epoch；Arc 上**预估** 20–40 分钟/候选，三个候选一天内 |

---

## 四、代码结构与训练步骤

### 4.1 要新建的文件

```
skytrain/
  __init__.py
  arch.py          # block/blk/dw_blk/SkyHead/TinyNet + build(name) -> (model, output_shape)
  labels.py        # 教师概率图落盘（16-bit PNG）+ 读回
  splits.py        # 40 帧分块，输出 train/val 块清单 CSV
  data.py          # Dataset：Y 提取 + 范围抖动 + 同步几何增广
  losses.py        # BCE(soft, gamma) + soft_dice + leakage/erosion 度量
  engine.py        # 训练循环（三个候选共用）
  export.py        # BN 折叠 + 烘 Sigmoid + ONNX 导出 + PyTorch 对拍
skytrain/
  dump_teacher.py  # 见 Phase 0 命令
tools/
  eval_checkpoint.py   # 读 CSV，出 leakage/erosion/F1 报告 + 三联图
  sanity_geo.py        # 增广对齐单元测试
```

数据落盘结构：

```
output/teacher_prob16/{1,2,3}/*.png    # uint16 = prob*65535，640x480
output/split/blocks.csv                 # domain, block_idx, frames, split
runs/<candidate>/best.pt
runs/<candidate>/onnx/<candidate>.onnx
```

### 4.2 Dataset 核心（把契约写进代码，不留给记忆）

```python
class YSkyDataset(Dataset):
    """Y 平面 + 教师软概率图。几何增广对两者用同一套仿射。"""

    def __init__(self, frames, net_hw, training, gamma, prob_dir, img_dir):
        ...

    def y_plane(self, image_bgr):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        y = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)          # Rec.601，与 ISP 写 Y 的权重一致
        if self.training and random.random() < 0.5:        # limited-range 抖动
            y = (y.astype(np.float32) - 16) * (255.0 / 219.0)
            y = np.clip(y, 0, 255).astype(np.uint8)
        return y

    def __getitem__(self, i):
        image = cv2.imread(str(self.img_dir / self.frames[i]))
        prob = cv2.imread(str(self.prob_dir / self.frames[i]), cv2.IMREAD_UNCHANGED)
        prob = prob.astype(np.float32) / 65535.0
        m = self.geometry_matrix()                          # 同一套参数
        y = self.warp(self.y_plane(image), m, cv2.INTER_LINEAR)
        p = self.warp(prob, m, cv2.INTER_LINEAR)            # 概率图绝不能 nearest
        y = cv2.resize(y, self.net_hw)
        p = cv2.resize(p, (p.shape[1] // 4, p.shape[0] // 4))   # 标签直接降到输出尺寸
        x = torch.from_numpy((y.astype(np.float32) / 127.5 - 1.0)[None])
        t = torch.from_numpy(p[None] ** self.gamma)          # 偏向项在这里
        return x, t, p
```

标签降到输出尺寸用双线性（不是先训全尺寸再缩），这样损失的定义域和输出域一致，也省下 4 倍显存。

### 4.3 Phase 表

| 阶段 | 内容 | 预估 | 通过条件 |
|---|---|---|---|
| **0** | 教师概率图落盘、`splits.py` 分块、`data.py`、`engine.py` 骨架、`sanity_geo.py` | 半天 | 一个 batch 能前向反向；XPU 不报缺算子；增广对齐测试通过 |
| **1** | **gamma 标定**：固定 C2，扫 `gamma ∈ {1.0, 1.3, 1.6, 2.0}`，画 leakage–erosion 曲线 | 半天 | 定出满足 leakage ≤ 1% 的最小 erosion 点，`gamma` 就此冻结 |
| **2** | **三个候选各训一轮**（C1/C2/C3），`gamma` 已定、分块已定 | 1 天 | 至少一个候选 leakage ≤ 1% 且 erosion ≤ 5% |
| **3** | 输出分辨率消融：stride 2 / 4 / 8 | 半天 | 验证 §1.2 保真度预测；若草叶帧敏感，优先提分辨率 |
| **4** | 三候选全转 RKNN INT8 + 上板 bench（与检测并发） | 1 天 | §7 漂移门槛达标；单核 ≥ 30 fps |
| **5** | 固定板端真实 NV12 范围末轮微调 + **检测级验收** | 半天 | §6.4 通过 |
| **6** | 文档、脚本整理、可复现训练 | 半天 | 一条命令重训 |

阶段 1 排在阶段 2 前面是有原因的：`gamma` 定了以后三个候选才在同一个工作点上比较；否则"哪个候选更好"会退化成"哪个候选的偏向调得更巧"。

### 4.4 命令

```bash
PY=C:/Users/Xiao/miniconda3/envs/yolo/python.exe

# Phase 0a：教师概率图（GPU 后端，全库约 60 秒）
$PY skytrain/dump_teacher.py --input_dir input --out_dir output/teacher_prob16 --device GPU

# Phase 0b：分块 + 训练（单候选）
$PY skytrain/splits.py --input_dir input --out skytrain/../output/split/blocks.csv
$PY skytrain/engine.py --arch C2 --epochs 120 --batch 32 --device xpu --gamma 1.3

# 评测 + 逐帧看
$PY tools/eval_checkpoint.py --arch C2 --ckpt runs/C2/best.pt --csv runs/C2/val.csv
$PY tools/eval_checkpoint.py --arch C2 --ckpt runs/C2/best.pt --show

# 导出
$PY skytrain/export.py --arch C2 --ckpt runs/C2/best.pt --onnx runs/C2/onnx/sky_c2.onnx

# RKNN（Linux/WSL）
python3 tools/convert_rknn.py --onnx sky_c2.onnx --target rk3588 \
       --calib calibration/val_frames --quant int8
```

**全量跑由你自己执行，不要让我代跑。** 脚本默认 `--limit 0`（全库），`--limit N` 只做抽样检查。

---

## 五、如何验证（训练中/训练后，判定模型本身）

### 5.1 主指标：方向分开的两个数

对称 IoU **不作为主指标**——它把"留地面"和"啃天空"当同等错误，而你的偏好不是对称的。

```
漏抑制率 leakage = |pred_sky ∩ true_ground| / |true_ground|
    地面残留在"天空"里 -> 叠加后地面目标还活着 -> 喂给检测器假目标。这是你要避免的。

误删率 erosion  = |pred_ground ∩ true_sky| / |true_sky|
    天空被啃 -> 丢贴地平线的目标。可接受但要最小化。
```

参照物是教师清理后的掩码。**验收门槛：`leakage ≤ 1%` 是硬约束，在此前提下最小化 `erosion`。** 三个候选必须在同一门槛下比较，否则选型无意义。IoU 降级为参考列。

`leakage`/`erosion` 都要**在双线性放大回 640×480 之后**计算，而不是在 96×72 上算——放大是板端真实路径，分辨率损失必须计入。

### 5.2 checkpoint 选择规则

```python
# 只接受满足硬约束的 epoch，否则该 epoch 不参选
if leakage <= 0.01:
    score = f1(leakage, erosion)     # 在约束内取最优
```

若整个训练从未有 epoch 满足 `leakage ≤ 1%`，**这不是训练失败，是 `gamma` 太小** → 回阶段 1，而不是加容量或加 epoch。

### 5.3 时序抖动（独立于精度的失效模式）

验证序列上逐帧统计：

```
jitter[t] = | mean(alpha[t]) - mean(alpha[t-1]) |
```

取 p50 / p95 / max 三个分位。掩码闪动会被帧差类传统算法**直接当成运动目标**，即使每帧掩码单独看都是对的。这条必须单独出数，因为它和 IoU 无关，精度达标不代表抖动达标。若 p95 过大，退路是对 alpha 做时域一阶低通（板端一行代码，不改模型）。

### 5.4 人工抽查（唯一的独立裁判）

标签是自动生成的，学生天花板就是教师——**任何自动指标都不能判断"标签本身对不对"**。
每域抽 30 帧出三联图（原图 / 教师叠加 / 学生叠加），`cv2.imshow` 逐帧看。
`tools/eval_checkpoint.py --show` 提供这个视图；`tools/compare_rgb_vs_y.py --sheets bad` 只出可疑帧。

---

## 六、如何测试（分层，每层有明确 pass/fail）

### 6.1 第 1 层：实现正确性（秒级，每次改代码都跑）

| 测试 | 判据 |
|---|---|
| 输出形状 | 输入 `1×1×288×384` → 输出 `1×1×72×96`（C1）或 `1×1×144×192`（C2/C3），精确无余数 |
| stride 整除 | 288 和 384 都能被 32 整除（防住 320×240 那类非法档位） |
| **增广对齐** | 构造一张对角梯度概率图，施加增广后 `mean(\|p_warped - warp(p_original)\|) < 1e-6` → 证明图像与概率图用的是同一套仿射 |
| 归一化往返 | `(Y/255-0.5)/0.5` 在 Y=0/127.5/255 处 = −1/0/+1 |
| gamma 单调 | `p=0.5` 时 `p**1.3 < 0.5`；`p=0` 和 `p=1` 处不动点 |
| Sigmoid 单调 | 输出 ∈ [0,1] 且 logit 增大时 alpha 增大 |

**增广对齐是这层最重要的一个**：它错了不会崩，只会让模型安静地学废，而且所有验证指标照样"看起来还行"。

### 6.2 第 2 层：PC 侧对拍（毫秒级）

| 测试 | 判据 |
|---|---|
| PyTorch vs ONNX Runtime | 同一 `1×1×288×384` 输入，逐像素 `max abs diff < 1e-3` |
| 上采样落点 | 单独核 `nearest`（C1 两处）和 `bilinear`（C2/C3 两处）的像素索引，不接受"整体 IoU 还行"——整体掩盖局部错位 |
| 端到端叠加 | 640×480 → RGA 模拟缩放 → 模型 → alpha → 双线性放大 → 叠加，与"直接在 640×480 跑教师再放大"比 IoU ≥ 0.98 |
| 静态形状 | `sess.get_inputs()[0].shape` 必须是常量列表；若之后要改尺寸需重导出 |

### 6.3 第 3 层：RKNN 量化漂移（Linux/WSL + 板子）

| 测试 | 判据 / 不达标退路 |
|---|---|
| INT8 vs fp32 掩码一致性（验证块） | `leakage` 不得劣化（仍 ≤ 1%），`erosion` 增幅 ≤ 0.5 个百分点 |
| 校准集来源 | **必须 64 张来自验证块**。来自训练集会系统性低估漂移 |
| 上板延迟 / 单核占用 | **单核 ≥ 30 fps**；`core_mask=NPU_CORE_0`，另两核留给现有 YOLO |
| **与检测并发时的掉帧** | 这才是真实工况。单独跑 NPU 的延迟数字对本项目没有意义 |
| 三候选横向 | 采 延迟 / 单核占用 / 并发掉帧 / 漂移，这是 §2.2 那个三角矛盾的唯一裁判 |

**不达标处理顺序（按代价从小到大，不要跳步）**：
```
per-channel 量化  ->  输出降到 stride 8  ->  末端 Sigmoid 改到板端做  ->  QAT
```

### 6.4 第 4 层：检测级验收（**真正的终点指标**）

拿你现有的 YOLO 检测器，在验证块上跑两组，各出完整的检测指标：

```
A 组：原图 640x480                     ->  检测结果 A
B 组：叠加 alpha 后（地面被抑制）      ->  检测结果 B
```

统计两个数：

| 数 | 期望 |
|---|---|
| 地面区域**假阳性**下降幅度 | 显著下降。这是整个项目的立项理由，不降则项目无收益 |
| 天空**真目标**漏检增加幅度 | ≤ 可接受上限（这个上限要你用业务定，方案里先留空） |

**只有这组数能证明整条链路有效**，前面所有 IoU / leakage / erosion 都是中间量。
若假阳性没降，或真目标丢了：**回 §3.2 调 `gamma`，不要调网络。**
若软 alpha 的半透明带让传统算法产生新伪影：退路是改硬阈值边界（§1.2）。

### 6.5 回归基线（每阶段末重跑，防倒退）

| 基线 | 实测值 | 用途 |
|---|---|---|
| 教师 RGB vs 亮度 300 帧 | IoU mean 0.995 / median 1.000 | 单通道输入的合法性依据 |
| 掩码降到 96×72 再放大 | IoU mean 0.9864 / p10 0.9797 / min 0.8440 | 输出分辨率档位选择依据 |
| 3ch vs 1ch @ `1×1×288×384` | 2.3041 → 2.2882 GFLOPs（−0.69%） | 防止再有人以为单通道省算力 |
| C1/C2/C3 参数量·GFLOPs | 见 §2.2 表 | 选型对照 |
| 标注吞吐 | 22 ms/帧（OpenVINO Arc GPU） vs 154 ms（onnxruntime CPU） | 后端选择依据 |

---

## 七、判定与交付

**最终交付**：一个 RKNN INT8 模型 + 一条可复现训练命令 + 本文件。

**选型规则**（阶段 4 结束后立刻执行，不留讨论）：

```
1. 淘汰所有 leakage > 1% 的候选。
2. 剩余候选里，先按「与检测并发时的 fps」排，不按 GFLOPs 排。
3. 前两名并列时，按 erosion 小者胜。
4. 都淘汰 -> 不是重训，是回阶段 1 重标 gamma。
```

**排除的帧**：`3_000001`、`3_000002`、`3_000003`（教师在此不稳定，且 RGB 上也是错的）→ 训练库 2627 帧。

---

## 八、风险与退路

| 风险 | 严重度 | 退路 |
|---|---|---|
| **天花板是教师**：无人工真值，学生只能逼近不可能可靠超过 | 高 | 人工标 100 帧作真值锚点，约 2 小时。若部署质量要求高于教师，这是唯一出路 |
| **域覆盖只有 3 段扫拍**：换季节/天气/镜头大概率掉点 | 高 | `leakage ≤ 1%` 门槛必须在板子实际场景重测，本机数字不算数 |
| **depthwise 是纯未知项**：C3 在 PC 上参数和算力都最优 | 中 | 不提前押注，阶段 4 三个都转都测 |
| **软 alpha 与检测器交互未验证**：半透明带可能让传统算法出伪影 | 中 | 改硬阈值边界，代价是阶梯；已为此保留测试（§6.4） |
| **HuggingFace 被墙**：下载预训练权重失败 | 低 | 用 `hf-mirror.com` |
| **Arc XPU 缺算子**：训练中途报不支持 | 低 | 退 `cpu`（16 核），慢但可用；C2/C3 结构简单，风险主要在 C1 的 C2f |
| **onnxruntime 静默退回 CPU**：本机只有 CPU/Azure EP，provider 探测不报错 | 中 | 一律用 `sky_only.make_session(device=...)`，非 `cpu` 走 OpenVINO；启动即打印实际后端，不要相信"没报错就是快了" |
| 默认 `--device cpu` 是刻意的 | — | 保住已有 2630 个标签的可复现性；换后端要显式传参 |

---

## 九、相关文件

| 文件 | 作用 |
|---|---|
| `sky_only.py` | 批量标注；`make_session(device)` 是后端选择入口；`sky_probability()` 返回原分辨率概率图 |
| `tools/compare_rgb_vs_y.py` | 图像三联图对比，`--show` 逐帧、`--sheets {all,bad,none}` |
| `tools/compare_rgb_vs_y_video.py` | 视频三联图，固定输出 fps（writer 按拼接行尺寸惰性打开） |
| `tools/bench_pc_backends.py` | PC 后端基准（onnxruntime-CPU / OpenVINO CPU/GPU/NPU） |
| `tools/clean_labels.py` | 连通域清理标签噪声 |
| `tools/label_qc.py` | 时序跳变质检 |
| `tools/make_rknn_onnx.py` | 教师 ONNX 裁剪为单输出（注意：常量折叠会烘死形状） |
| `tools/convert_rknn.py` | Linux 侧 ONNX→RKNN |
| `board/bench_rknn.py` | 板端延迟/fps 实测 |
| `models/skyseg_rknn.onnx` | 教师（单输出，`1×3×320×320`，需 ImageNet 归一化） |
| `output/label/mask_clean/{1,2,3}/` | 二值掩码 2630 张（**蒸馏不要用这个，要用概率图**） |
| `TRAIN_PLAN.md` | 上一版方案（v3），本文件是其替代；冲突处以本文件为准 |
