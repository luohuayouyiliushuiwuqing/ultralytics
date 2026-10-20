# 天空分割工作流（路线 A：YOLO-seg）

一句话：**用一个大而准的教师模型把无标注帧自动标成天空掩码，清洗、质检，再转成 YOLOv8-seg 能直接训练的数据集**。

```
datasets/images/                 源帧（平铺，永不修改）
  └─ 01 教师推理        → work/mask（二值掩码）+ work/prob（概率图）
      └─ 02 连通域清洗  → work/mask_clean
          └─ 03 时序质检 → work/qc_frames.csv + qc_suspects.png        (可选)
              └─ 04 掩码 → YOLO 多边形 → work/labels/*.txt
                  └─ 05 分块划分 → sky_seg/{images,labels}/{train,val} + data.yaml
                      └─ 06 训练 → runs/…/weights/best.pt
                          └─ 07 导出 ONNX → models/*.onnx
                              └─ 板端: RKNN 转换与压测（见 §9）

08 回渲校验：在 04/05 之后、06 之前，作为质检闸门运行
```

`08_verify_dataset.py` 不是流水线的末端，而是 `04`/`05` 的**质检闸门**，必须在训练前跑。

> **范围说明**：本仓库另有一条相互独立的路线 —— `skytrain/` 下的"自研学生 + 软标签蒸馏"，
> 其说明见 [../skytrain/README.md](../skytrain/README.md)，**本文档不涉及**。
> 两条路线唯一的接口是 `datasets/work/prob/`（教师概率图），其余数据与代码互不干涉。
>
> 根目录 `README.md` 是 **ultralytics 官方库的文档，与本项无关，不要改动**。

---

## 1. 快速上手

### 1.1 环境依赖

已验证可跑的组合：

| 依赖 | 版本 | 用途 |
|---|---|---|
| Python | 3.10（conda env `yolov8`） | — |
| PyTorch | 2.7.1+cu128 | 训练与导出 |
| torchvision | 0.22.1+cu128 | ultralytics 依赖 |
| ultralytics | 8.4.182（**本仓库以开发模式安装**） | 构建/训练/验证/导出 |
| opencv-python | 4.13.0 | 全流程图像读写与几何变换 |
| numpy | 2.2.6 | 全流程 |
| onnxruntime | 1.23.2 | 教师推理（默认路径）与导出校验 |
| openvino | 2024.6.0 | 教师推理可选加速（`--device GPU/NPU`） |
| onnx | 1.20.1 | 导出后的算子统计 |
| matplotlib | 3.10.3 | ultralytics 绘图（06 硬编码 `plots=True`，训练曲线与混淆矩阵） |

### 1.2 安装

```bash
git clone <本仓库> && cd ultralytics
conda create -n yolov8 python=3.10 -y && conda activate yolov8

# ultralytics 必须以开发模式安装（本路线依赖仓库内的 ultralytics 包）
pip install -e .

pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install opencv-python numpy onnxruntime onnx
pip install openvino            # 可选：教师推理走 GPU/NPU

# 教师模型（168 MB）缺失时自动下载，也可手动取：
#   主源 https://hf-mirror.com/JianyuanWang/skyseg/resolve/main/skyseg.onnx
#   备源 https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx?download=true
```

> 本机 `huggingface.co` 不可达、`hf-mirror.com` 可达，代码按"主镜像 → 备用官方"依次尝试。

### 1.3 一键跑通

把帧放进 `datasets/images/`，然后：

```bash
# 教师标注 → 清洗 → 多边形标签 → 数据集 → 质检闸门
python sky_workflow/build_dataset.py

# 训练（数据集路径由上一命令打印）
python sky_workflow/stages/06_train_yolo.py \
    --data datasets/sky_seg/data.yaml \
    --model sky_workflow/models/yolov8n-sky1-seg.yaml \
    --epochs 100 --imgsz 320 --device 0 --pretrained none
```

`build_dataset.py` **可重复执行**（幂等），中断后重跑不会损坏已有产物。

---

## 2. 目录结构

代码与数据严格分离：**代码在 `sky_workflow/`，数据与中间件在 `datasets/`。**

```
sky_workflow/
├── README.md               本文件（仅路线 A）
├── DATASET_README.md       数据格式与目录说明
├── EXECUTION_PLAN.md       分阶段执行计划
├── build_dataset.py        本路线一键流水线
├── models/
│   ├── skyseg_teacher.onnx        教师（44M 参数，裁剪成单输出，168 MB）
│   └── yolov8n-sky1-seg.yaml      学生：YOLOv8n-seg 改 nc=1, nm=1
├── stages/                 01…08 八个阶段 + 共享模块 sky_only.py
├── tools/                  诊断脚本
└── runs/                   训练产物（06 的默认 --project）

datasets/                   数据侧（本路线相关部分）
├── images/                 源帧，平铺 .jpg/.png
├── work/                   中间件
│   ├── mask/               二值掩码           (01)
│   ├── prob/               概率图 .npy        (01)  ← 同时是另一条路线的软标签来源
│   ├── mask_clean/         清洗后掩码         (02)
│   ├── labels/             YOLO 多边形 .txt   (04)
│   ├── qc_frames.csv  qc_suspects.png        (03)
│   ├── polygon_stats.csv                      (04)
│   ├── verify_worst.png                       (08)
│   └── excluded.txt        人工剔除清单       (01 --review)
└── sky_seg/                成品数据集：data.yaml + images/{train,val} + labels/{train,val}
```

---

## 3. 脚本说明

> 每个阶段脚本的 `docstring` 就是该阶段的原理和坑，改代码前先读它。

| 脚本 | 功能 | 输入 → 输出 | 关键参数 |
|---|---|---|---|
| **`build_dataset.py`** | **一键流水线**，按序调用 01→02→04→05→08 | `datasets/images` → `datasets/sky_seg` | `--images --work -o --epsilon --max-points --block --val-every --limit --qc --skip-teacher` |
| `stages/01_label_teacher.py` | 教师推理：出二值掩码 + 概率图；**带交互式复核旁支** | 帧 → `work/mask`, `work/prob` | `-o --prob -m -t --device --limit --review --review-only --excluded` |
| `stages/02_clean_labels.py` | 连通域清洗：先补天空内的洞，再去天空外的斑（**顺序反了会把掩码取反**） | `work/mask` → `work/mask_clean` | `mask_root -o --min-area --dry-run` |
| `stages/03_qc_labels.py` | 无真值下的质检：时序平滑度 + 可疑帧拼图 | `work/mask_clean` → `qc_frames.csv`, `qc_suspects.png` | `--images --report --sheet --worst --jump-p99` |
| `stages/04_masks_to_yolo.py` | 二值掩码 → YOLO-seg 多边形 `.txt` | `work/mask_clean` → `work/labels` | `--images --masks -o --epsilon --max-points --min-area --stats --excluded` |
| `stages/05_build_dataset.py` | 按**连续块**划分，组装可训练数据集（硬链接，不拷贝） | 帧 + `work/labels` → `sky_seg/` | `--images --labels -o --block --val-every --copy --excluded` |
| `stages/06_train_yolo.py` | 训练裁剪版 YOLOv8n-seg | `sky_seg/data.yaml` → `runs/…/weights/best.pt` | `--data --model --epochs --imgsz --batch --device --pretrained --max-det --val-conf` |
| `stages/07_export_onnx.py` | 导出 ONNX、做数值对齐检查、可选折叠为亮度输入 | `best.pt` → `models/*.onnx` | `--weights -o --imgsz --opset --frames --count --fold-luma` |
| `stages/08_verify_dataset.py` | **质检闸门**：标签回渲与原掩码算 IoU，训练前必跑 | `sky_seg/` + `work/mask_clean` → `verify_worst.png` | `--dataset --masks --sheet --worst --accept` |
| `stages/sky_only.py` | 共享模块：常量（ImageNet 归一化、320 输入）、`make_session`、`ensure_teacher`（教师缺失自动下载）、`Progress` 进度条 | 被 01/03/04/05/07/08 导入 | — |
| `tools/check_inference_settings.py` | 诊断：扫 `max_det` 各挡位的耗时 vs 天空覆盖率 | 权重 → 耗时/覆盖率表 | `--weights --images --labels --count --imgsz --device --max-det` |

**注意**：`07` 的默认 `--frames input`、`tools/check_inference_settings.py` 的默认
`runs/smoke/weights/best.pt` 与 `dataset/images/val` 在**本仓库不存在**（属原作者机器），
用默认值会直接失败，必须显式传参。

---

## 4. 数据准备

### 4.1 输入要求

- 位置：`datasets/images/`（平铺，`--images` 可改）
- 格式：`.jpg/.jpeg/.png/.bmp/.tif/.tiff/.webp`
- 命名：`域_帧号`，例如 `1_000001`（原数据集）或 `d3_1015_001`、`demo_960`（本仓库）
  - 域 = 首个下划线之前的前缀（决定质检分组）
  - 帧号 = 其余部分的数字串（决定块内排序）
- 尺寸：任意。教师内部固定 320×320，结果会重采样回原尺寸
- **无需任何人工标注** —— 标签全部由教师生成

### 4.2 产物格式

| 产物 | 格式 | 含义 |
|---|---|---|
| `work/mask/*.png` | 8-bit 灰度 | 255 = 天空，0 = 地面（`prob >= 0.5`） |
| `work/prob/*.npy` | float32，原帧尺寸 | 教师天空概率 ∈ [0,1] |
| `work/labels/*.txt` | YOLO-seg 多边形 | `0 x1 y1 x2 y2 …`，按该帧自身宽高归一化，类别 0 = 天空 |
| `sky_seg/data.yaml` | YOLO 数据集描述 | `path / train / val / nc:1 / names:0=sky` |
| `work/excluded.txt` | 纯文本，一行一个 stem | 人工剔除的帧，04/05 会跳过 |

**教师两个易错点，都会静默失败**：
1. 需要 **ImageNet 归一化**，直接喂 `/255` 会返回"看起来合理"的垃圾；
2. 输入固定 **320×320**，640×480 帧会先压成正方形再拉回 —— 边界位置正确，但教师判断的是一张变形的图。

**多边形会丢两类信息**（04 已量化记录）：
1. 软边界消失（多边形只有 0/1，没有 0.1–0.9 过渡带）；
2. 地面孤岛被静默填成天空（YOLO 多边形无 even-odd 填充），本仓库实测最坏占帧面积 **8.81%**。

### 4.3 划分策略

**按连续块划分，绝不按帧随机。** 手持扫拍的相邻帧近乎重复，随机划分会把训练帧的"双胞胎"
放进验证集，让所有指标虚高到毫无意义 —— 这是本数据集最容易犯、犯了又毫无感觉的错。

- 默认：每 20 帧一块，每第 4 块进验证 → 本仓库 **262 / 80（23.4%）**
- 原环境：`--block 40 --val-every 10` → 2455 / 172，三个域都进验证集

若算出空划分，05 会**直接报错**而不是产出不可用数据集（调 `--block`/`--val-every`）。

### 4.4 数据清理（交互式复核）

```bash
python sky_workflow/stages/01_label_teacher.py datasets/images \
    -o datasets/work/mask --review            # 打完标签后复核
    ... --review-only                          # 只看已有掩码，不重跑教师
```

左原图、右天空着色叠加，底部显示序号 / 文件名 / 天空占比 / 按键：

| 键 | 动作 |
|---|---|
| `a` | 上一帧 |
| `d` | 下一帧 |
| `x` | 标记删除 |
| `u` | 撤销 |
| `q` / `Esc` | 退出保存 |

按 `x` **不删任何文件**，只把该帧 stem 写进 `datasets/work/excluded.txt`，由 04/05 跳过。
每次操作都会打印，随时可用 `u` 或直接编辑文本文件撤销。需要真实交互显示（headless 会拒绝运行）。

---

## 5. 分步运行（不走一键时）

```bash
PY=/home/igs/miniconda3/envs/yolov8/bin/python

$PY sky_workflow/stages/01_label_teacher.py datasets/images \
    -o datasets/work/mask --prob datasets/work/prob --device cpu
$PY sky_workflow/stages/02_clean_labels.py datasets/work/mask \
    -o datasets/work/mask_clean
$PY sky_workflow/stages/03_qc_labels.py datasets/work/mask_clean \
    --images datasets/images --report datasets/work/qc_frames.csv \
    --sheet datasets/work/qc_suspects.png                         # 可选
$PY sky_workflow/stages/04_masks_to_yolo.py --images datasets/images \
    --masks datasets/work/mask_clean -o datasets/work/labels \
    --excluded datasets/work/excluded.txt
$PY sky_workflow/stages/05_build_dataset.py --images datasets/images \
    --labels datasets/work/labels -o datasets/sky_seg \
    --block 20 --val-every 4 --excluded datasets/work/excluded.txt
$PY sky_workflow/stages/08_verify_dataset.py --dataset datasets/sky_seg \
    --masks datasets/work/mask_clean                              # 闸门，不过别训练
$PY sky_workflow/stages/06_train_yolo.py --data datasets/sky_seg/data.yaml \
    --epochs 100 --imgsz 320 --device 0 --pretrained none
$PY sky_workflow/stages/07_export_onnx.py --weights <best.pt> --frames datasets/images
$PY sky_workflow/tools/check_inference_settings.py --weights <best.pt> \
    --images datasets/sky_seg/images/val --labels datasets/sky_seg/labels/val --device 0
```

**设备提示**：`--device` 依机器而定。本机 **CUDA 可用**（RTX 5080 / torch 2.7.1+cu128）；
**XPU 不可用**（未装 Intel 扩展），原脚本默认的 `xpu` 需改成 `0` 或 `cpu`。

---

## 6. 关键设计决定（均有实测支撑）

**划分按连续块，绝不按帧随机。** 见 §4.3。

**`nm=1`：把 32 个掩码原型砍成 1 个。** 天空是**一个**区域，32 个原型是死重；更关键的是，
常规 RKNN 部署里"原型混合 + resize 回帧分辨率"发生在 NPU 图**之外**的 CPU 上。`nm=1` 让混合退化成单通道。

```
stock nc=80 nm=32    3.41M params    3.04 GFLOPs @320
nc=1  nm=32          3.26M params    2.87 GFLOPs @320
nc=1  nm=1（本方案）  3.09M params    2.13 GFLOPs @320        教师: 44.0M / 117 GFLOPs
```

**学生用 YOLOv8n 骨干而非 MobileNetV2。** RK3588 的 MAC 阵列对 3×3 depthwise 利用率差，
稠密卷积反而更划算。

**输入走亮度（Y）。** 教师 RGB 与复制亮度在 300 帧上 mean IoU 0.995，分歧只集中在
`3_000001`–`000003`（教师自身不稳定，那三帧亮度反而更接近正确）。收益：板端不做色彩转换、
输入字节降到 1/3；代价：这三帧需从训练中排除。

**保留教师概率图 `work/prob/`。** ultralytics 只吃硬多边形标签，拿不到软标签蒸馏的收益；
这批 `.npy` 是另一条路线（`skytrain/`）的软标签来源。

**实例化头的后处理开销（实测，和"慢"直接相关）。**
smoke 训练打印 `24.3ms inference, 185.2ms postprocess` —— **后处理是网络的 7.6 倍**，
两个独立原因，都不在网络里：

1. **验证默认 `conf=0.001`。** 为画完整 PR 曲线，validator 把上千候选框喂进 Python NMS。
   同一份权重、`max_det=16`、16 帧：`conf=0.25`→0.4 ms/帧（留 0 个实例）、`0.05`→8.8 ms（4.8 个）、
   `0.001`→187.1 ms（11.0 个）。这是**统计成本，不是部署成本** —— 板子上不会有人跑 conf=0.001。
   要 mAP 可比就保留默认，嫌慢用 `--val-conf 0.25`（代价：PR 曲线被截断）。
2. **`max_det` 默认 300。** 天空是一个区域，但头每帧只吐 ~4 个实例。24 张验证帧上扫描：

```
max_det     wall      实例/帧   IoU(标签) 均值/最小   对比并集
    300    88.0 ms      4      0.9106/0.8582
     32    15.9 ms      4      0.9106/0.8582      1.0000/1.0000
      8    15.8 ms      4      0.9106/0.8582      1.0000/1.0000
      4    15.8 ms      3      0.8955/0.8582      0.9835/0.9581
      1    15.7 ms      1      0.7380/0.7296      0.8096/0.7804
```

   结论有两面，都得记下：
   - **`max_det` 300→32 是白捡的 5.5 倍**，输出逐位相同 —— 但这只在**部署侧**成立。
     验证里别压 `max_det`：它同时砍掉画 PR 曲线的候选框，两次只差这个参数的 1-epoch 跑，
     mask mAP50 从 0.535 掉到 0.14。所以 `06` 默认仍是 300，验证提速走 `--val-conf`。
   - **`max_det=1` 是坑不是优化**：模型真的把天空切成多个实例，只留最高分会丢 19% 的天空
     （IoU 0.911→0.738）。板端后处理必须**并集所有实例**（`07` 的 `sky_mask()` 就是这么做的），
     别改成 top-1。挡位取 16~32：够覆盖，又不用为 300 个空位付费。

---

## 7. 实测结果

### 7.1 原环境（2630 帧 640×480，`input/{1,2,3}`，Arc XPU）

| 阶段 | 结果 |
|---|---|
| 01 教师标注 | 2630/2630，GPU 上 36.6 帧/秒 |
| 02 清洗 | 1984 张有改动，补洞 3532、去斑 3544，单帧最大改动 584 px（0.19%） |
| 03 质检 | 邻帧跳变 median 0.25%、p90 0.82%、**max 69.1%**；p99 以上 27 帧待人工看 |
| 04 转多边形 | 2627 个标签（排除 3 帧），median 1 个多边形/帧、52 点/帧 |
| 05 划分 | train 2455 / val 172，三个域都进验证集 |
| 08 往返校验 | mean IoU **0.9948**，median 0.9954，min 0.9566；低于 0.97 的仅 2 帧 |

**怎么读 08 的数字**：`min=0.9566` 会触发 `REVIEW`，但看 `work/verify_worst.png` 就知道不是标签错误 ——
最差的全是**天空被复杂天际线压成细长锯齿带**的帧；周长巨大时，边界错 1 像素就吃掉可观面积比，
XOR 那一列显示差异确实只有 1 像素描边。该担心的是**分布尾部形状**而非最小值：`p05=0.9895`
说明 95% 的标签几乎无损；哪天 `p05` 掉到 0.95 以下，那才是 `--epsilon`/`--max-points` 太狠。

### 7.2 本仓库（342 帧 = 201×d3 + 141×demo，RTX 5080 / CUDA）

| 阶段 | 结果 |
|---|---|
| 01 教师标注 | 342/342，CPU 约 7–10 帧/秒 |
| 02 清洗 | 278 张改动，补洞 5290、去斑 2386，单帧最大 2728 px（0.89%） |
| 04 转多边形 | 342 个标签；不可表达的"洞"像素最坏 **8.81%** 帧面积 |
| 05 划分 | 262 / 80（23.4%） |
| 08 往返校验 | train mean 0.9954 / min 0.8668；val mean 0.9933 / min 0.9555 |
| 06 训练（100 ep，从零） | Mask mAP50 **0.686**、mAP50-95 **0.517** |

**`--epsilon/--max-points` 的实测影响**（342 帧）：`0.0015/120` 有 16 帧低于 0.97 门槛
（train min 0.855）；`0.0008/300` 降到 11 帧、val 均值升到 0.9933。
**提高点数只能改善边界化简，无法解决"洞"** —— 洞像素总数两次完全相同。当前默认已取后者。

### 7.3 冒烟记录（原环境）

`--epochs 1 --pretrained none` 证明数据集能被 ultralytics 吃下：1 epoch 后
`mask mAP50 = 0.535`（`runs/smoke`）、`0.14`（`runs/smoke2`，同数据、`max_det=16`）。
这两个数只说明"格式合法、能收敛"，**不代表任何可用精度**。
（`runs/smoke*` 与 `input/` 在本仓库不存在，属原作者机器。）

---

## 8. 已知问题与未解决

- **天花板是教师。** 没有人工真值，学生只能逼近、不可能超过教师。
  要突破就得人工标一批（建议 100 帧，约 2 小时）当真值锚点。
- **03 标记的 27 帧没人工看过。** `excluded.txt` 目前只有 3 帧已知错误。
  `1_000060` 附近连续 7 帧跳变超阈值，更像快速平移而非标签错误，需看 `qc_suspects.png` 确认。
- **实例数 / 覆盖率 / 耗时都是在 1-epoch 权重上量的。** 收敛后每帧吐几个实例、`max_det=16` 够不够、
  并集 vs top-1 的差距都会变。正式训练后用
  `tools/check_inference_settings.py --weights <best.pt>` 重测再定板端参数。
- **域覆盖只有 3 段扫拍。** 换天气、换镜头大概率掉点，上线前要在板子实际场景重跑 03/08。
- **letterbox 浪费约 25% 算力**（640×480 → 320×240 → 补边 320×320）。板端必须复现同样的补边，
  否则模型看到的图和训练时不一样。
- **NV12 的 Y 是 full 还是 limited range 未确认。** `07 --fold-luma` 走 full range；
  板子若出 16–235，要么在 RGA 里拉回来，要么训练时做范围抖动增广。
- **本机取不到 COCO 预训练权重**（GitHub 超时、HF 镜像 404），目前是从零训练。
  拿到 `yolov8n-seg.pt` 后用 `--pretrained yolov8n-seg.pt` 重跑，预计明显更好。
- **本仓库数据极度偏天空**（中位 97%，141 张 demo 里 72 张 >95%），
  多数多边形标签退化为整帧矩形，模型容易学成"全判天空"。
- **`07` 导出对齐待复验**：本机用 `sky_yolov8n/best.pt` 导出时，24 帧的 PyTorch vs ONNX
  parity IoU 报 **0.0000**，需排查是导出流程问题还是参数/环境问题（未解决）。

---

## 9. 板端部署（RK3588）

> 以下脚本在**本仓库不存在**，属 Linux/板子一侧，列出以便衔接。

- `tools/convert_rknn.py`：ONNX → RKNN INT8。用
  `mean_values=[[0,0,0]] std_values=[[255,255,255]]` 把 ultralytics 内部的 `/255` 折进 NPU，
  板端直接喂 uint8。校准集必须从**验证块**取，不能取训练集。
- `board/bench_rknn.py`：多核配置下的延迟/fps + INT8 相对 PC 掩码的 IoU 漂移。
- 验收门槛：INT8 vs fp32 在验证块上 mean IoU ≥ 0.98；单核 ≥ 30 fps，`core_mask` 只占一个核。
- 导出停在 NMS 之前，这部分开销在板子上是 CPU 的。板端后处理必须**并集所有实例**（见 §6）。

---

## 10. 相关文档

| 文档 | 内容 |
|---|---|
| [DATASET_README.md](DATASET_README.md) | 数据格式、目录结构、处理流程 |
| [EXECUTION_PLAN.md](EXECUTION_PLAN.md) | 分阶段执行计划与验收标准 |
| [../SKY_TRAINING.md](../SKY_TRAINING.md) | 训练记录与排查过程 |
| [../skytrain/README.md](../skytrain/README.md) | **另一条路线**（自研学生 + 软标签蒸馏），与本路线相互独立 |
| 各 stage 的 `docstring` | 该阶段原理、参数含义与坑 |
