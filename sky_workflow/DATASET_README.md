# sky_workflow 数据处理架构文档

本文只描述 **`sky_workflow` 这条链路**：从一堆无标注原图开始，经教师自动标注、清洗、质检、
转换为多边形标签，最终产出可直接喂给 ultralytics 训练的 YOLO-seg 数据集。

全文分三部分：**一、脚本功能与定位**；**二、完整运行步骤**；**三、数据整理与准备方法**。

> 相关文档：流程原理与实测结论见 [README.md](README.md)；分阶段计划见 [EXECUTION_PLAN.md](EXECUTION_PLAN.md)。
> 根目录 `README.md` 是 ultralytics 官方库文档，与本项无关。

---

# 一、脚本功能说明与用途

## 1.1 流程总览

```
datasets/images/                     源帧（平铺，全程不被修改）
  │
  ├─01 教师推理 ──> work/mask（二值掩码）+ work/prob（概率图）
  │     └─02 连通域清洗 ──> work/mask_clean
  │           └─03 时序质检 ──> work/qc_frames.csv + qc_suspects.png   （可选）
  │                 └─04 掩码 → 多边形 ──> work/labels/*.txt
  │                       └─05 分块划分 ──> sky_seg/{images,labels}/{train,val} + data.yaml
  │                             └─06 训练 ──> runs/…/weights/best.pt
  │                                   └─07 导出 ONNX ──> models/*.onnx
  │                                         └─板端：RKNN 转换与压测
  │
  └─08 回渲校验（质检闸门）：在 04/05 之后、06 之前运行

01 另带交互式复核旁支（--review）：人工剔除错误帧 -> work/excluded.txt -> 04/05 跳过
```

`08_verify_dataset.py` **不是流水线末端，而是 04/05 的质检闸门**，必须在训练前跑。

## 1.2 脚本逐一说明

| # | 脚本 | 作用 | 输入 → 输出 | 关键参数 |
|---|---|---|---|---|
| — | **`build_dataset.py`** | **一键编排**，按序调用 01→02→04→05→08 | `datasets/images` → `datasets/sky_seg` | `--images --work -o --epsilon --max-points --block --val-every --limit --qc --skip-teacher` |
| — | `stages/sky_only.py` | 共享模块：ImageNet 归一化常量与 320 输入尺寸、`make_session`、`ensure_teacher`（教师缺失自动下载）、`Progress` 进度条 | 被 01/03/04/05/07/08 导入 | — |
| 01 | `stages/01_label_teacher.py` | 教师推理：出二值掩码与概率图；**内置交互式复核旁支** | 帧 → `work/mask`、`work/prob` | `-o --prob -m -t --device --limit --review --review-only --excluded` |
| 02 | `stages/02_clean_labels.py` | 连通域清洗：先补天空内的洞，再去天空外的斑（**两步顺序反了会把掩码取反**） | `work/mask` → `work/mask_clean` | `mask_root -o --min-area --dry-run` |
| 03 | `stages/03_qc_labels.py` | 无真值下的质检：邻帧跳变报告 + 可疑帧拼图，供人工判定 | `work/mask_clean` → `qc_frames.csv`、`qc_suspects.png` | `--images --report --sheet --worst --jump-p99` |
| 04 | `stages/04_masks_to_yolo.py` | 二值掩码 → YOLO-seg 多边形 `.txt` | `work/mask_clean` → `work/labels` | `--images --masks -o --epsilon --max-points --min-area --stats --excluded` |
| 05 | `stages/05_build_dataset.py` | 按**连续块**划分并组装数据集（硬链接，不拷贝） | 帧 + `work/labels` → `sky_seg/` | `--images --labels -o --block --val-every --copy --excluded` |
| 06 | `stages/06_train_yolo.py` | 训练裁剪版 YOLOv8n-seg | `sky_seg/data.yaml` → `runs/…/weights/best.pt` | `--data --model --epochs --imgsz --batch --device --pretrained --max-det --val-conf` |
| 07 | `stages/07_export_onnx.py` | 导出 ONNX、做数值对齐检查、可选折叠为亮度输入 | `best.pt` → `models/*.onnx` | `--weights -o --imgsz --opset --frames --count --fold-luma` |
| 08 | `stages/08_verify_dataset.py` | **质检闸门**：把标签回渲成掩码，与原掩码算 IoU | `sky_seg/` + `work/mask_clean` → `verify_worst.png` | `--dataset --masks --sheet --worst --accept` |
| — | `tools/check_inference_settings.py` | 诊断：扫 `max_det` 各挡位的耗时 vs 天空覆盖率，用于定板端参数 | 权重 → 耗时/覆盖率表 | `--weights --images --labels --count --imgsz --device --max-det` |

## 1.3 模型文件

| 文件 | 说明 |
|---|---|
| `models/skyseg_teacher.onnx` | 教师模型（44M 参数，裁剪成单输出，168 MB）。缺失时自动下载 |
| `models/yolov8n-sky1-seg.yaml` | 学生定义：YOLOv8n-seg 改 `nc=1, nm=1`（86 层 / 3.08M 参数 / 8.4 GFLOPs） |

---

# 二、完整运行步骤

## 2.1 环境依赖

| 依赖 | 版本 | 用途 |
|---|---|---|
| Python | 3.10（conda env `yolov8`） | — |
| PyTorch | 2.7.1+cu128 | 训练与导出 |
| torchvision | 0.22.1+cu128 | ultralytics 依赖 |
| ultralytics | 8.4.182（**以开发模式安装本仓库**） | 构建 / 训练 / 验证 / 导出 |
| opencv-python | 4.13.0 | 图像读写与几何变换 |
| numpy | 2.2.6 | 数组计算 |
| onnxruntime | 1.23.2 | 教师推理（默认路径）与导出校验 |
| openvino | 2024.6.0 | 教师推理可选加速（`--device GPU/NPU`） |
| onnx | 1.20.1 | 导出后算子统计 |
| matplotlib | 3.10.3 | ultralytics 绘图（06 硬编码 `plots=True`） |

## 2.2 安装

```bash
git clone <本仓库> && cd ultralytics
conda create -n yolov8 python=3.10 -y && conda activate yolov8

# ultralytics 必须以开发模式安装（本链路依赖仓库内的 ultralytics 包）
pip install -e .

pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install opencv-python numpy onnxruntime onnx matplotlib
pip install openvino            # 可选：教师推理走 GPU / NPU

# 教师模型（168 MB）缺失时自动下载，也可手动取：
#   主源  https://hf-mirror.com/JianyuanWang/skyseg/resolve/main/skyseg.onnx
#   备源  https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx?download=true
```

## 2.3 执行顺序

### 一键（推荐）

```bash
# 01→02→04→05→08：教师标注 → 清洗 → 多边形标签 → 数据集 → 质检闸门
python sky_workflow/build_dataset.py

# 训练（数据集路径由上一命令末尾打印）
python sky_workflow/stages/06_train_yolo.py \
    --data datasets/sky_seg/data.yaml \
    --model sky_workflow/models/yolov8n-sky1-seg.yaml \
    --epochs 100 --imgsz 320 --device 0 --pretrained none

# 导出
python sky_workflow/stages/07_export_onnx.py --weights <best.pt> --frames datasets/images
```

`build_dataset.py` **幂等**，中断后重跑不会损坏已有产物。

### 分步

```bash
PY=/home/igs/miniconda3/envs/yolov8/bin/python

$PY sky_workflow/stages/01_label_teacher.py datasets/images \
    -o datasets/work/mask --prob datasets/work/prob --device cpu
$PY sky_workflow/stages/02_clean_labels.py datasets/work/mask \
    -o datasets/work/mask_clean
$PY sky_workflow/stages/03_qc_labels.py datasets/work/mask_clean \
    --images datasets/images --report datasets/work/qc_frames.csv \
    --sheet datasets/work/qc_suspects.png                          # 可选
$PY sky_workflow/stages/04_masks_to_yolo.py --images datasets/images \
    --masks datasets/work/mask_clean -o datasets/work/labels \
    --excluded datasets/work/excluded.txt
$PY sky_workflow/stages/05_build_dataset.py --images datasets/images \
    --labels datasets/work/labels -o datasets/sky_seg \
    --block 20 --val-every 4 --excluded datasets/work/excluded.txt
$PY sky_workflow/stages/08_verify_dataset.py --dataset datasets/sky_seg \
    --masks datasets/work/mask_clean                               # 闸门，不过就别训练
$PY sky_workflow/stages/06_train_yolo.py --data datasets/sky_seg/data.yaml \
    --epochs 100 --imgsz 320 --device 0 --pretrained none
$PY sky_workflow/stages/07_export_onnx.py --weights <best.pt> --frames datasets/images
$PY sky_workflow/tools/check_inference_settings.py --weights <best.pt> \
    --images datasets/sky_seg/images/val --labels datasets/sky_seg/labels/val --device 0
```

## 2.4 设备与网络注意事项

- **CUDA 可用**（RTX 5080 / torch 2.7.1+cu128）；**XPU 不可用**（未装 Intel 扩展）。
  原脚本默认的 `--device xpu` 需改成 `0` 或 `cpu`。
- **教师模型**：本机 `huggingface.co` 不可达、`hf-mirror.com` 可达，代码按"主镜像 → 备用官方"
  依次尝试；下载先写 `.part` 再原子替换，避免中断留下截断文件。
- **COCO 预训练权重取不到**（GitHub 超时、HF 镜像 404），目前是从零训练
  （`--pretrained none`）。拿到 `yolov8n-seg.pt` 后改用它重跑，预计明显更好。
- **部分脚本默认值指向已不存在的路径**（属原作者机器），必须显式传参：
  `07` 的 `--frames input`、`tools/check_inference_settings.py` 的
  `runs/smoke/weights/best.pt` 与 `dataset/images/val`。

---

# 三、数据整理与准备方法

## 3.1 源数据格式要求

| 项目 | 要求 |
|---|---|
| 位置 | `datasets/images/`（平铺；`--images` 可改） |
| 后缀 | `.jpg / .jpeg / .png / .bmp / .tif / .tiff / .webp` |
| 命名 | `域_帧号`，如 `1_000001`（原数据集）或 `d3_1015_001`、`demo_960`（本仓库） |
| 域 | 首个下划线**之前**的前缀 —— 决定质检时的分组 |
| 帧号 | 其余部分的数字串 —— 决定块内排序 |
| 尺寸 | 任意。教师内部固定 320×320，结果会重采样回原尺寸 |
| 标注 | **无需任何人工标注**，标签全部由教师生成 |

## 3.2 目录结构规范

代码与数据严格分离：**代码在 `sky_workflow/`，数据与中间件全在 `datasets/`。**

```
datasets/
├── images/                    源帧，平铺，全程不被修改
│
├── work/                      教师通路中间件（可整目录重建：跑 build_dataset.py 即可）
│   ├── mask/                  二值掩码，255=天空 0=地面              (01)
│   ├── prob/                  概率图 .npy，float32，原帧尺寸         (01)
│   ├── mask_clean/            清洗后掩码                             (02)
│   ├── labels/                YOLO 多边形 .txt                       (04)
│   ├── qc_frames.csv          逐帧天空占比与邻帧跳变                 (03)
│   ├── qc_suspects.png        可疑帧拼图                             (03)
│   ├── polygon_stats.csv      逐帧多边形点数/洞像素统计              (04)
│   ├── verify_worst.png       回渲校验对照图（原图|原掩码|回渲|XOR）  (08)
│   └── excluded.txt           人工剔除清单，一行一个 stem            (01 --review)
│
└── sky_seg/                   成品数据集（06 的直接输入）
    ├── data.yaml              path / train / val / nc:1 / names:0=sky
    ├── manifest.csv           frame,stem,sequence,block,split
    ├── images/{train,val}/    本仓库 342 帧时：262 / 80
    └── labels/{train,val}/    YOLO 多边形 .txt，一帧一个文件

sky_workflow/
├── models/                    教师 ONNX + 学生 yaml
└── runs/                      训练产物（06 的默认 --project）
```

**成品数据集用硬链接而非拷贝**：不占额外磁盘，且数据集不可能与源树产生偏差。
（跨文件系统时自动回退为复制；`05` 加 `--copy` 可强制复制。）

## 3.3 处理流程（逐步）

| 步 | 做什么 | 关键约定 |
|---|---|---|
| 01 | 教师对每帧推理，输出概率图与二值掩码 | 需 **ImageNet 归一化**（直接 `/255` 会得到"看起来合理"的垃圾）；输入固定 320×320，非方形帧会先压方再拉回 |
| 02 | 连通域清洗：先填天空内的洞，再去天空外的孤立斑 | **两步顺序不可颠倒**，颠倒会把掩码取反 |
| 03 | 质检：邻帧天空占比跳变过大即为可疑，出报告与拼图 | 剔除是**人工决定**，写进 `excluded.txt`，不做自动剔除（会误杀快速平移） |
| 04 | 掩码 → YOLO 多边形：取外轮廓，`approxPolyDP` 化简到点数预算内 | 只保留外轮廓；**洞无法表达**，其像素会被静默算作天空并记入 `polygon_stats.csv` |
| 05 | 按连续块切分，硬链接组装 `images/`、`labels/`，写 `data.yaml` | 见 §3.5 |
| 08 | 闸门：把 `.txt` 回渲成掩码，与原掩码算 IoU | 见 §3.7 |
| 06 | 训练裁剪版 YOLOv8n-seg | 见 §3.8 的实测数字 |
| 07 | 导出 ONNX | 板端后处理必须**并集所有实例**，不能取 top-1 |

## 3.4 产物格式详解

| 产物 | 格式 | 含义 |
|---|---|---|
| `work/mask/*.png` | 8-bit 灰度 | 255 = 天空，0 = 地面（阈值 `prob >= 0.5`，`01 -t` 可调） |
| `work/prob/*.npy` | float32，原帧尺寸 | 教师天空概率 ∈ [0,1]，保留下来供需要连续概率的下游使用；本链路生成多边形只用其阈值化结果 |
| `work/labels/*.txt` | YOLO-seg 多边形 | 每行 `0 x1 y1 x2 y2 …`，坐标按**该帧自身**宽高归一化；类别 0 = 天空；空文件表示"该帧无天空" |
| `sky_seg/data.yaml` | YOLO 数据集描述 | `path` / `train: images/train` / `val: images/val` / `nc: 1` / `names: 0: sky` |
| `work/excluded.txt` | 纯文本，一行一个 stem | 04/05 会跳过；按 **stem** 匹配，与后缀无关 |

**多边形会丢两类信息**（04 已量化记录，务必查看 `polygon_stats.csv`）：
1. 软边界消失 —— 多边形只有 0/1，没有 0.1–0.9 过渡带；
2. 地面孤岛被静默填成天空 —— YOLO 多边形无 even-odd 填充。本仓库实测最坏占帧面积 **8.81%**。

## 3.5 划分策略

**按连续块划分，绝不按帧随机。** 手持扫拍的相邻帧近乎重复，随机划分会把训练帧的"双胞胎"
放进验证集，让所有指标虚高到毫无意义 —— 这是本数据集最容易犯、犯了又毫无感觉的错。

- 本仓库默认：`--block 20 --val-every 4` → 每 20 帧一块，每第 4 块进验证 → **262 / 80（23.4%）**
- 原环境：`--block 40 --val-every 10` → 2455 / 172，三个域都进验证集

若某侧算出空划分，05 会**直接报错**而不是产出不可用数据集 —— 此时调小 `--block`
或 `--val-every`。（默认 40/10 在帧数较少时就会得出 val=0。）

## 3.6 数据清理（交互式复核）

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
每次操作都会打印，随时可用 `u` 或直接编辑文本文件撤销。需要**真实交互显示**
（headless 环境会拒绝运行，而不是消费无效按键）。

## 3.7 数据校验：怎么读 08 的数字

08 把每个 `.txt` 回渲成掩码，与该帧的源掩码算 IoU，并输出最差帧的对照图
（`原图 | 原掩码 | 回渲 | XOR`）。

**要看分布的尾部形状，不要只看最小值。** 本仓库 342 帧实测：

| 划分 | mean | median | p05 | min | 低于 0.97 |
|---|---|---|---|---|---|
| train | 0.9954 | 0.9972 | 0.9877 | 0.8668 | 6（2.3%） |
| val | 0.9933 | 0.9977 | 0.9681 | 0.9555 | 5（6.2%） |

`min=0.8668` 会触发 `REVIEW`，但那不是标签错误：最差的都是**天空被复杂天际线压成细长锯齿带**
的帧 —— 周长巨大时，边界错 1 像素就吃掉可观面积比，XOR 那一列显示差异确实只有 1 像素描边。
`p05=0.9877` 说明 95% 的标签几乎无损；**哪天 p05 掉到 0.95 以下，那才是
`--epsilon`/`--max-points` 太狠**。

`--epsilon/--max-points` 的实测影响（342 帧）：
`0.0015/120` 有 16 帧低于 0.97（train min 0.855）；`0.0008/300` 降到 11 帧。
**提高点数只能改善边界化简，无法解决"洞"** —— 洞像素总数两次完全相同。当前默认取后者。

## 3.8 本链路实测结果

### 原环境（2630 帧 640×480，`input/{1,2,3}`，Arc XPU）

| 阶段 | 结果 |
|---|---|
| 01 教师标注 | 2630/2630，36.6 帧/秒 |
| 02 清洗 | 1984 张改动，补洞 3532、去斑 3544，单帧最大 584 px（0.19%） |
| 03 质检 | 邻帧跳变 median 0.25%、p90 0.82%、max 69.1%；p99 以上 27 帧待人工看 |
| 04 转多边形 | 2627 个标签（排除 3 帧），median 1 个多边形/帧、52 点/帧 |
| 05 划分 | 2455 / 172 |
| 08 校验 | mean IoU 0.9948，median 0.9954，min 0.9566 |

### 本仓库（342 帧 = 201×d3 + 141×demo，RTX 5080 / CUDA）

| 阶段 | 结果 |
|---|---|
| 01 教师标注 | 342/342，CPU 约 7–10 帧/秒 |
| 02 清洗 | 278 张改动，补洞 5290、去斑 2386，单帧最大 2728 px（0.89%） |
| 04 转多边形 | 342 个标签；不可表达的洞像素最坏 8.81% 帧面积 |
| 05 划分 | 262 / 80（23.4%） |
| 08 校验 | 见 §3.7 |
| 06 训练（100 ep，从零） | Mask mAP50 **0.686**、mAP50-95 **0.517** |

---

# 四、已知限制（使用必读）

- **天花板是教师。** 没有人工真值，学生只能逼近、不可能超过教师。要突破须人工标一批（建议 100 帧）当锚点。
- **数据极度偏天空**（本仓库中位 97%，141 张 demo 里 72 张 >95%），多数多边形标签退化为整帧矩形，
  模型容易学成"全判天空"。
- **letterbox 浪费约 25% 算力**（640×480 → 320×240 → 补边 320×320）。板端必须复现同样的补边，
  否则模型看到的图与训练时不一致。
- **NV12 的 Y 是 full 还是 limited range 未确认**。`07 --fold-luma` 走 full range；
  板子若出 16–235，需在 RGA 拉回或训练时做范围抖动增广。
- **`max_det=1` 是坑不是优化**：模型真会把天空切成多个实例，只留最高分会丢约 19% 天空
  （IoU 0.911→0.738）。板端后处理必须并集所有实例；挡位取 16~32。
- **`07` 导出对齐待复验**：本机用 `sky_yolov8n/best.pt` 导出时，24 帧的 PyTorch vs ONNX
  parity IoU 报 0.0000，需排查是导出流程问题还是参数/环境问题。
- **域覆盖有限**（原环境仅 3 段扫拍）。换天气、换镜头大概率掉点，上线前需在板子实际场景重跑 03/08。
