# 数据集说明 — 天空分割（sky segmentation）

本文介绍天空数据集是如何生成的、每个文件的含义，以及两条训练路径分别消费哪些数据。
所有数据都存放在 `datasets/` 下；所有代码都位于 `sky_workflow/`（YOLO-seg 路径）与
`skytrain/`（学生模型蒸馏路径）。

---

## 1. 构建方式

```bash
# 教师模型掩码 + 概率图 + 多边形标签 + YOLO-seg 数据集（+ 校验）
python sky_workflow/build_dataset.py

# 学生数据集（软标签、切分、train/val 链接）+ 训练 + 导出 ONNX
python -m skytrain.run all --arch C2 --epochs 100 --device cuda
```

两者都是幂等的 —— 重复运行安全且开销很低。

---

## 2. 目录结构

```
datasets/
├── images/                  SOURCE（源）帧 —— 平铺存放，.jpg/.png，从不修改
│                            共 342 帧：201 张 d3_*，141 张 demo_*
│
├── work/                    教师阶段生成的中间产物
│   ├── mask/                 二值天空掩码，255=天空 0=地面      (stage 01)
│   ├── prob/                 float32 .npy 概率图，480x640       (stage 01)
│   ├── mask_clean/           去除斑点/孔洞后的掩码              (stage 02)
│   ├── labels/               YOLO 多边形 .txt                  (stage 04)
│   ├── qc_frames.csv         时间平滑度质检报告                (stage 03)
│   ├── qc_suspects.png       质检拼图                          (stage 03)
│   ├── polygon_stats.csv     逐帧多边形/孔洞统计               (stage 04)
│   ├── verify_worst.png      往返校验对照表                     (stage 08)
│   └── excluded.txt          人工剔除的帧                      (stage 01 复核)
│
├── sky_seg/                 YOLO-seg 数据集  -> 06_train_yolo.py
│   ├── data.yaml            nc:1, names: 0=sky
│   ├── manifest.csv         frame,stem,sequence,block,split
│   ├── images/{train,val}/  262 / 80
│   └── labels/{train,val}/  YOLO 多边形 .txt，每帧一个
│
├── teacher_prob16/          学生软标签，16 位 PNG（uint16 = prob*65535）
│   ├── d3/   201 张 *.png
│   └── demo/ 141 张 *.png
│
├── skytrain_data/           学生数据集根目录  -> skytrain
│   ├── blocks.csv           domain,block_idx,frame_stem,split
│   └── dataset/images/{train,val}/   281 / 61  （符号链接到 datasets/images）
│
└── runs_skytrain/           学生模型检查点、训练历史、扫描汇总
    └── C2_g1.3/{best.pt,history.json,c2_sky.onnx}
```

帧一律使用**符号链接**，从不复制：不额外占用磁盘，且数据集不会与源帧树产生偏差。

---

## 3. 文件格式

| 文件 | 格式 | 含义 |
|---|---|---|
| `work/mask/*.png` | 8 位灰度图 | 255 = 天空，0 = 地面（`prob >= 0.5`） |
| `work/prob/*.npy` | float32，480×640 | 教师模型天空概率，取值 [0,1] |
| `work/labels/*.txt` | YOLO-seg 多边形 | `0 x1 y1 x2 y2 ...` 已归一化到该帧自身的 W/H；类别 0 = 天空 |
| `teacher_prob16/*.png` | 16 位灰度图 | `uint16 = round(prob * 65535)`；读回时除以 `65535.0` |
| `sky_seg/data.yaml` | YOLO 数据集 | `path`、`train`、`val`、`nc: 1`、`names` |

**为什么软标签用 16 位：** 8 位只有 256 级（约 0.39% 步长），16 位有 65536 级
（约 0.0015%）。0.1–0.9 的过渡带正是软 alpha 必须保持平滑的区域。之所以用 PNG（而非
`.npy`），是因为加载器使用 cv2，几何增强必须对图像和概率图施加相同的仿射变换。

**为什么多边形会丢失信息：** YOLO `.txt` 多边形是二值的且没有奇偶填充规则，因此
（a）软边界丢失，（b）天空内部的地面孤岛会被静默地填成天空 —— 实测最多可达一帧的
**8.81%**。这也正是学生路径转而蒸馏稠密概率图的原因。

---

## 4. 数据集切分

**按连续的帧块切分，绝不做随机切分。** 这是手持扫拍的画面，第 N 帧与第 N+1 帧几乎是
同一张图；随机切分会让每个训练帧的近似副本进入验证集，使所有指标失去意义。

- YOLO-seg 路径（`05_build_dataset.py`）：每块 20 帧，每 4 块取 1 块进 val → **262 / 80（23.4%）**
- 学生路径（`skytrain`）：按**域**，每块 20 帧，每 4 块取 1 块进 val → **281 / 61（17.8%）**

两种切分都按域/序列进行，因此每个域都会同时出现在 train *和* val 中。
若切分结果为空，该阶段会**直接报错**，而不是交付一个不可用的数据集 ——
此时需要调整 `--block` / `--val-every`。

---

## 5. 数据清洗（交互式复核）

```bash
python sky_workflow/stages/01_label_teacher.py datasets/images \
  -o datasets/work/mask --review --device cpu        # 标注之后
  ... --review-only                                   # 只复核已有的掩码
```

并排显示帧及其掩码；按键含义：

| 按键 | 动作 |
|---|---|
| `a` | 上一帧 |
| `d` | 下一帧 |
| `x` | 标记为删除 |
| `u` | 撤销 |
| `q` / `Esc` | 退出并保存 |

`x` **绝不会删除文件** —— 它只是把帧 stem 记录到 `datasets/work/excluded.txt`，
第 04、05 阶段已经会跳过这些帧。每一次标记都会被打印且可撤销（`u`，或直接编辑该文本文件）。
需要一个真实的交互式显示环境：在无显示器的机器上，复核功能会拒绝运行，而不是误读零散输入。

---

## 6. 各路径消费什么

| 路径 | 输入 | 模型 | 输出 |
|---|---|---|---|
| **YOLO-seg**（`06_train_yolo.py`） | `sky_seg/`（图像 + 多边形标签） | `yolov8n-sky1-seg.yaml`（nc=1, nm=1） | 实例掩码，mAP |
| **学生**（`skytrain`） | `skytrain_data/`（图像）+ `teacher_prob16/`（软标签） | TinyNet U-Net（C2/C3） | 单通道软 alpha，ONNX |

YOLO-seg 路径是真实的 YOLOv8n-seg，被裁剪为 1 个类别、1 个掩码原型
（86 层 / 3.08 M 参数 / 8.4 GFLOPs）。学生路径是一个自行编写、通过蒸馏训练的
网络 —— 详见 `skytrain/README.md`。

---

## 7. 当前质量（本分支，342 帧）

**标签保真度**（stage 08，多边形渲染 vs 源掩码，IoU）：

```
train  mean 0.9954  median 0.9972  p05 0.9877  min 0.8668   below 0.97:  6 (2.3%)
val    mean 0.9933  median 0.9977  p05 0.9681  min 0.9555   below 0.97:  5 (6.2%)
```

残差尾巴来自上文所述的孔洞限制，而非多边形粗糙 —— 提高 `--max-points` 无法修复它。

**已训练模型：**

| 模型 | 结果 |
|---|---|
| YOLO-seg（100 ep） | Mask mAP50 **0.686**，mAP50-95 **0.517** |
| 学生 C2（100 ep，γ=1.3） | 泄漏 **11.42%**（目标 ≤1%）—— **未通过门限** |

---

## 8. 不容忽视的注意事项

- **数据绝大多数都是天空。** 天空覆盖率中位数约 **97%**；141 张 `demo_*`
  帧中有 72 张超过 95%。多数多边形标签退化为整帧矩形，模型只要预测"全是天空"就能拿高分。
- **Mask mAP 是相对教师生成的标签来衡量的。** 它永远不可能超过教师，也不能证明教师本身是对的。
- **教师模型固定输入 320×320** 且需要 ImageNet 归一化。一帧 640×480 的图像在推理前会被压成
  正方形再拉伸回来 —— 边界位置是对的，但教师判断的是一张畸变后的图像。
- 本机**无法获取 COCO 预训练权重**（GitHub 超时，HF 镜像返回 404），因此 YOLO-seg
  这一路是从零开始训练的。
