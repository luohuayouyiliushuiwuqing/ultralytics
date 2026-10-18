# v4 天空掩码学生模型 — 训练与验证工作流

把「44M 参数天空分割教师」蒸馏成一个能在 RK3588 NPU 单核 ≥30 fps、与现有 YOLO
检测器并发的 INT8 单通道学生，用于把天空压低、抑制地面假目标。

本目录（`skytrain/`）是这条流水线**独立、可一键运行**的实现入口与代码集合。
配套的设计与分阶段说明见 `sky_workflow/EXECUTION_PLAN.md`（冲突时以该文件为准）。

---

## 1. 接口契约（写在代码里，不是记忆里）

| 项 | 规定 |
|---|---|
| 板端真实输入 | NV12 的 **Y 平面**（Rec.601 亮度），RGA 缩到 `1×1×288×384` |
| 归一化 | `(Y/255 − 0.5)/0.5 → [−1, 1]`（板端等价 `mean=[[127.5]] std=[[127.5]]`） |
| 模型输出 | `1×1×72×96`（C1）/ `1×1×144×192`（C2/C3），**Sigmoid 后的软 alpha** |
| 板端后处理 | RGA 双线性拉回 640×480，叠加，**零模型后处理** |
| 硬约束 | `leakage ≤ 1%`（地面残留在天空里），必须在**双线性放大回 640×480 之后**计算 |
| 规模档位 | 4:3 零补边合法档位仅 `k·128 × k·96`（320×240 实测非法） |

三候选（同条件横向比较，详见 EXECUTION_PLAN §3.2）：

| # | 架构 | 参数 | INT8 | GFLOPs | 输出 |
|---|---|---|---|---|---|
| C1 | YOLOv8n backbone+neck + 自定义 SkyHead | ~3.09M | 3.09MB | 2.26 | 96×72 |
| C2 | Tiny 普通卷积 U-Net 16/32/64 | ~95K | 95KB | 1.03 | 192×144 |
| C3 | Tiny depthwise U-Net 32/64/128 | ~49K | 50KB | 0.46 | 192×144 |

---

## 2. 目录结构

```
skytrain/
├── README.md            # 本文件
├── config.py            # 共享常量（路径/SKY_ROOT、NET_HW、GAMMA_DEFAULT 等）
├── arch.py              # C1/C2/C3 模型定义，build(name)->(model, out_shape)
├── labels.py            # 教师 .npy 概率图 -> 16-bit PNG 软标签（Phase 0）
├── splits.py            # 连续 40 帧分块切分 -> blocks.csv（Phase 0）
├── data.py              # YSkyDataset：Y 平面 + 共享仿射增广 + 范围抖动
├── losses.py            # 蒸馏损失 BCE(logit,p^γ)+0.5·soft_dice + leakage/erosion 指标
├── engine.py            # 训练循环 + γ 标定 + eval_checkpoint（验证闸门）
├── export.py            # ONNX 导出（BN 折叠 + 烘焙 Sigmoid + 算子普查）
├── dump_teacher.py      # 教师概率图重落盘（兜底，v4 已有 .npy 缓存一般不用）
├── run.py               # ★ 统一入口：一个命令跑完整条流水线
├── tools/
│   ├── sanity_geo.py    # Phase 0 闸门：图与概率图必须同一套仿射
│   ├── eval_checkpoint.py  # 验证：逐帧 leakage/erosion/F1
│   └── visualize.py     # 单帧推理可视化（原图/掩码/抑制/教师对照）
└── tests/
    ├── test_skytrain_sanity.py  # 8 项实现正确性单测
    └── run_all.py               # 免 pytest 测试运行器
```

数据资产（位于主仓库 `ultralytics/`，通过绝对路径引用）：

```
ultralytics/sky_workflow/
├── dataset/images/{train,val}/   # 2627 帧 640×480 原图（排除 3_000001/2/3）
└── work/prob/{1,2,3}/*.npy       # 2630 张全分辨率 float32 教师概率图（3.1GB）
ultralytics/output/
├── teacher_prob16/{1,2,3}/*.png  # 由 labels 转换出的 uint16 软标签
└── split/blocks.csv              # 由 splits 生成的分块切分
ultralytics/runs/                 # 训练产物：<arch>_g<γ>/{best.pt,history.json}
                                   #         <arch>_gamma_sweep.{json,png}
```

---

## 3. 环境与依赖

```bash
# 推荐环境（已验证）：conda 的 yolo 环境，torch 2.x + onnx/onnxruntime/cv2/matplotlib
conda activate yolo
python -m pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
    torch torchvision onnx onnxruntime opencv-python matplotlib numpy

# 导出 ONNX 还需 onnxscript（清华镜像无 py3.13 轮子时退回官方 pypi）
python -m pip install onnxscript
```

> torch/onnx 的 CPU 推理即可完成全部流程验证；**真机训练用 `--device cuda`**。

---

## 4. 快速开始

```bash
cd <repo-root>            # ultralytics-sky-v4 工作树根（feature 分支 feat/sky-v4）

# 0) 实现正确性闸门（秒级，无需真数据）
python -m skytrain.run test

# 1) 完整流水线（小样本先跑通，再换 GPU 全量）
python -m skytrain.run all --arch C2 --device cpu --limit 100 --epochs 20
#    全量（GPU 到位后）：
python -m skytrain.run all --arch C2 --device cuda
```

`all` 等价于依次执行：labels → splits → sweep（Phase 1，冻结 γ）→
train（Phase 2，用冻结的 γ）→ export（ONNX）。每一步都可单独跑（见下）。

---

## 5. 各子命令

| 命令 | 作用 | 关键参数 |
|---|---|---|
| `run labels` | 教师 .npy → uint16 PNG 软标签 | `--npy-dir --out --limit` |
| `run splits` | 连续 40 帧分块切分 | `--input --out` |
| `run train --arch C2` | 固定 γ 训练（Phase 2） | `--gamma --epochs --batch --device --limit` |
| `run sweep --arch C2` | γ 标定：扫 γ，逐 γ 训练+验证+出曲线（Phase 1） | `--sweep-gamma 1.0,1.3,1.6,2.0` |
| `run export --arch C2 --ckpt ... --onnx ...` | 导出 ONNX（BN 折叠+烘焙 Sigmoid） | — |
| `run eval --arch C2 --ckpt ...` | **验证闸门**：逐帧 leakage/erosion/F1 @640×480 | `--csv --out` |
| `run viz --image ... --onnx ... --out ...` | 单帧效果可视化 | `--teacher`（可选参考） |
| `run test` | Phase 0 正确性单测（8 项） | — |
| `run all` | 端到端（labels+splits+sweep+train+export） | `--arch --device --limit` |

示例（单阶段）：

```bash
# 先标定 γ（全量 40 epoch）
python -m skytrain.run sweep --arch C2 --epochs 40 --device cuda
# 看 runs/C2_gamma_sweep.png 与 .json，取推荐的 γ（如 1.3）

# 用冻结的 γ 正式训练 + 导出 + 验证
python -m skytrain.run train  --arch C2 --gamma 1.3 --epochs 80 --device cuda
python -m skytrain.run export --arch C2 \
    --ckpt runs/C2_g1.3/best.pt --onnx runs/C2_g1.3/sky_c2.onnx
python -m skytrain.run eval   --arch C2 --ckpt runs/C2_g1.3/best.pt

# 看一张图的效果
python -m skytrain.run viz \
    --image ../sky_workflow/dataset/images/val/1_000361.png \
    --onnx   runs/C2_g1.3/sky_c2.onnx \
    --teacher ../output/teacher_prob16/1/1_000361.png \
    --out    runs/visualize/1_000361.png
```

---

## 6. 验证指标（方向性，非对称 IoU）

对称 IoU 会把「留地面」和「啃天空」当同等错误，故主指标用两个**有方向**的量，
且**必须在双线性放大回 640×480 之后再算**（分辨率损失要计入）：

```
leakage = pred_sky ∩ true_ground / true_ground   # 地面残留在天空里 -> 坏（硬门槛 ≤1%）
erosion  = pred_ground ∩ true_sky  / true_sky    # 天空被啃 -> 可接受，越小越好
```

`eval` 还会输出逐像素 F1 作为辅助，并把逐帧结果写进 CSV 便于排查单帧失败。

---

## 7. 已验证状态（截至本提交）

- ✅ **Phase 0 闸门**：`run test` 8/8 通过；增广对齐 200 次随机仿射 `max_err<1e-3`；
  C2/C3 真实数据前向+反向 OK（in `32×1×288×384` → out `32×1×144×192`）。
- ✅ **流程跑通**：小样本（100 帧 × 20 epoch × 4γ）训练→评测→γ 标定曲线→导出
  ONNX→ONNXRuntime 推理全部打通；C2/C3 均导出干净、Resize 为 bilinear。
- ✅ **可视化**：验证集未见图像送入 C2 ONNX，天空/建筑结构正确（边界偏软属预期，
  见下「限制」）。
- ✅ **耗时**：CPU 上 C2 纯前向 1.55 ms/帧（≈645 fps），完整链路 3.06 ms/帧
  （≈327 fps），相对 30 fps 门槛有 >10× 余量（NPU 只会更宽裕）。

---

## 8. 已知限制 / 下一步

1. **过拟合小样本**：当前 `runs/C2_g1.3` 仅 100 帧训练，边界比教师软；**全量
   2458 帧 × 40~80 epoch（GPU）** 才会锐化。真正的 γ 标定也需全量数据。
2. **C1（SkyHead+YOLOv8n backbone）**：backbone 接法留待 Phase 2，本轮未验证导出。
3. **INT8 / 上板**：Phase 4 的 RKNN INT8 转换 + 与检测器并发帧率 + 漂移，需真机。
4. **检测级验收**：`leakage≤1%` 只是模型门槛，终点指标是「叠加后地面假阳性下降、
   真目标漏检 ≤ 业务上限」，需 Phase 5 接回检测器评测。

---

## 9. 复现命令速查（全量，GPU 到位后）

```bash
python -m skytrain.run all --arch C2 --device cuda        # 一路到底
# 或分阶段：
python -m skytrain.run sweep  --arch C2 --epochs 40 --device cuda
python -m skytrain.run train  --arch C2 --gamma <推荐γ> --epochs 80 --device cuda
python -m skytrain.run export --arch C2 --ckpt runs/C2_g<γ>/best.pt --onnx runs/C2_g<γ>/sky_c2.onnx
python -m skytrain.run eval   --arch C2 --ckpt runs/C2_g<γ>/best.pt
```
