# 天空分割工作流：原图 → 教师标注 → YOLO-seg 可训练数据集

一句话：**用一个大而准的教师模型把无标注帧自动标成天空掩码，清洗、质检，再转成 YOLOv8-seg
能直接训练的数据集**，从而在 RK3588 上训一个比教师轻 55 倍的学生模型。

```
input/{1,2,3}/*.png          2630 帧 640×480，手持扫拍，灰蒙天空
  └01 教师推理 ──> work/mask + work/prob
      └02 连通域清洗 ──> work/mask_clean
          └03 时序质检 ──> work/qc_frames.csv, work/excluded.txt(人工)
              └04 掩码转多边形 ──> work/labels/*.txt
                  └05 分块划分 ──> dataset/{images,labels}/{train,val} + data.yaml
                      └06 训练 ──> runs/…/weights/best.pt
                          └07 导出 ONNX + 亮度折叠 ──> models/*.onnx
                              └ Linux/板子: tools/convert_rknn.py, board/bench_rknn.py
```

`08_verify_dataset.py` 不是流水线的末端，是 `04`/`05` 的**质检闸门**，必须在训练前跑。

## 目录

| 路径 | 内容 |
|---|---|
| `stages/01…08` | 八个阶段脚本，每个的 docstring 就是该阶段的原理和坑 |
| `tools/check_inference_settings.py` | 扫 `max_det`：每个挡位的墙钟成本 vs 天空覆盖率 |
| `models/skyseg_teacher.onnx` | 教师（44M 参数，裁剪成单输出，168 MB） |
| `models/yolov8n-sky1-seg.yaml` | 学生：YOLOv8n-seg 改 `nc=1, nm=1` |
| `runs/smoke`、`runs/smoke2` | 两次 1 epoch 冒烟跑的权重和曲线，留着是为了能复现上面的耗时表 |
| `dataset/` | **成品数据集**，`data.yaml` 直接喂 ultralytics |
| `work/` | 中间产物。`prob/` 是 3.1 GB 概率图，不要了可以删 |

## 怎么跑

已经在这台机器上真跑过一遍的（01/02/03/04/05/08）不用重复。要重跑：

```bash
PY=C:/Users/Xiao/miniconda3/envs/yolo/python.exe
$PY sky_workflow/stages/01_label_teacher.py input -o sky_workflow/work/mask \
    --prob sky_workflow/work/prob --device GPU          # 74 s / 2630 帧
$PY sky_workflow/stages/02_clean_labels.py              # 用默认路径
$PY sky_workflow/stages/03_qc_labels.py                 # 出可疑帧拼图，人工判定
$PY sky_workflow/stages/04_masks_to_yolo.py
$PY sky_workflow/stages/05_build_dataset.py
$PY sky_workflow/stages/08_verify_dataset.py            # 闸门，不过就别训练
$PY sky_workflow/stages/06_train_yolo.py --epochs 100 --imgsz 320 --device xpu
$PY sky_workflow/stages/07_export_onnx.py --fold-luma
$PY sky_workflow/tools/check_inference_settings.py --weights <best.pt>   # 定 max_det
```

冒烟跑（`--epochs 1 --pretrained none`）已经证明这个数据集能被 ultralytics 吃下去：1 epoch 后
`mask mAP50 = 0.535`（`runs/smoke`）、`0.14`（`runs/smoke2`，同一份数据、`max_det=16`）。
这两个数只说明"格式合法、能收敛"，不代表任何可用精度。

## 本次实测结果

| 阶段 | 结果 |
|---|---|
| 01 教师标注 | 2630/2630，GPU 上 36.6 帧/秒 |
| 02 清洗 | 1984 张有改动，补洞 3532、去斑 3544，单帧最大改动 584 px（0.19%） |
| 03 质检 | 邻帧跳变 median 0.25%、p90 0.82%、**max 69.1%**；p99 以上 27 帧待人工看 |
| 04 转多边形 | 2627 个标签（排除 3 帧），median 1 个多边形/帧、52 点/帧 |
| 05 划分 | train 2455 / val 172，三个域都进验证集 |
| 08 往返校验 | **mean IoU 0.9948，median 0.9954，min 0.9566**，低于 0.97 的只有 2 帧 |

### 怎么读 08 的数字

`min=0.9566` 触发了 `REVIEW`，但看 `work/verify_worst.png` 就知道这不是标签错误：最差的全是
**天空被复杂天际线压成一条细长锯齿带**的帧。周长巨大时，边界错 1 像素就吃掉可观的面积比，
IoU 对这种帧天然苛刻，而 XOR 那一列显示差异确实只有 1 像素描边。

真正该担心的是**分布的尾部形状**，不是最小值：`p05=0.9895` 说明 95% 的标签几乎无损。
如果哪天 `p05` 掉到 0.95 以下，那才是 `--epsilon`/`--max-points` 太狠。

## 实例化头的额外开销（实测，和"慢"直接相关）

smoke 训练打印 `Speed: 24.3ms inference, 185.2ms postprocess per image`——**后处理是网络的 7.6 倍**。
拆开后是两个独立原因，都不在网络里：

**1. 验证默认 conf=0.001。** 为了画完整 PR 曲线，validator 把上千个候选框喂进 Python NMS。
同一份权重、`max_det=16`、16 帧：

```
conf=0.25   postprocess  0.4 ms/帧   (留 0 个实例)
conf=0.05   postprocess  8.8 ms/帧   (留 4.8 个)
conf=0.001  postprocess 187.1 ms/帧  (留 11.0 个)   ← 训练日志里那个数
```

这是**统计成本，不是部署成本**——板子上不会有人跑 conf=0.001。想要 mAP 可比就保留默认，
嫌验证太慢用 `--val-conf 0.25`（代价是 PR 曲线被截断）。
**2. `max_det` 默认 300。** 天空是**一个**区域，但头每帧只吐 ~4 个实例，NMS 之后还要给
每个保留实例插值一张掩码。`tools/check_inference_settings.py` 在 24 张验证帧上扫（Arc XPU，
`imgsz=320`，进程内墙钟，只看同一次运行内的相对差）：

```
max_det     wall      实例/帧   IoU(标签) 均值/最小   对比并集
    300    88.0 ms      4      0.9106/0.8582
     32    15.9 ms      4      0.9106/0.8582      1.0000/1.0000
      8    15.8 ms      4      0.9106/0.8582      1.0000/1.0000
      4    15.8 ms      3      0.8955/0.8582      0.9835/0.9581
      1    15.7 ms      1      0.7380/0.7296      0.8096/0.7804
```

结论有两面，都得记下：

- **`max_det` 从 300 降到 32 是白捡的 5.5 倍**，输出逐位相同——但这只在**部署侧**成立。
  验证里不要压 `max_det`：它同时砍掉画 PR 曲线用的候选框，两次只差这个参数的 1-epoch 跑，
  mask mAP50 从 0.535 掉到 0.14（各自新建权重，不算严格对照）。所以 `06` 的默认仍是 300，
  验证提速走 `--val-conf`。
- **`max_det=1` 是坑，不是优化。** 模型真的把天空切成多个实例，只留最高分那个会丢掉
  19% 的天空（IoU 0.911 → 0.738）。所以板端后处理必须**并集所有实例**，`07_export_onnx.py` 里的
  `sky_mask()` 就是这么做的，别改成取 top-1。挡位取 16~32：够覆盖，又不用为 300 个空位付费。

导出的 ONNX 停在 NMS 之前，所以这部分开销在板子上是 CPU 的。真正的免 NMS 路线是阈值化那
唯一一个掩码系数——那也是 `TRAIN_PLAN.md` 自定义学生路线唯一在这件事上的优势：它根本没有实例头。

## 关键设计决定（都有实测支撑）

**划分按连续块，绝不按帧随机。** 相邻帧近乎重复，随机划分会让验证 IoU 虚高到毫无意义——
这是本数据集上最容易犯、犯了又完全无感的错。`--block 40 --val-every 10`，且按序列分别分块，
保证三个域都出现在验证集里。

**输入走亮度。** 教师 RGB 与复制亮度在 300 帧上 mean IoU 0.995，全库分歧只集中在
`3_000001`–`000003`（教师自身不稳定，且那三帧亮度反而更接近正确）。所以：板端不做色彩转换、
输入字节降到 1/3、这三帧从训练里排除。

**学生用 YOLOv8n 骨干而不是 MobileNetV2。** RK3588 的 MAC 阵列对 3×3 depthwise 利用率很差，
稠密卷积反而更划算。实测三档：

```
stock nc=80 nm=32    3.41M params    3.04 GFLOPs @320
nc=1  nm=32          3.26M params    2.87 GFLOPs @320
nc=1  nm=1 (本方案)   3.09M params    2.13 GFLOPs @320      教师: 44.0M / 117 GFLOPs
```

`nm=1` 把 32 个掩码原型砍成 1 个，顺带解决了 RKNN 部署里"原型混合 + resize 要回 CPU 做"的老麻烦。

**`work/prob/` 保留教师概率图。** ultralytics 只吃硬多边形标签，拿不到软标签蒸馏的收益；
如果之后回到自定义学生（`TRAIN_PLAN.md` 的原始路线），这批 `.npy` 就是软标签的来源。

## 交给 Linux / 板子的一侧

- `tools/convert_rknn.py`：ONNX→RKNN INT8。`mean_values=[[0,0,0]] std_values=[[255,255,255]]`
  把 ultralytics 内部的 `/255` 折进 NPU，板端直接喂 uint8。校准集从**验证块**取，不能取训练集。
- `board/bench_rknn.py`：多核配置下的延迟/fps + INT8 相对 PC 掩码的 IoU 漂移。
- 门槛：INT8 vs fp32 在验证块上 mean IoU ≥ 0.98；单核 ≥ 30 fps，`core_mask` 只占一个核。

## 还没解决

- **天花板是教师。** 没有人工真值，学生只能逼近不可能超过教师。要突破就得人工标一批
  （建议 100 帧，约 2 小时）当真值锚点。
- **03 标记的 27 帧没人工看过。** 目前 `excluded.txt` 只有那 3 帧已知错误。`1_000060` 附近
  连续 7 帧跳变超阈值，更像快速平移而非标签错误，但需要看 `work/qc_suspects.png` 确认。
- **实例数、覆盖率和耗时都是在 1-epoch 权重上量的。** 收敛后每帧吐几个实例、`max_det=16` 够不够、
  并集 vs top-1 的差距都会变。正式训练后用 `tools/check_inference_settings.py --weights <best.pt>`
  重测一遍再定板端参数。
- **域覆盖只有 3 段扫拍。** 换天气、换镜头大概率掉点，上线前要在板子实际场景重跑 03/08。
- **letterbox 浪费约 25% 算力**（640×480 → 320×240 → 补边到 320×320）。板端必须复现同样的补边，
  否则模型看到的图和训练时不一样。真要省掉它，就得回到自定义 320×240 输入的学生。
- **NV12 的 Y 是 full 还是 limited range 未确认。** `07 --fold-luma` 走的是 full range；
  板子若出 16–235，要么在 RGA 里拉回来，要么训练时做范围抖动增广。
