# coding=utf-8
# vim:set fileencoding=utf-8:
import os
from pathlib import Path

import onnx

from ultralytics import YOLO

run_path = Path.cwd().parent

model_path = r"C:\Users\Xiao\Desktop\PythonProject\NudenetDetect\models\nudenet-v34-640m.pt"
# 加载模型
model = YOLO(model=model_path)
print(model.names)

# 导出针对 Intel CPU 优化的 FP16 模型
model.export(format="openvino", half=True)

# model.export(format="onnx")
