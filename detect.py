from ultralytics import YOLO

ov_model = YOLO(r"C:\Users\Xiao\Desktop\PythonProject\NudenetDetect\models\nudenet-v34-640m_openvino_model")  # 注意是加载文件夹路径

# 指定推理设备，例如 CPU、GPU 或 NPU
results = ov_model(r"C:\Users\Xiao\Desktop\demo", device="intel:npu")
results[0].show()