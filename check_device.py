import openvino as ov

core = ov.Core()
print("可用设备:", core.available_devices)