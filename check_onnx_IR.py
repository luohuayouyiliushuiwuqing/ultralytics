# coding=utf-8
# vim:set fileencoding=utf-8:
import os

import onnx


def check_onnx_ir_version(model_path, ir_version):
    try:
        # 加载 ONNX 模型
        model = onnx.load(os.path.join(model_path, "best.onnx"))
        # 获取 IR 版本
        print(f"ONNX 模型的 IR 版本是: {model.ir_version}")

        model.ir_version = ir_version
        onnx.save_model(model, os.path.join(model_path, "best_ir8.onnx"))
        print(f"ONNX 模型的 new IR 版本是: {model.ir_version}")
        return ir_version
    except FileNotFoundError:
        print(f"错误: 文件 {model_path} 未找到")
    except Exception as e:
        print(f"错误: 加载模型时发生异常: {e}")


# 使用示例
if __name__ == "__main__":
    model_path = "/data/runs/v8/Aviation-UAV-Drone/train_server/yolov8n/weights"  # 替换为你的 ONNX 模型路径
    check_onnx_ir_version(model_path, 8)
