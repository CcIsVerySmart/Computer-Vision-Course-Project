# -*- coding: utf-8 -*-
import cv2
from ultralytics import YOLO

# 模型路径
MODEL_PATH = r"C:\Users\13579\Desktop\cv\weights\best_wyc.pt"

# 加载模型
model = YOLO(MODEL_PATH)

def infer_image(image_path, save_path="result.jpg", conf=0.5):
    """
    使用YOLOv8对单张图片进行推理并保存结果
    :param image_path: 输入图片路径
    :param save_path: 保存推理结果的路径
    :param conf: 置信度阈值
    """
    # 读取图片
    img = cv2.imread(image_path)

    # 模型推理
    results = model.predict(img, conf=conf, verbose=False)

    # 绘制检测结果
    res_img = results[0].plot()

    # 保存结果
    cv2.imwrite(save_path, res_img)

    # 打印检测信息
    num_objects = len(results[0].boxes)
    print(f"检测完成，检测到目标数量: {num_objects}")
    print(f"结果已保存至: {save_path}")

    return num_objects, save_path


if __name__ == "__main__":
    # 输入图片路径
    input_image = r"C:\Users\13579\Desktop\cv\images\office_4.jpg"
    output_image = r"C:\Users\13579\Desktop\cv\test_result.jpg"

    infer_image(input_image, output_image, conf=0.5)
