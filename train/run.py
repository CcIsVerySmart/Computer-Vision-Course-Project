import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("C:\\Users\\13579\\Desktop\\yolov8-Gold\\ultralytics\\cfg\\models\\v8\\yolov8.yaml")
    model.load('yolov8n.pt') # 我这里用的n的权重文件，大家可以自行替换自己的版本的
    model.train(data=r'yolo-bvn2.yaml',
                cache=False,
                imgsz=640,
                epochs=150,
                batch=16,
                close_mosaic=0,
                workers=0,
                device=0,
                optimizer='SGD', # using SGD
                amp=False,# close amp
                )