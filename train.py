# from ultralytics import YOLO

# if __name__ == "__main__":
#     model = YOLO(r"weights\yolov12\yolo12n.pt")
#     model.train(data="data_new.yaml", # 教程中提到了这个文件的编写方式
#                 epochs=200, # 训练多少轮，每一轮指的是让模型看一遍所有的训练集
#                 lr0=0.0001,  # 再降低学习率，减少参数更新幅度
#                 batch=8, # 每次让模型看到多少个样本，电脑性能差的记得降低此数值
#                 workers=0 # 数据处理线程数，电脑性能差的记得降低此数值，写0也可以，表示不用多线程
#                 )
# model = YOLO(r"ultralytics/models/v8/yolov8n_cbam .yaml",, verbose=True)

# from ultralytics import YOLO
# import torch 
# if __name__ == "__main__":
#     device = 0 if torch.cuda.is_available() else 'cpu'
#     model = YOLO(r"ultralytics/models/v8/seg/yolov8n_cb1.yaml")

#     model.train(data="data_new.yaml",
#                 epochs=500,
#                 batch=4,
#                 imgsz=480,
#                 lr0=0.01,
#                 optimizer="SGD",
#                 workers=0,
#                 device=device,
#                 augment=True,
#                 hsv_h=0.01,
#                 hsv_s=0.01,
#                 hsv_v=0.01,
#                 degrees=5,
#                 flipud=0.0,
#                 fliplr=0.5
#                 )



import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

label_dir = r"C:/Users/86182/Desktop/ultralytics-main (3)/dataset_org/labels"
wh_ratios = []

for txt_file in tqdm(os.listdir(label_dir)):
    if txt_file.endswith(".txt"):
        with open(os.path.join(label_dir, txt_file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                # 解析YOLO标签格式：class x_center y_center w h
                _, _, _, w, h = map(float, line.strip().split())
                if h > 0:
                    wh_ratios.append(w / h)

plt.figure(figsize=(6, 3))
# 修正color参数的语法错误
plt.hist(wh_ratios, bins=10, color='#00bfff', alpha=0.7)
plt.xlabel("Bounding Box Width/Height Ratio")
plt.ylabel("Count")
plt.title("Aspect Ratio Distribution of Bounding Boxes")
plt.tight_layout()
plt.show()

