from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"weights\yolov12\yolo12n.pt")
    model.train(data="data_new.yaml", 
                epochs=500, 
                lr0=0.01,  
                batch=4, 
                workers=0 
                )
model = YOLO(r"ultralytics/models/v8/yolov8n_cbam .yaml",, verbose=True)

from ultralytics import YOLO
import torch 
if __name__ == "__main__":
    device = 0 if torch.cuda.is_available() else 'cpu'
    model = YOLO(r"ultralytics/models/v8/seg/yolov8n_cbam.yaml")

    model.train(data="data_new.yaml",
                epochs=500,
                batch=4,
                imgsz=480,
                lr0=0.01,
                optimizer="SGD",
                workers=0,
                device=device,
                augment=True,
                hsv_h=0.01,
                hsv_s=0.01,
                hsv_v=0.01,
                degrees=5,
                flipud=0.0,
                fliplr=0.5
                )



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
                _, _, _, w, h = map(float, line.strip().split())
                if h > 0:
                    wh_ratios.append(w / h)

plt.figure(figsize=(6, 3))

plt.hist(wh_ratios, bins=10, color='#00bfff', alpha=0.7)
plt.xlabel("Bounding Box Width/Height Ratio")
plt.ylabel("Count")
plt.title("Aspect Ratio Distribution of Bounding Boxes")
plt.tight_layout()
plt.show()

