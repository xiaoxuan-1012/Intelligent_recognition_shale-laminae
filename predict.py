# Evaluate the trained model
from ultralytics import YOLO

if __name__ == "__main__": 
    model = YOLO(r"runs/detect/train3/weights/best.pt")
    model.predict(r"dataset\test\images", save=True)


