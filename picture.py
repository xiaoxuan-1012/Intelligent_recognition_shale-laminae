import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import random

# ===================== 1. Configuration path=====================
INPUT_DIR = r"C:/Users/86182/Desktop/image"  
OUTPUT_DIR = r"C:/Users/86182/Desktop/result" 
TARGET_SIZE = (640, 640)
AUGMENT_NUM = 5 

# ===================== 2. Create Output Folder =====================
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===================== 3. Image enhancement function =====================
def augment_image(img):
    """
    hue fluctuation,saturation fluctuation,random rotation,random vertical flipping
    """
    # Randomly flip up and down (with a 50% probability)
    if random.random() > 0.5:
        img = cv2.flip(img, 0) 

    # 2. Random rotation (randomly ranging from 0 to 360 degrees, automatically filling the edges)
    angle = random.randint(0, 360)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    img = cv2.warpAffine(img, M, (w, h), borderValue=(0, 0, 0))

    # Convert to PIL format (for easy adjustment of hue/saturation)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # 3. Saturation fluctuation (randomly adjusted by 0.8 to 1.2 times)
    saturation_factor = random.uniform(0.8, 1.2)
    img_pil = ImageEnhance.Color(img_pil).enhance(saturation_factor)

    # 4. Tone fluctuation (fine adjustment of hue)
    img_hsv = img_pil.convert("HSV")
    h, s, v = img_hsv.split()
    h = h.point(lambda x: (x + random.randint(-10, 10)) % 256)  # 色相±10波动
    img_pil = Image.merge("HSV", (h, s, v)).convert("RGB")

    # Return to OpenCV format
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img


# ===================== 4. Batch processing of images =====================
def process_rock_slices():

    for img_name in os.listdir(INPUT_DIR):
        
        if img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            img_path = os.path.join(INPUT_DIR, img_name)
            print(f"正在处理：{img_name}")

            
            img_original = cv2.imread(img_path)
            if img_original is None:
                print(f"跳过损坏文件：{img_name}")
                continue

            
            img_resized = cv2.resize(img_original, TARGET_SIZE, interpolation=cv2.INTER_AREA)

            
            base_name = os.path.splitext(img_name)[0]
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_resized.jpg"), img_resized)

            
            for i in range(AUGMENT_NUM):
                img_aug = augment_image(img_resized.copy())
                save_path = os.path.join(OUTPUT_DIR, f"{base_name}_aug_{i + 1}.jpg")
                cv2.imwrite(save_path, img_aug)
                print(f"  已生成增强样本：{i + 1}/{AUGMENT_NUM}")

    print("\n✅ 所有图片处理完成！缩放+增强数据已保存至：", OUTPUT_DIR)


# ===================== 5. 运行主程序 =====================
if __name__ == "__main__":
    process_rock_slices()
