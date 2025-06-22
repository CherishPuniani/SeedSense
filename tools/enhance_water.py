import os
import cv2
import numpy as np
import random
import shutil
from tqdm import tqdm


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "LoveDA")
DATA_DIR = os.path.abspath(DATA_DIR)
def enhance_water():
    folders = ['Urban', 'Rural']
    split = 'Train'  

    for folder in folders:
        image_dir = os.path.join(DATA_DIR, split, folder, 'images_png')
        mask_dir = os.path.join(DATA_DIR, split, folder, 'masks_png_convert_rgb')
        output_dir = os.path.join(DATA_DIR, split, folder, 'images_png')
        os.makedirs(output_dir, exist_ok=True)

        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        num_to_augment = max(1, int(len(image_files) * 0.20))
        augment_files = set(random.sample(image_files, num_to_augment))

        for file in tqdm(image_files, desc=f"Processing {split}/{folder}"):
            image_path = os.path.join(image_dir, file)
            mask_path = os.path.join(mask_dir, file)
            output_path = os.path.join(output_dir, file)

            image = cv2.imread(image_path)
            if image is None:
                print(f"Skipping {file} due to image load error.")
                continue

            if file in augment_files:
                mask = cv2.imread(mask_path)
                if mask is None:
                    print(f"Skipping {file} due to mask load error.")
                    continue

                # Water body mask (blue mask area)
                blue_mask = (mask[:, :, 0] > 200) & (mask[:, :, 1] < 50) & (mask[:, :, 2] < 50)

                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv)

                # Enhance blue intensity
                h[blue_mask] = np.clip(h[blue_mask] + 20, 100, 140)
                s[blue_mask] = np.clip(s[blue_mask] + 60, 0, 255)
                v[blue_mask] = np.clip(v[blue_mask] + 20, 0, 255)

                hsv_enhanced = cv2.merge([h, s, v])
                enhanced_image = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
                cv2.imwrite(output_path, enhanced_image)
            else:
                shutil.copy(image_path, output_path)
