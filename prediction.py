import os
import cv2
import torch
import argparse
import numpy as np
import pandas as pd
from torch import nn
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from train import *
import ttach as tta
import sys

# Import the model and training class
from tools.cfg import py2cfg
import types

# Uncomment the following function if you want prediction to visualize all classes in different colors
# def label2rgb(mask):
#     """Convert label mask to RGB visualization"""
#     h, w = mask.shape[0], mask.shape[1]
#     mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
#     mask_convert = mask[np.newaxis, :, :]
#     mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]  # Background
#     mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]      # Building
#     mask_rgb[np.all(mask_convert == 2, axis=0)] = [255, 255, 0]    # Road
#     mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 0, 255]      # Water
#     mask_rgb[np.all(mask_convert == 4, axis=0)] = [159, 129, 183]  # Barren
#     mask_rgb[np.all(mask_convert == 5, axis=0)] = [0, 255, 0]      # Forest
#     mask_rgb[np.all(mask_convert == 6, axis=0)] = [255, 195, 128]  # Agricultural
#     return mask_rgb

def label2rgb(mask):
    """Convert label mask to RGB visualization with Building, Road, Water, Forest as one color"""
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    total_pixels = h*w
    # Set background
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]  # Background
    red_percent = 0
    green_percent = 0
    # Set Building, Road, Water, Forest to red
    for cls in [1, 2, 3, 5]:
        cls_count = np.sum(mask == cls)
        red_percent += round(100 * cls_count / total_pixels,2)

        mask_rgb[np.all(mask_convert == cls, axis=0)] = [255, 0, 0]
    # print("red=", red_percent)

    # Keep other classes as before
    for cls in [4,6]:
        cls_count = np.sum(mask == cls)
        green_percent += round(100 * cls_count / total_pixels,2)

        mask_rgb[np.all(mask_convert == cls, axis=0)] = [0, 255, 0]
    # print("green=", green_percent)

    return mask_rgb

def img_writer(inp):
    (mask,  mask_id, rgb) = inp
    if rgb:
        mask_name_tif = mask_id + '.png'
        mask_tif = label2rgb(mask)
        mask_tif = cv2.cvtColor(mask_tif, cv2.COLOR_RGB2BGR)
        cv2.imwrite(mask_name_tif, mask_tif)
    else:
        mask_png = mask.astype(np.uint8)
        mask_name_png = mask_id + '.png'
        cv2.imwrite(mask_name_png, mask_png)

def get_args():
    parser = argparse.ArgumentParser(description="Predict segmentation masks for a folder of images using SFANet")
    parser.add_argument("-c", "--config_path", type=str, required=True, help="Path to config file")
    parser.add_argument("-i", "--image_dir", type=str, required=True, help="Path to input image folder")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Path to save output masks")
    parser.add_argument("--show", action="store_true", help="Display the input image and prediction")
    parser.add_argument("-t", "--tta", help="Test time augmentation.", default=None, choices=[None, "d4", "lr"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                       help="Device to use (cuda/cpu)")
    return parser.parse_args()

def calculate_percentage(mask):
    total_pixels = mask.size
    unique_classes, counts = np.unique(mask, return_counts=True)
    percentages = {cls: (count / total_pixels) * 100 for cls, count in zip(unique_classes, counts)}
    return percentages

def main():
    args = get_args()
    config = py2cfg(args.config_path)
    device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from: {os.path.join(config.weights_path, config.test_weights_name+'.ckpt')}")
    model = Supervision_Train.load_from_checkpoint(
        os.path.join(config.weights_path, config.test_weights_name+'.ckpt'),
        config=config
    )
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    meta_path = os.path.join(os.path.dirname(os.path.dirname(args.image_dir)), "img_meta.csv")
    meta_df = pd.read_csv(meta_path)

    for image_name in os.listdir(args.image_dir):
        if not image_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        image_path = os.path.join(args.image_dir, image_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image = image.copy()

        target_size = (1024, 1024)
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(image_tensor)
            probs  = nn.Softmax(dim=1)(logits)

            top2 = probs.topk(k=2, dim=1)
            p1, p2 = top2.values[:,0], top2.values[:,1] 
            c1, c2 = top2.indices[:,0], top2.indices[:,1]

            delta = 1
            is_clutter = (c1 == 0)
            clutter_confident = is_clutter & ((p1 - p2) > delta)
            final_pred = c1.clone()
            swap_mask = is_clutter & ~clutter_confident
            final_pred[swap_mask] = c2[swap_mask]

            prediction = final_pred[0].cpu().numpy()

        prediction_rgb = label2rgb(prediction)
        # prediction_rgb = cv2.resize(prediction_rgb, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)

        output_path = os.path.join(args.output_dir, os.path.splitext(image_name)[0] + "_mask.png")
        cv2.imwrite(output_path, cv2.cvtColor(prediction_rgb, cv2.COLOR_RGB2BGR))
        print(f"Saved: {output_path}")

        green_mask = np.all(prediction_rgb == [0, 255, 0], axis=-1).astype(np.uint8)
        green_area_percent = round(100 * np.sum(green_mask) / green_mask.size, 2)

        image_base = os.path.splitext(image_name)[0]
        if image_base in meta_df["image_name"].values:
            meta_df.loc[meta_df["image_name"] == image_base, "plantable"] = green_area_percent

        if args.show:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.title("Original Image")
            plt.imshow(original_image)
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.title("Prediction")
            plt.imshow(prediction_rgb)
            plt.axis('off')
            plt.tight_layout()
            plt.show()

    meta_df.to_csv(meta_path, index=False)
    print(f"Updated meta CSV saved to: {meta_path}")

if __name__ == "__main__":
    main()
