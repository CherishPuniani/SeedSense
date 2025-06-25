import os
import cv2
import torch
import argparse
import numpy as np
from torch import nn
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from train import *
import ttach as tta

# Import the model and training class
from tools.cfg import py2cfg
import types


def label2rgb(mask):
    """Convert label mask to RGB visualization"""
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]  # Background
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]      # Building
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [255, 255, 0]    # Road
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 0, 255]      # Water
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [159, 129, 183]  # Barren
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [0, 255, 0]      # Forest
    mask_rgb[np.all(mask_convert == 6, axis=0)] = [255, 195, 128]  # Agricultural
    return mask_rgb

# def label2rgb(mask):
#     """Convert label mask to RGB visualization with Building, Road, Water, Forest as one color"""
#     h, w = mask.shape[0], mask.shape[1]
#     mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
#     mask_convert = mask[np.newaxis, :, :]
#     total_pixels = h*w
#     # Set background
#     mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]  # Background
#     red_percent = 0
#     green_percent = 0
#     # Set Building, Road, Water, Forest to red
#     for cls in [1, 2, 3, 5]:
#         cls_count = np.sum(mask == cls)
#         red_percent += round(100 * cls_count / total_pixels,2)
        
#         mask_rgb[np.all(mask_convert == cls, axis=0)] = [255, 0, 0]
#     print("red=", red_percent)

#     # Keep other classes as before
#     for cls in [4,6]:
#         cls_count = np.sum(mask == cls)
#         green_percent += round(100 * cls_count / total_pixels,2)
        
#         mask_rgb[np.all(mask_convert == cls, axis=0)] = [0, 255, 0]
#     print("green=", green_percent)
    
#     return mask_rgb


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
    parser = argparse.ArgumentParser(description="Predict segmentation mask for a single image using SFANet")
    parser.add_argument("-c", "--config_path", type=str, required=True, help="Path to config file")
    parser.add_argument("-i", "--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("-o", "--output_path", type=str, default="output.png", help="Path to save output mask")
    parser.add_argument("--show", action="store_true", help="Display the input image and prediction")
    parser.add_argument("-t", "--tta", help="Test time augmentation.", default=None, choices=[None, "d4", "lr"]) ## lr is flip TTA, d4 is multi-scale TTA
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                       help="Device to use (cuda/cpu)")
    return parser.parse_args()

def calculate_percentage(mask):
  """
  Calculate the percentage of area covered by each class in the segmentation mask.

  Args:
      mask (numpy.ndarray): The predicted mask with class indices.

  Returns:
      dict: A dictionary with class indices as keys and their percentage coverage as values.
  """
  total_pixels = mask.size  # Total number of pixels in the mask
  unique_classes, counts = np.unique(mask, return_counts=True)
  percentages = {cls: (count / total_pixels) * 100 for cls, count in zip(unique_classes, counts)}
  return percentages


# def py2cfg(config_path):
#     """
#     Load a python config file as a module and return its attributes as an object.
#     """
#     import importlib.util

#     config_path = os.path.abspath(config_path)
#     spec = importlib.util.spec_from_file_location("config", config_path)
#     config_module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(config_module)

#     # Convert module attributes to a simple object (namespace)
#     class Config: pass
#     cfg = Config()
#     for key in dir(config_module):
#         if not key.startswith("__"):
#             setattr(cfg, key, getattr(config_module, key))
#     return cfg

def main():
    # Parse arguments
    args = get_args()
    config = py2cfg(args.config_path)
    device = torch.device(args.device)
    
    # Load the model
    print(f"Loading model from: {os.path.join(config.weights_path, config.test_weights_name+'.ckpt')}")
    model = Supervision_Train.load_from_checkpoint(
        os.path.join(config.weights_path, config.test_weights_name+'.ckpt'),
        config=config
    )
    model.to(device)
    model.eval()

    # # ---- ADD THIS BLOCK FOR TTA ----
    # if args.tta == "lr":
    #     transforms_tta = tta.Compose(
    #         [
    #             tta.HorizontalFlip(),
    #             tta.VerticalFlip()
    #         ]
    #     )
    #     model = tta.SegmentationTTAWrapper(model, transforms_tta)
    # elif args.tta == "d4":
    #     transforms_tta = tta.Compose(
    #         [
    #             tta.HorizontalFlip(),
    #             # tta.VerticalFlip(),
    #             # tta.Rotate90(angles=[0, 90, 180, 270]),
    #             tta.Scale(scales=[0.75, 1.0, 1.25, 1.5], interpolation='bicubic', align_corners=False),
    #             # tta.Multiply(factors=[0.8, 1, 1.2])
    #         ]
    #     )
    #     model = tta.SegmentationTTAWrapper(model, transforms_tta)
    # # ---- END TTA BLOCK ----

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Read image
    image = cv2.imread(args.image_path)
    if image is None:
        raise ValueError(f"Could not read image from {args.image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Store original image for display
    original_image = image.copy()
    
    # Resize to model's expected input size (adjust if needed)
    target_size = (1024, 1024)  
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
  
    #Converting to gray scale severly impaired the performance as expected
    # image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # image_gray_3ch = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)

    # # Apply preprocessing to the grayscale image
    # image_tensor = transform(image_gray_3ch).unsqueeze(0).to(device)

    # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    # lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    # lab[:,:,0] = clahe.apply(lab[:,:,0])
    # image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # alpha = 1.5  # Contrast control (1.0-3.0)
    # beta = 10    # Brightness control (0-100)
    # image_contrast = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # Apply preprocessing to the contrast-enhanced image
    image_tensor = transform(image).unsqueeze(0).to(device)


    # # Apply preprocessing
    # image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(image_tensor)                 # shape (1,7,H,W)
        probs  = nn.Softmax(dim=1)(logits)           # shape (1,7,H,W)

        # --- relative-margin thresholding ----
       
        top2 = probs.topk(k=2, dim=1)                
        p1, p2 = top2.values[:,0], top2.values[:,1] 
        c1, c2 = top2.indices[:,0], top2.indices[:,1]

        delta = 1  # margin threshold
        is_clutter = (c1 == 0)  
        clutter_confident = is_clutter & ((p1 - p2) > delta)


        final_pred = c1.clone()
        swap_mask = is_clutter & ~clutter_confident
        final_pred[swap_mask] = c2[swap_mask]

        
        prediction = final_pred[0].cpu().numpy()
    
    # Convert prediction to RGB visualization
    prediction_rgb = label2rgb(prediction)
    prediction_rgb = cv2.resize(prediction_rgb, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Save output
    cv2.imwrite(args.output_path, cv2.cvtColor(prediction_rgb, cv2.COLOR_RGB2BGR))
    print(f"Prediction saved to: {args.output_path}")
    
    # Show results if requested
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


if __name__ == "__main__":
    main()