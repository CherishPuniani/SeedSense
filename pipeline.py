import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from torchvision import transforms
import click
import gdown  # Add this import for Google Drive downloads
from train import Supervision_Train
from tools.cfg import py2cfg
from tools.stich_mask import stitch_images
from tools.hex_grid import hex_packed_seed_points

def label2rgb(mask):
    """Convert label mask to RGB visualization with Building, Road, Water, Forest as one color."""
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    # Set background white
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    # Set Building, Road, Water, Forest (classes 1,2,3,5) to red
    for cls in [1, 2, 3, 5]:
        mask_rgb[np.all(mask_convert == cls, axis=0)] = [255, 0, 0]
    # Set classes 4 and 6 to green
    for cls in [4, 6]:
        mask_rgb[np.all(mask_convert == cls, axis=0)] = [0, 255, 0]
    return mask_rgb


def download_weights_from_gdrive(gdrive_file_id, local_weights_path="model_weights"):
    """Download model weights from Google Drive if they don't exist locally."""
    if os.path.exists(local_weights_path):
        click.echo(f"Weights already exist at: {local_weights_path}")
        return local_weights_path
    
    click.echo(f"Downloading weights from Google Drive to: {local_weights_path}")
    os.makedirs(os.path.dirname(local_weights_path), exist_ok=True)
    
    gdrive_url = f"https://drive.google.com/uc?id={gdrive_file_id}"
    gdown.download(gdrive_url, local_weights_path, quiet=False)
    https://drive.google.com/drive/folders?usp=sharing
    if os.path.exists(local_weights_path):
        click.echo("Successfully downloaded weights from Google Drive")
        return local_weights_path
    else:
        raise FileNotFoundError("Failed to download weights from Google Drive")

def run_pipeline(config_path, image_dir, output_dir, map_csv, stitched_output, hex_output, spacing, device, show, gdrive_file_id="1-oz6q723IljvUGO0rTDsC3QjJYdbPdOk"):

    os.makedirs(output_dir, exist_ok=True)

    meta_path = map_csv
    meta_df = pd.read_csv(meta_path)

    click.echo("Loading configuration and model...")
    config = py2cfg(config_path)
    model_ckpt = os.path.join(config.weights_path, config.test_weights_name + '.ckpt')
    
    # Download weights from Google Drive if they don't exist locally
    if gdrive_file_id and not os.path.exists(model_ckpt):
        model_ckpt = download_weights_from_gdrive(gdrive_file_id, model_ckpt)
    
    click.echo(f"Loading model from: {model_ckpt}")
    model = Supervision_Train.load_from_checkpoint(model_ckpt, config=config)
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    click.echo("Running predictions on input images...")
    for image_name in os.listdir(image_dir):
        if not image_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(image_dir, image_name)
        image = cv2.imread(img_path)
        if image is None:
            click.echo(f"Skipping unreadable image: {img_path}")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image = image.copy()

        target_size = (1024, 1024)
        resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        image_tensor = transform(resized_image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(image_tensor)
            probs = nn.Softmax(dim=1)(logits)
            top2 = probs.topk(k=2, dim=1)
            p1, p2 = top2.values[:, 0], top2.values[:, 1]
            c1, c2 = top2.indices[:, 0], top2.indices[:, 1]
            delta = 1
            is_clutter = (c1 == 0)
            clutter_confident = is_clutter & ((p1 - p2) > delta)
            final_pred = c1.clone()
            swap_mask = is_clutter & ~clutter_confident
            final_pred[swap_mask] = c2[swap_mask]
            prediction = final_pred[0].cpu().numpy()

        prediction_rgb = label2rgb(prediction)
        # Resize prediction back to original image size
        # prediction_rgb = cv2.resize(prediction_rgb, (original_image.shape[1], original_image.shape[0]),
        #                             interpolation=cv2.INTER_NEAREST)

        out_path = os.path.join(output_dir, os.path.splitext(image_name)[0] + "_mask.png")
        cv2.imwrite(out_path, cv2.cvtColor(prediction_rgb, cv2.COLOR_RGB2BGR))
        click.echo(f"Saved predicted mask to: {out_path}")


        green_mask = np.all(prediction_rgb == [0, 255, 0], axis=-1).astype(np.uint8)
        green_area_percent = round(100 * np.sum(green_mask) / green_mask.size, 2)

        image_base = os.path.splitext(image_name)[0]
        if image_base in meta_df["image_name"].values:
            meta_df.loc[meta_df["image_name"] == image_base, "plantable"] = green_area_percent
        else:
            new_row = pd.DataFrame({"image_name": [image_base], "latitude": [None], "longitude": [None], "plantable": [green_area_percent]})
            meta_df = pd.concat([meta_df, new_row], ignore_index=True)

        if show:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.title("Original Image")
            plt.imshow(original_image)
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.title("Prediction")
            plt.imshow(prediction_rgb)
            plt.axis("off")
            plt.tight_layout()
            plt.show()

    meta_df.to_csv(meta_path, index=False)
    click.echo(f"Updated meta CSV saved to: {meta_path}")

    click.echo("Stitching predicted masks...")
    stitch_images(output_dir, meta_path, stitched_output)
    click.echo(f"Stitched image saved to: {stitched_output}")

    click.echo("Generating hex grid overlay...")
    hex_packed_seed_points(stitched_output,map_csv=meta_path, spacing=spacing, output_path=hex_output)
    click.echo(f"Hex grid image saved to: {hex_output}")


@click.command()
@click.option('--config_path', required=True, type=click.Path(exists=True), help="Path to config file")
@click.option('--image_dir', required=True, type=click.Path(exists=True), help="Directory containing input images")
@click.option('--output_dir', required=True, type=click.Path(), help="Directory to save predicted masks")
@click.option('--map_csv', required=True, type=click.Path(exists=True), help="CSV file with geo-meta data")
@click.option('--stitched_output', required=True, type=click.Path(), help="File path to save the stitched mask image")
@click.option('--hex_output', required=True, type=click.Path(), help="File path to save the hex grid overlay image")
@click.option('--spacing', default=20, type=int, help="Spacing between seed points for the hex grid")
@click.option('--device', default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda/cpu)")
@click.option('--show', is_flag=True, help="Display images during processing")
@click.option('--gdrive_file_id', default=None, help="Google Drive file ID for model weights")
def pipeline(config_path, image_dir, output_dir, map_csv, stitched_output, hex_output, spacing, device, show, gdrive_file_id):
    """Run the complete plantable area detection pipeline."""
    run_pipeline(config_path, image_dir, output_dir, map_csv, stitched_output, hex_output, spacing, device, show, gdrive_file_id)

if __name__ == '__main__':
    pipeline()