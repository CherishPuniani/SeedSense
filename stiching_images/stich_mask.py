# Read CSV for image to longitude and latitude mapping
# Make a grid for the images
# Stich the images together based on the grid
import os
import cv2
import numpy as np
import pandas as pd
import click

def stitch_images(mask_image_path, map_csv, output_path):
    df = pd.read_csv(map_csv)
    df = df.sort_values(by=['latitude', 'longitude'])
    df = df.reset_index(drop=True)

    grouped = df.groupby('latitude')
    image_matrix = []

    max_row_width = 0  
    row_heights = []   

    for _, group in grouped:
        row_images = []
        max_height = 0
        max_width = 0

        imgs = []
        for _, row in group.iterrows():
            image_path = os.path.join(mask_image_path, row['image_name'])
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                imgs.append(img)
                max_height = max(max_height, img.shape[0])
                max_width += img.shape[1]
                # print(max_width)

        padded_row = []
        for img in imgs:
            h, w = img.shape[:2]
            # Pad image to match max height of the row
            if h < max_height:
                pad_height = max_height - h
                pad = np.zeros((pad_height, w, img.shape[2]) if img.ndim == 3 else (pad_height, w), dtype=img.dtype)
                img = np.vstack((img, pad))
            padded_row.append(img)

        if padded_row:
            row_img = np.hstack(padded_row)
            image_matrix.append(row_img)
            row_heights.append(max_height)
            max_row_width = max(max_row_width, row_img.shape[1])
    
    # Pad rows to match max_row_width
    for i in range(len(image_matrix)):
        row_img = image_matrix[i]
        h, w = row_img.shape[:2]
        if w < max_row_width:
            pad_width = max_row_width - w
            if row_img.ndim == 3:
                pad = np.zeros((h, pad_width, row_img.shape[2]), dtype=row_img.dtype)
            else:
                pad = np.zeros((h, pad_width), dtype=row_img.dtype)
            image_matrix[i] = np.hstack((row_img, pad))

    # Stack all rows vertically
    stitched_image = np.vstack(image_matrix)

    # Save result
    cv2.imwrite(output_path, stitched_image)
    # print(f"Stitched image saved to {output_path}")

@click.command()
@click.argument("mask_image_path", type=click.Path(exists=True, file_okay=False)) 
@click.argument("map_csv", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_path", type=click.Path())
def cli(mask_image_path, map_csv, output_path):
    """Stitch images in MASK_IMAGE_PATH using lat/lon from MAP_CSV and save to OUTPUT_PATH."""
    stitch_images(mask_image_path, map_csv, output_path)

if __name__ == "__main__":
    cli()
