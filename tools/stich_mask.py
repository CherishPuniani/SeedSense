import sys
import os
import cv2
import numpy as np
import pandas as pd
import click


def stitch_images(mask_image_path, map_csv, output_path,scale_factor=1):
    """Stitch mask images based on geographic coordinates using projection mapping."""

    image_ext = '.png'
    image_size = (1024, 1024) 
    
    # Read CSV
    df = pd.read_csv(map_csv)
    coords = list(zip(df['latitude'], df['longitude']))
    image_names = df['image_name'].tolist()
    
    # Reference projection to (x, y) in meters
    ref_lat, ref_lon = coords[0]
    
    def latlon_to_xy(lat, lon, ref_lat, ref_lon):
        R = 6371000  # Earth radius in meters
        x = R * np.radians(lon - ref_lon) * np.cos(np.radians(ref_lat))
        y = R * np.radians(lat - ref_lat)
        return x, y
    
    positions_m = [latlon_to_xy(lat, lon, ref_lat, ref_lon) for lat, lon in coords]
    
    s = scale_factor  

    xs, ys = zip(*positions_m)
    min_x, min_y = min(xs), min(ys)
    shifted = [(x - min_x, y - min_y) for x, y in positions_m]
    
    # Meters → pixels
    grid_px = [(dx / s, dy / s) for dx, dy in shifted]
    
    gx, gy = zip(*grid_px)
    canvas_w = int(np.ceil((max(gx) + 1) * image_size[0]))
    canvas_h = int(np.ceil((max(gy) + 1) * image_size[1]))
    
  
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    
    
    for name, (px, py) in zip(image_names, grid_px):
        try:
            img_path = os.path.join(mask_image_path, f"{name}_mask{image_ext}")
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            
            if img is not None:
                
                if img.shape[:2] != (image_size[1], image_size[0]):
                    img = cv2.resize(img, image_size)

                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 4:  
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
  
                x_int = int(round(px * image_size[0]))
                y_int = int(round(py * image_size[1]))

                if (x_int >= 0 and y_int >= 0 and 
                    x_int + image_size[0] <= canvas_w and 
                    y_int + image_size[1] <= canvas_h):
                    
                    canvas[y_int:y_int + image_size[1], 
                           x_int:x_int + image_size[0]] = img
                
        except Exception as e:
            print(f"Warning: failed to load {name}: {e}")

    full_output_path = os.path.join(os.path.dirname(os.path.dirname(mask_image_path)), output_path)
    cv2.imwrite(full_output_path, canvas)
    print(f"✅ Saved stitched map to {full_output_path}")

@click.command()
@click.argument("mask_image_path", type=click.Path(exists=True, file_okay=False)) 
@click.argument("map_csv", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_path", type=click.Path(), default="stitched_output.png")
def cli(mask_image_path, map_csv, output_path):
    """Stitch images in MASK_IMAGE_PATH using lat/lon from MAP_CSV and save to OUTPUT_PATH."""
    stitch_images(mask_image_path, map_csv, output_path)

if __name__ == "__main__":
    cli()
