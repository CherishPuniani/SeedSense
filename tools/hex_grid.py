import cv2
import numpy as np
import click
import pandas as pd
import math

def hex_packed_seed_points(image_path, map_csv, spacing=20, output_path="seed_output.png", scale_factor=1):

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    green_mask = np.all(image_rgb == [0, 255, 0], axis=-1).astype(np.uint8)
    h, w = green_mask.shape
    seed_points_pixels = []  
    
    row_height = int(spacing * np.sqrt(3) / 2)
    
    for row in range(0, h, row_height):
        offset = spacing // 2 if (row // row_height) % 2 == 1 else 0
        for col in range(offset, w, spacing):
            if green_mask[row, col] == 1:
                seed_points_pixels.append((col, row))

    # Reading CSV to get lat/lon info used during stitching
    df = pd.read_csv(map_csv)
    coords = list(zip(df['latitude'], df['longitude']))
    ref_lat, ref_lon = coords[0]

    def latlon_to_xy(lat, lon, ref_lat, ref_lon):
        R = 6371000  # Earth radius in meters
        x = R * np.radians(lon - ref_lon) * np.cos(np.radians(ref_lat))
        y = R * np.radians(lat - ref_lat)
        return x, y

    positions_m = [latlon_to_xy(lat, lon, ref_lat, ref_lon) for lat, lon in coords]
    xs_m, ys_m = zip(*positions_m)
    min_x, min_y = min(xs_m), min(ys_m)
    
    image_size = (1024,1024) 
    s = scale_factor
    
    R = 6371000 
    
    def pixel_to_latlon(x_int, y_int):
        grid_x = x_int / image_size[0]
        grid_y = y_int / image_size[1]
        x_m = grid_x * s + min_x
        y_m = grid_y * s + min_y

        lat = ref_lat + (y_m / R) * (180 / np.pi)
        lon = ref_lon + (x_m / (R * np.cos(np.radians(ref_lat)))) * (180 / np.pi)
        lat = float(f"{lat:.6f}")
        lon = float(f"{lon:.6f}")
        return lat, lon
    
    seed_points_geo = [pixel_to_latlon(x, y) for (x, y) in seed_points_pixels]

    seed_points_txt_path = output_path.rsplit(".", 1)[0] + "_seed_points.txt"
    with open(seed_points_txt_path, "w") as f:
        for lat, lon in seed_points_geo:
            f.write(f"{lat},{lon}\n")
    print(f"Seed points saved to: {seed_points_txt_path}")
    print(f"Seed points placed: {len(seed_points_geo)}")

    output_img = image_rgb.copy()
    for (x, y) in seed_points_pixels:
        cv2.circle(output_img, (x, y), radius=5, color=(0, 0, 0), thickness=-1)
    
    output_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, output_bgr)
    print(f"Saved output with seed points to: {output_path}")

@click.command()
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("map_csv", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_path", type=click.Path())
@click.option("--spacing", default=20, help="Spacing between seed points in pixels.")
def cli(image_path, map_csv, output_path, spacing):
    """
    Place hexagonally packed seed points in the stitched image (IMAGE_PATH) using green pixels and
    save the corresponding geographic coordinates (lat, lon) from MAP_CSV to a text file.
    """
    hex_packed_seed_points(image_path, map_csv, spacing, output_path)

if __name__ == "__main__":
    cli()
