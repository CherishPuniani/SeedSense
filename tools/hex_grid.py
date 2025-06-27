import cv2
import numpy as np
import click

def hex_packed_seed_points(image_path, spacing=20, output_path="seed_output.png"):
    """
    Place hexagonally packed seed points in white regions of the image.
    Args:
        image_path (str): Path to the input image.
        spacing (int): Distance between seed points.
        output_path (str): Output image path with seed points drawn.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    green_mask = np.all(image_rgb == [0, 255, 0], axis=-1).astype(np.uint8)

    h, w = green_mask.shape
    seed_points = []

    row_height = int(spacing * np.sqrt(3) / 2)

    for row in range(0, h, row_height):
        offset = spacing // 2 if (row // row_height) % 2 == 1 else 0
        for col in range(offset, w, spacing):
            if green_mask[row, col] == 1:
                seed_points.append((col, row))
                # Save seed points to a file
                # seed_points_path = output_path.rsplit(".", 1)[0] + "_seed_points.txt"
                # with open(seed_points_path, "w") as f:
                #     for x, y in seed_points:
                #         f.write(f"{x},{y}\n")
                # print(f"Seed points saved to: {seed_points_path}")
    print(f"Seed points placed: {len(seed_points)}")

    # Draw points
    output_img = image_rgb.copy()
    for (x, y) in seed_points:
        cv2.circle(output_img, (x, y), radius=2, color=(0, 0, 0), thickness=-1)

    # Save
    output_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, output_bgr)
    print(f"Saved output with seed points to: {output_path}")

@click.command()
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_path", type=click.Path())
@click.option("--spacing", default=20, help="Spacing between seed points in pixels.")

def cli(image_path, output_path, spacing):
    """
    Places seed points in white areas of IMAGE_PATH and saves to OUTPUT_PATH.
    """
    hex_packed_seed_points(image_path, spacing, output_path)

if __name__ == "__main__":
    cli()
