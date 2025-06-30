import streamlit as st
import os
import tempfile
import shutil
import pandas as pd
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import subprocess
import sys
from click.testing import CliRunner
from pipeline import run_pipeline
from tools.cfg import py2cfg

st.set_page_config(
    page_title="SeedSense Pipeline",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

def calculate_total_green_area(image_path):
    """Calculate total green area percentage from the final hex grid image, excluding black pixels"""
    image = cv2.imread(image_path)
    if image is None:
        return 0.0

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Mask for green pixels
    green_mask = np.all(image_rgb == [0, 255, 0], axis=-1).astype(np.uint8)
    # Mask for black pixels
    white_mask = np.all(image_rgb == [255, 255, 255], axis=-1).astype(np.uint8)
    # Exclude black pixels from total
    total_pixels = image_rgb.shape[0] * image_rgb.shape[1] - np.sum(white_mask)
    green_pixels = np.sum(green_mask)
    if total_pixels == 0:
        return 0.0
    green_percentage = (green_pixels / total_pixels) * 100
    return round(green_percentage, 2)


def validate_csv_format(df):
    """Validate that CSV has required columns"""
    required_columns = ['image_name', 'latitude', 'longitude', 'plantable']
    missing_columns = [col for col in required_columns if col not in df.columns]
    return len(missing_columns) == 0, missing_columns

def main():
    st.title("🌱Plantable Area Detection Pipeline")
    st.markdown("Upload images and metadata to analyze plantable areas and generate hex grid seed points")

    st.sidebar.header("Configuration")
    spacing = st.sidebar.number_input("Hex Grid Spacing", value=20, step=5)
    device = st.sidebar.selectbox("Device", ["cuda", "cpu"], index=0)
    model = st.sidebar.selectbox("Model",["No_augmentation", "10% Water_augmentation","15% Water_augmentation"])
    

    col1, col2 = st.columns(2)

    with col1:
        st.header("📁 Input Files")
        
        # Image upload
        st.subheader("Upload Images")
        uploaded_images = st.file_uploader(
            "Choose image files",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Upload multiple images for processing"
        )
        
        # CSV upload
        st.subheader("Upload Metadata CSV")
        uploaded_csv = st.file_uploader(
            "Choose CSV file",
            type=['csv'],
            help="CSV file with columns: image_name, latitude, longitude"
        )
        
        if uploaded_csv is not None:
            try:
                df = pd.read_csv(uploaded_csv)
                st.write("CSV Preview:")
                st.dataframe(df.head())
                
                is_valid, missing_cols = validate_csv_format(df)
                if not is_valid:
                    st.error(f"CSV missing required columns: {missing_cols}")
                else:
                    st.success("✅ CSV format is valid")
            except Exception as e:
                st.error(f"Error reading CSV: {str(e)}")
    
    with col2:
        st.header("🖼️ Uploaded Images")
        if uploaded_images:
            st.write(f"Total images uploaded: {len(uploaded_images)}")
            
            cols = st.columns(3)
            for idx, uploaded_file in enumerate(uploaded_images[:6]):  # Show first 6 images
                with cols[idx % 3]:
                    image = Image.open(uploaded_file)
                    st.image(image, caption=uploaded_file.name, use_container_width=True)
            
            if len(uploaded_images) > 6:
                st.write(f"... and {len(uploaded_images) - 6} more images")

    st.markdown("---")
    
    if st.button("🚀 Run Pipeline", type="primary", use_container_width=True):
        if not uploaded_images:
            st.error("Please upload at least one image")
            return
        
        if uploaded_csv is None:
            st.error("Please upload a CSV file")
            return
        
        # To validate CSV format
        try:
            uploaded_csv.seek(0)
            df = pd.read_csv(uploaded_csv)
            is_valid, missing_cols = validate_csv_format(df)
            if not is_valid:
                st.error(f"CSV missing required columns: {missing_cols}")
                return
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
            return
        
        input_dir = "data_pred/input_images_temp"
        output_dir = "data_pred/pred_masks_temp"
        csv_path = "data_pred/img_meta_temp.csv"
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        stitched_output = "data_pred/stitched_mask.png"
        hex_output = "data_pred/hex_grid_output.png"
        if model == "No_augmentation":
            config_path = "config/loveda/sfanet_pred_naug.py"
        elif model == "10% Water_augmentation":
            config_path = "config/loveda/sfanet_pred_aug10.py"
        elif model == "15% Water_augmentation":
            config_path = "config/loveda/sfanet_pred_aug15.py"

        uploaded_csv.seek(0)  # Reseting the file pointer
        with open(csv_path, "wb") as f:
            f.write(uploaded_csv.read())

        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, uploaded_file in enumerate(uploaded_images):
            image_path = os.path.join(input_dir, uploaded_file.name)
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            progress_bar.progress((idx + 1) / len(uploaded_images) * 0.3)

            status_text.text("Running pipeline...")
            progress_bar.progress(0.3)
            

        run_pipeline(
            config_path=config_path,
            image_dir=input_dir,
            output_dir=output_dir,
            map_csv=csv_path,
            stitched_output=stitched_output,
            hex_output=hex_output,
            spacing=spacing,
            device=device,
            show=False
        )
        
        pipeline_success = os.path.isfile(stitched_output) and os.path.isfile(hex_output)

        
        if pipeline_success:
            status_text.text("✅ Pipeline completed successfully!")

            st.markdown("---")
            st.header("📊 Results")
            final_image = Image.open(hex_output)
            st.image(final_image, caption="Hex Grid with Seed Points",width=1000)
            
            green_area = calculate_total_green_area(str(hex_output))
            st.text(f"Total Green Area: {green_area}%")


        else:
            status_text.text("❌ Pipeline failed or incomplete")
            st.error("Pipeline execution failed or did not produce expected outputs")
                    

if __name__ == "__main__":
    main()
