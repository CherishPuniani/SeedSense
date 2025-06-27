# SeedSense

- This repository uses SFA-Net based image-segmentation architecture to determine areas in which we can sow seeds autonomously.

## Install

```
conda create -n ssenv python=3.11
conda activate ssenc
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
pip install -r requirements.txt
```

## Folder Structure

Prepare the following folders to organize this repo:
```none
SFA-Net
├── network
├── config
├── data_pred
├── tools
├── model_weights (save the model weights)
├── fig_results (save the masks predicted)
├── lightning_logs (CSV format training logs)
├── data
│   ├── LoveDA
│   │   ├── Train
│   │   │   ├── Urban
│   │   │   │   ├── images_png (original images)
│   │   │   │   ├── masks_png (original masks)
│   │   │   │   ├── masks_png_convert (converted masks used for training)
│   │   │   │   ├── masks_png_convert_rgb (original rgb format masks)
│   │   │   ├── Rural
│   │   │   │   ├── images_png 
│   │   │   │   ├── masks_png 
│   │   │   │   ├── masks_png_convert
│   │   │   │   ├── masks_png_convert_rgb
│   │   ├── Val (the same with Train)
│   │   ├── Test
│   │   ├── train_val (Merge Train and Val)
```

<hr>
<span style="font-size:1.2em"><b>Note</b>: To access pre-trained models, you can download them from <a href="https://drive.google.com/drive/folders/1rkPqYzauU6ohBEWycomYcYxvHXnV9e1g?usp=drive_link">LoveDA weights</a>. Place the downloaded files in the <code>model_weights</code> directory.</span>
<hr>

## Data Preprocessing

Download Dataset: [LoveDA](https://codalab.lisn.upsaclay.fr/competitions/421)

Configure the folder as shown in 'Folder Structure' above.

```
python tools/loveda_mask_convert.py --mask-dir data/LoveDA/Train/Rural/masks_png --output-mask-dir data/LoveDA/Train/Rural/masks_png_convert
```
```
python tools/loveda_mask_convert.py --mask-dir data/LoveDA/Train/Urban/masks_png --output-mask-dir data/LoveDA/Train/Urban/masks_png_convert
```
```
python tools/loveda_mask_convert.py --mask-dir data/LoveDA/Val/Rural/masks_png --output-mask-dir data/LoveDA/Val/Rural/masks_png_convert
```
```
python tools/loveda_mask_convert.py --mask-dir data/LoveDA/Val/Urban/masks_png --output-mask-dir data/LoveDA/Val/Urban/masks_png_convert
```

This alters the LoveDa dataset images in training set to improve the water detection capability of the model
```
python tools/enhance_loveda.py
```

## Training

"-c" means the path of the config

```
python train.py -c config/uavid/sfanet.py
```

## Testing

"-c" denotes the path of the config, Use different **config** to test different models. 

"-o" denotes the output path 

"-t" denotes the test time augmentation (TTA), can be [None, 'lr', 'd4'], default is None, 'lr' is flip TTA, 'd4' is multiscale TTA

"--rgb" denotes whether to output masks in RGB format


**LoveDA** ([Online Testing](https://codalab.lisn.upsaclay.fr/competitions/421))

<img src="fig_ex/loveda.png" width="50%"/>

- To get RGB files:
```
python test_loveda.py -c config/loveda/sfanet.py -o fig_results/loveda/sfanet_loveda --rgb -t "d4"
```

- For submitting to the online test site:
```
python test_loveda.py -c config/loveda/sfanet.py -o fig_results/loveda/sfanet_loveda -t "d4"
```

- For generating the prediction for one image:
```
python prediction.py -c config/loveda/sfanet.py -i /path/to/image -o /path/to/output -t "d4" 
```

## Using the Streamlit App

The Streamlit app serves as an interactive interface to run the plantable area detection pipeline. Follow these steps to use the app:

1. **Launch the App**  
   Open a terminal in the repository's root directory and run:
   ```sh
   streamlit run app.py
   ```
   Streamlit will start and open your default browser to display the app interface.

2. **Use the Interface**  
   - **Input Files**: Use the sidebar and file uploader components to select image files (PNG, JPG, JPEG) and a metadata CSV file.  
   - **Configuration**: Adjust parameters like hex grid spacing, device (cuda/cpu), and model type directly in the sidebar.
   - **Run Pipeline**: Once your files are uploaded and validated, click the "🚀 Run Pipeline" button. The pipeline will process the images, update the metadata, and output results.
   - **View Results**: After processing, the app will display outputs such as the stitched mask with a hex grid overlay and the calculated green area percentage.

## Acknowledgement

- [SFA-Net](https://github.com/j2jeong/SFA-Net)
- [pytorch lightning](https://www.pytorchlightning.ai/)
- [timm](https://github.com/rwightman/pytorch-image-models)
- [pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt)
- [ttach](https://github.com/qubvel/ttach)
- [catalyst](https://github.com/catalyst-team/catalyst)
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
- [GeoSeg](https://github.com/WangLibo1995/GeoSeg)

