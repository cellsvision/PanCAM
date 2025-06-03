# PanCAM

## Paper Introduction
Accurate detection of lymph node metastasis is crucial for precise tumour staging and treatment decisions. However, conventional pathological methods are prone to miss lymph node micrometastasis, resulting in potential underdiagnosis and suboptimal clinical outcomes. We developed a Pan-Cancer Artificial intelligence diagnostic Model (PanCAM), a general and highly sensitive tool for detecting lymph node metastasis across diverse cancers.

14,110 Whole Slide Images (WSIs) from one centre (Sun Yat-sen Memorial Hospital) were used for PanCAM training and internal validation, 45,251 WSIs from 15 centres were used for retrospective external validation, and 10,141 WSIs from 9 centres were used for prospective validation. Based on DeepLabV3+ & RegNet_Y40 architecture, PanCAM was trained in pixel-level labelled WSIs of 9 common cancers, and validated in 33 types of cancer in the real-world settings. With overall sensitivity >0.98 and specificity >0.95 in external datasets, PanCAM provides malignant probability heatmaps and WSI-level classification, being able to assist pathologists in clinical practice in an interpretable way.

## Data Structure
### Dataset Organization
```
dataset/
├── img/
│   └──XXXX.tiff/                    # WSI name based DIR
│        └── valid_area_x.png        # valid area image
├── mask/                           
│   └──XXXX.tiff/
│        └── valid_area_x.png        # Ground truth masks
├── details_info/                    # WSI details information
│   ├── fg/                          # Foreground region details
│   └── no_fg/                       # Background region details
```

### Data Format
- Images: PNG format, RGB channels
- Masks: PNG format, single channel, PIL P mode
  - 0: Background
  - 1: Cancer region
  - 2: Ignore region (optional)

## Training
### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt
```

### Training Configuration
Key parameters in `run_train_spec.sh`:
- `--num_classes`: Number of classes (default: 2)
- `--crop_size`: Training patch size (default: 2048)
- `--batch_size`: Batch size (default: 8)
- `--epochs`: Training epochs (default: 100)
- `--lr`: Learning rate (default: 0.0001)
- `--optimizer`: Optimizer type (default: RAdam)
- `--scheduler`: Learning rate scheduler (default: CosineAnnealingWarmupRestarts)

### Data Sampling Strategy
The training process uses a balanced sampling strategy with four types of data:
1. Unlabeled negative data (neg_ratio)
2. Labeled pure background (pure_bg_ratio)
3. random labeled data (mix_ratio)
4. Labeled foreground samples (pos_ratio)

### Start Training
```bash
# Single GPU training
bash run_train_spec.sh
```
```bash
# Multi-GPU training, change os.environ["CUDA_VISIBLE_DEVICES"] = "0" in train_cv_vx.py
bash run_train_spec.sh
```

## Testing
### Model Evaluation
1. Prepare test data:
   - Place test images in `path_to_data/images`
   - Place test masks in `path_to_data/masks`

2. change the content in val_roi_box_v2.py
```python
if __name__ == '__main__':    
    input_model_path = 'path_to_model_dir'
    input_model_config_path = 'path_to_model_dir/config.yml'
    input_target_wsi_csv = 'path_to_target_wsi.csv'
    model_name_list = ['xx1.pth', 'xx2.pth', 'xx3.pth']
    input_img_path = 'path_to_input_images'
    input_label_path = 'path_to_input_masks'
    output_mask_path = 'path_to_output_masks'
    csv_output = 'path_to_output_csv'
```

3. Run inference:
```bash
python val_roi_box_v2.py 
```

## Citation
https://github.com/qubvel-org/segmentation_models.pytorch
https://github.com/sail-sg/Adan
https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup

