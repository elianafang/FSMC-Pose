# FSMC-Pose: Frequency and Spatial Fusion with Multiscale Self-calibration for Cattle Mounting Pose Estimation

FSMC-Pose is a real-time framework for accurate dairy cattle mounting pose estimation. This configuration is based on MMPose. Please install MMPose first following the instructions below.

<p align="center">
  <a href="PAPER_URL">
    <img src="https://img.shields.io/badge/FSMC--Pose-Paper-E76F51?style=flat-square&logo=arxiv&logoColor=white" />
  </a>
  &nbsp;&nbsp;
  <a href="DATASET_URL">
    <img src="https://img.shields.io/badge/MOUNT--Cattle-Dataset-c5a730?style=flat-square&logo=huggingface&logoColor=white" />
  </a>
    &nbsp;&nbsp;
  <a href="CODE_URL">
    <img src="https://img.shields.io/badge/FSMC--Pose-Code-4C9BDB?style=flat-square&logo=github&logoColor=white" />
  </a>
  </p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/f2e31c7a-c4bf-4cbf-9098-8738540362bd" width="52%" />
  <img src="https://github.com/user-attachments/assets/cddac540-0f7e-4244-b263-78b3702fcf70" width="47%" />
</p>


## Updates/News 📣
* 🏆 **News (Feb. 2026)**: We are excited to announce that FSMC-Pose has been accepted for CVPR 2026 Findings! 🎉
* 🔗 **News**: Please find the open-source dataset on Hugging Face: [MOUNT-Cattle](URL).

## Dataset
We created the MOUNT-Cattle dataset with 1,176 mounting instances in COCO format, supporting drop-in training across pose estimation models. MOUNT-Cattle is designed to advance cattle pose estimation and is available for download on Hugging Face [here](URL). In our experiments, we used a comprehensive dataset that combines MOUNT-Cattle with the public NWAFU-Cattle dataset. 

## Installation

### 1. Environment Requirements

- CUDA 11.2 (or compatible version)
- 64GB RAM
- GPU: P100 or compatible

**Note:** All installation steps follow the [official MMPose documentation](https://mmpose.readthedocs.io/en/latest/).

### 2. Create Conda Virtual Environment

```bash
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

### 3. Install PyTorch

```bash
pip install torch==1.10.1+cu102 torchvision==0.11.2+cu102 torchaudio==0.10.1 -f 
```

Verify the installation:

```bash
python -c "import torch; import torchvision; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Torchvision version: {torchvision.__version__}')"
```

Expected output:

```
PyTorch version: 1.10.1+cu102
CUDA available: True
Torchvision version: 0.11.2+cu102
```

### 4. Install MMEngine, MMCV, and MMDetection using MIM

**Important:** Please refer to the official MMPose documentation for version compatibility. Different versions of MMPose require different versions of mmcv, mmdet, etc. Version mismatches may cause runtime errors.

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.1"
mim install "mmdet>=3.1.0"
```

If pip is too slow, use the Tsinghua mirror:

```bash
pip install -U openmim -i 
```

### 5. Install MMPose from Source

```bash
cd mmpose
pip install -r requirements.txt
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

### 6. Verify Installation

**Step 1:** Download config and checkpoint files.

```bash
mim download mmpose --config td-hm_hrnet-w48_8xb32-210e_coco-256x192 --dest .
```

**Step 2:** Run inference demo.

```bash
python demo/image_demo.py tests/data/coco/000000000785.jpg td-hm_hrnet-w48_8xb32-210e_coco-256x192.py td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth --out-file vis_results.jpg --draw-heatmap
```

Expected output:

```
Loads checkpoint by local backend from path: td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth
07/24 07:46:43 - mmengine - INFO - the output image has been saved at vis_results.jpg
```

## Training

### Multi-GPU Training

```bash
CUDA_VISIBLE_DEVICES=0,1 PORT=29500 bash ./tools/dist_train.sh /data/configs/FSMCPose.py 2
```

**Parameters:**

- `CUDA_VISIBLE_DEVICES`: Specify GPU IDs (e.g., 0,1 for 2 GPUs)
- `PORT`: Port number for distributed training (default: 29500)
- `config_file`: Path to configuration file
- `num_gpus`: Number of GPUs to use (2 in this example)

**Example:**

```bash
CUDA_VISIBLE_DEVICES=8,9 PORT=29527 bash ./tools/dist_train.sh /data/configs/FSMCPose.py 2
```

### Single-GPU Training

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py /data/configs/FSMCPose.py
```

## Testing

### Test with Best Checkpoint

```bash
python tools/test.py /data/configs/FSMCPose.py /data/work_dirs/FSMCPose/best_coco_AP_epoch_XXX.pth
```

Replace `XXX` with the actual epoch number of the best checkpoint.

**Example:**

```bash
python tools/test.py /data/configs/FSMCPose.py /data/work_dirs/FSMCPose/best_coco_AP_epoch_600.pth
```

### Test with Latest Checkpoint

```bash
python tools/test.py /data/configs/FSMCPose.py /data/work_dirs/FSMCPose/latest.pth
```

## Checkpoint Location

**Work Directory:** `/data/work_dirs/FSMCPose/`

**Checkpoint Files:**

- **Best Model:** `best_coco_AP_epoch_XXX.pth` (saved based on COCO AP metric)
- **Latest Model:** `latest.pth` (latest checkpoint)
- **Epoch Checkpoints:** `epoch_XXX.pth` (checkpoints at specific epochs)

**Note:** The checkpoint hook saves the best model based on `coco/AP` metric with `greater` rule (higher is better). Maximum 10 checkpoints are kept.

## Contact
[elianafang](lifangjing@bjtu.edu.cn)
