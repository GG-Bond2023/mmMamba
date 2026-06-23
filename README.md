<h1>mmMamba </h1>


**Important Note**

Due to the file size constraints for ACML-2026 supplementary materials, this zip archive only contains a minimal version of the source code. 
 
For the most up-to-date code, pre-trained models, network weights, complete evaluation results, and further detailed documentation, please visit our fully anonymized repository: 
**[https://anonymous.4open.science/r/mmMamba-3741/].**


## Getting Started
{proj_dir} refers to the path where the code is stored on the disk.
### Dataset preparation

Download and extract RADIal dataset from https://github.com/valeoai/RADIal

Move RADIal dataset to the  path: {proj_dir}/segmentation/data/RADIal/

The structure of the folder should be like this:

```bash
RADIal/
├── ADC_Data
│   ├── adc_000000.npy
│   ├── adc_000001.npy
│   ├── adc_000002.npy
│   ├── adc_000003.npy
│   ├── adc_000004.npy
│   └── ... (More 15084 .npy files)
├── camera
│   ├── image_000018.jpg
│   ├── image_000037.jpg
│   ├── image_000038.jpg
│   ├── image_000039.jpg
│   ├── image_000040.jpg
│   └── ... (More 8247 .jpg files)
├── laser_PCL
│   ├── pcl_000018.npy
│   ├── pcl_000037.npy
│   ├── pcl_000038.npy
│   ├── pcl_000039.npy
│   ├── pcl_000040.npy
│   └── ... (More 8247 .npy files)
├── radar_FFT
│   ├── fft_000018.npy
│   ├── fft_000037.npy
│   ├── fft_000038.npy
│   ├── fft_000039.npy
│   ├── fft_000040.npy
│   └── ... (More 8099 .npy files)
├── radar_Freespace
│   ├── freespace_000018.png
│   ├── freespace_000037.png
│   ├── freespace_000038.png
│   ├── freespace_000039.png
│   ├── freespace_000040.png
│   └── ... (More 8247 .png files)
├── radar_PCL
│   ├── pcl_000018.npy
│   ├── pcl_000037.npy
│   ├── pcl_000038.npy
│   ├── pcl_000039.npy
│   ├── pcl_000040.npy
│   └── ... (More 8247 .npy files)
├── CalibrationTable.npy
├── labels.csv
├── labels_CVPR.csv
```
### Installation

**Step 1:**

```bash
cd {proj_dir}
```

**Step 2: Environment Setup:**

mmMamba recommends setting up a conda environment and installing dependencies via pip. Use the following commands to set up your environment:
Also, We recommend using the pytorch>=2.0, cuda>=11.8. But lower version of pytorch and CUDA are also supported.

***Create and activate a new conda environment***

```bash
conda create -n mmMamba python=3.10
conda activate mmMamba
```

***Install Dependencies***

```bash
pip install torch==2.2 torchvision torchaudio triton pytest chardet yacs termcolor fvcore seaborn packaging ninja einops numpy==1.24.4 timm==0.4.12
pip install https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.2cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
cd {proj_dir}/kernels/selective_scan && pip install .
```
<!-- cd kernels/cross_scan && pip install . -->

***Check Selective Scan (optional)***

* If you want to check the modules compared with `mamba_ssm`, install [`mamba_ssm`](https://github.com/state-spaces/mamba) first!

* If you want to check if the implementation of `selective scan` of ours is the same with `mamba_ssm`, `selective_scan/test_selective_scan.py` is here for you. Change to `MODE = "mamba_ssm_sscore"` in `selective_scan/test_selective_scan.py`, and run `pytest selective_scan/test_selective_scan.py`.

* If you want to check if the implementation of `selective scan` of ours is the same with reference code (`selective_scan_ref`), change to `MODE = "sscore"` in `selective_scan/test_selective_scan.py`, and run `pytest selective_scan/test_selective_scan.py`.

* `MODE = "mamba_ssm"` stands for checking whether the results of `mamba_ssm` is close to `selective_scan_ref`, and `"sstest"` is preserved for development. 

* If you find `mamba_ssm` (`selective_scan_cuda`) or `selective_scan` ( `selctive_scan_cuda_core`) is not close enough to `selective_scan_ref`, and the test failed, do not worry. Check if `mamba_ssm` and `selective_scan` are close enough [instead](https://github.com/state-spaces/mamba/pull/161).

* ***If you are interested in selective scan, you can check [mamba](https://github.com/state-spaces/mamba), [mamba-mini](https://github.com/MzeroMiko/mamba-mini), [mamba.py](https://github.com/alxndrTL/mamba.py) [mamba-minimal](https://github.com/johnma2006/mamba-minimal) for more information.***

***Dependencies for `Detection` and `Segmentation`***

```bash
pip install mmengine==0.10.1 mmcv==2.1.0 opencv-python-headless ftfy regex
pip install mmdet==3.3.0 mmsegmentation==1.2.2 mmpretrain==1.2.0
```
Then make softlink to mmcv, mmengine, mmseg by (take Ubuntu 22.04 for example):
```bash
ln -s /root/miniconda3/envs/mmMamba/lib/python3.10/site-packages/mmcv {proj_dir}/
ln -s /root/miniconda3/envs/mmMamba/lib/python3.10/site-packages/mmengine {proj_dir}/
ln -s /root/miniconda3/envs/mmMamba/lib/python3.10/site-packages/mmseg {proj_dir}/
```
### Model Training and Inference

To train models on RADIal dataset, use the following commands for different configurations:

For **Tiny** model:
```bash
conda activate mmMamba
export PYTHONPATH={proj_dir}:$PYTHONPATH
cd {proj_dir}
/root/miniconda3/envs/mmMamba/bin/python ./segmentation/tools/train.py ./segmentation/configs/mamba/vssm_RADIal_ADC_mba-atten_depth-2222-attenlayer-scan_RRDS-patch_h1_w2.py
```

For **Basic** model:
```bash
conda activate mmMamba 
export PYTHONPATH={proj_dir}:$PYTHONPATH
cd {proj_dir}
/root/miniconda3/envs/mmMamba/bin/python ./segmentation/tools/train.py ./segmentation/configs/mamba/vssm_RADIal_ADC_mba-atten_depth-2244-attenlayer_[n,n,13,13]-scan_RRDS-patch_h1_w2.py
```



If you only want to test the performance:

For **Tiny** model:
```bash
conda activate mmMamba 
export PYTHONPATH={proj_dir}:$PYTHONPATH
cd {proj_dir}
/root/miniconda3/envs/mmMamba/bin/python ./segmentation/tools/test.py ./segmentation/configs/mamba/vssm_RADIal_ADC_mba-atten_depth-2244-attenlayer_[n,n,13,13]-scan_RRDS-patch_h1_w2.py ./work_dirs/vssm_RADIal_ADC_mba-atten_depth-2222-attenlayer-scan_RRDS-patch_h1_w2/20260207_151032/iter_175400.pth --cfg-options default_hooks.visualization.type="RadialVisualizationHook" default_hooks.visualization.interval=1 default_hooks.visualization.draw=True default_hooks.visualization.threshold=0.5 default_hooks.visualization.mapping_color="[46,206,254]"
```
For **Basic** model:
```bash
conda activate mmMamba 
export PYTHONPATH={proj_dir}:$PYTHONPATH
cd {proj_dir}
/root/miniconda3/envs/mmMamba/bin/python ./segmentation/tools/test.py ./segmentation/configs/mamba/vssm_RADIal_ADC_mba-atten_depth-2244-attenlayer_[n,n,13,13]-scan_RRDS-patch_h1_w2.py ./work_dirs/vssm_RADIal_ADC_mba-atten_depth-2244-attenlayer_[n,n,13,13]-scan_RRDS-patch_h1_w2/20260211_153835/iter_175400.pth --cfg-options default_hooks.visualization.type="RadialVisualizationHook" default_hooks.visualization.interval=1 default_hooks.visualization.draw=True default_hooks.visualization.threshold=0.5 default_hooks.visualization.mapping_color="[46,206,254]"
```
