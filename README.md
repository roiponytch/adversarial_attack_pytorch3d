# Adversarial attack with rendering 
Render adversarial image

- load a mesh and textures from .obj (e.g person)
- set up a renderer
- render the mesh
- vary the rendering settings such as lighting and camera position
- load object detection model (e.g FaaterRcnn pretrained on COCO)
- optimizing texture to fool the object detection model 

# Installation

## Requirements

- Ubuntu
- CUDA 11.3
- Python 3.8 

## Install wheels for Linux

### 1. PyTorch and other requirements
```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```
### 2. PyTorch3D
For example, to install for Python 3.8, PyTorch 1.11.0 and CUDA 11.3
```
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html
```
### 3. mmcv-full
```
 pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
```
### 4. mmdetection
```
git clone https://github.com/roiponytch/mmdetection
cd mmdetection
pip install -r requirements/build.txt
pip install -v -e .
```

## Install mmcv-full + mmdetection wheels for Linux (from source)
```
pip uninstall mmcv-full
pip uninstall mmdet
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
MMCV_WITH_OPS=1 pip install -e .  # package mmcv-full will be installed after this step
cd ..
git clone https://github.com/roiponytch/mmdetection
cd mmdetection
pip install -r requirements/build.txt
pip install -v -e .
```

## Running the code:
In order to run the code some files (trained model, 3D model obj, etc.) need to be download inside the project under 'run_files' folder.  
```
https://drive.google.com/drive/folders/11FOrsMEv3xMQCzF6T8elDf7XlwbQkSye?usp=sharing
```
Run the script 'mmdetection_adv_attack_pytorch3d.py'