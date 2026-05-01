# Open3D: A Toolbox for 3D Object Detection Methods

## 1. Setup
### 1.1. Using conda
#### 1.1.1. Using environment.yml
```bash
conda env create -f envs/environment.yml
conda activate anomaly
pip install -e .
```

#### 1.1.2 Using requirements.txt
```bash
conda create --name anomaly python=3.10.12
conda activate anomaly
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r envs/requirements.txt
pip install -e .
```

### 1.2 Using uv
```bash
uv venv anomaly --python=3.10.12
source anomaly/bin/activate
uv pip install -e .
```

## 2. Citation
If you find our work useful, please cite the following:
```
@misc{Chi2025,
  author       = {Chi Tran},
  title        = {Open3D: A Toolbox for 3D Object Detection Methods},
  publisher    = {GitHub},
  booktitle    = {GitHub repository},
  howpublished = {https://github.com/IceIce1ce/Open3D},
  year         = {2025}
}
```

## 3. Contact
If you have any questions, feel free to contact `Chi Tran`
([ctran743@gmail.com](ctran743@gmail.com) or [tdc2000@skku.edu](tdc2000@skku.edu)).

## 4. Acknowledgement
Our framework is built using multiple open source, thanks for their great contributions.