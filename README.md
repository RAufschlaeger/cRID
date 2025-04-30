# Experiments on Person Re-ID using Cross-Modal Intelligence

## Copyright info

This work is mostly based on the repository [https://github.com/michuanhaohao/reid-strong-baseline](https://github.com/michuanhaohao/reid-strong-baseline) (README: [Bag of Tricks and A Strong ReID Baseline](#bag-of-tricks-and-a-strong-reid-baseline), License: LICENCE (REID-STRONG-BASELINE) copy.md (MIT)). We are grateful using their work and aware to stand on the shoulders of giants.

Worth mentioning, interpretations in ./inference are conducted using [https://github.com/jordan7186/GAtt](https://github.com/jordan7186/GAtt).

All adapted files are commented.

The software is licensed using MIT (LICENSE.md).


## Installation

**OS**: Ubuntu 22.04.4 LTS

### Install CUDA and CUDNN

```bash
conda install -c "nvidia/label/cuda-11.3.0" cuda-nvcc
conda install -c anaconda cudatoolkit
conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia

wget https://developer.download.nvidia.com/compute/cudnn/9.2.1/local_installers/cudnn-local-repo-ubuntu2204-9.2.1_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2204-9.2.1_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-9.2.1/cudnn-local-643687A8-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudnn
sudo apt update
```

### Create and set up Conda environment

```bash
conda create -n dolphin python=3.11
conda activate dolphin
conda update -n base -c defaults conda
```

### Install project requirements

```bash
pip install -r requirements.txt
conda install -c "nvidia/label/cuda-11.3.0" cuda-nvcc
conda install -c anaconda cudatoolkit
conda install pytorch pytorch-cuda -c pytorch -c nvidia
conda install torchvision torchaudio -c pytorch -c nvidia
```

### Download ResNet50
```bash
python
>>> from torchvision import models
>>> torch.hub.set_dir('/home/raufschlaeger/.torch/models')
>>> model = models.resnet50(pretrained=True)
```

### Set up Jupyter kernel (optional)

```bash
python -m ipykernel install --user --name dolphin --display-name "dolphin"
```

### Navigate to project directory

```bash
cd crid
```

### preprocessing

#### Download Market1501 (https://www.kaggle.com/datasets/whurobin/market1501)
```
cd data/raw
kaggle datasets download -d whurobin/market1501
unzip market1501.zip
```

#### Download CUHK03:
download from google drive link given at https://github.com/zaidbhat1234/CUHK03-dataset?tab=readme-ov-file
unzip in data/raw

```
.
├── cuhk03-np
│   ├── detected
│   │   ├── bounding_box_test
│   │   ├── bounding_box_train
│   │   └── query
│   └── labeled
│       ├── bounding_box_test
│       ├── bounding_box_train
│       └── query
└── Market-1501-v15.09.15
    ├── bounding_box_test
    ├── bounding_box_train
    ├── gt_bbox
    ├── gt_query
    ├── query
    └── readme.txt
```

## Repository Structure

This repository is organized for modularity and extensibility. Below is an overview of the main directories and their purposes.

### `data/` - Data Handling and Processing

- **`data/datasets/`**: Contains dataset loader classes for various benchmarks (e.g., Market1501, CUHK03, CUHK03-np, MSMT17, VeRi, DukeMTMC-reID). Each file implements a dataset-specific loader inheriting from a common base. Currently, we only use Market1501 and CUHK03-np/detected.
- **`data/build.py`**: Main entry point for constructing PyTorch data loaders for training and evaluation, supporting both image and graph modalities.
- **`data/scripts/`**: Utility scripts for dataset preprocessing, annotation file creation, and conversion between formats (e.g., `create_annotation_files.py`, `create_graph_dataset.py`, `create_image_dataset.py`, `write_graphs.py`).
- **`data/src/`**: Custom code for advanced data processing and modeling.
  - **`data/src/data/graph_dataset.py`**: Defines a PyTorch dataset for scene graphs, including error handling and embedding generation.
  - **`data/src/processing/scene_graph.py`**: Scene graph parsing, cleaning, and conversion utilities.
  - **`data/src/models/`**: Custom model wrappers (e.g., Molmo-7B).
  - **`data/src/utils/`**: Utility functions.
- **`data/__init__.py`**: Imports the main data loader function for use elsewhere in the codebase.

### Other Key Directories

- **`configs/`**: YAML configuration files for different training/testing setups and model backbones.
- **`modeling/`**: Model architectures and backbone definitions.
- **`tools/`**: Training, testing, and utility scripts.
- **`inference/`**: Scripts for model interpretation and visualization (e.g., attention maps).
- **`imgs/`**: Images for documentation and pipeline visualization.
- **`logs/`**: Output logs and checkpoints from training/testing runs.

### Example Workflow

1. **Preprocessing**: Use scripts in `data/scripts/` to generate annotation files and graph/image datasets.
2. **Training**: Use configuration files in `configs/` and launch training via scripts in `tools/`.
3. **Evaluation/Inference**: Use scripts in `tools/` and `inference/` for testing and visualization.
