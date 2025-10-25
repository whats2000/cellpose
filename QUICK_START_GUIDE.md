# Bone Marrow Stem Cell Age Analysis: Quick Start Guide

## 🚀 Quick Start (5 minutes)

### 1. Environment Setup

```bash
# Create Python environment
conda create -n cellpose-active python=3.10
conda activate cellpose-active

# Install Cellpose-SAM
pip install cellpose[gui]

# Install other dependencies
pip install scikit-learn pandas matplotlib seaborn tqdm
```

### 2. Check Data

```bash
# Run data checking script
python scripts/check_data.py
```

### 3. Tile Data

```bash
# Run microscope tiling
python scripts/tile_images.py
```
