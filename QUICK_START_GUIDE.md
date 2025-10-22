# 骨髓幹細胞年齡分析：快速開始指南

## 🚀 快速開始 (5分鐘)

### 1. 環境準備

```bash
# 建立 Python 環境
conda create -n cellpose-active python=3.10
conda activate cellpose-active

# 安裝 Cellpose-SAM
pip install cellpose[gui]

# 安裝其他依賴
pip install scikit-learn pandas matplotlib seaborn tqdm
```

### 2. 檢查資料

```bash
# 執行資料檢查腳本
python scripts/check_data.py
```

### 3. 切片資料

```bash
# 執行顯微鏡切片
python scripts/tile_images.py
```
