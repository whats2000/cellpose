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

### 3. 執行主動學習

```bash
# 啟動主動學習管線
python src/active_learning_framework.py
```

---

## 📋 完整工作流程

### Phase 0: 環境設定 (第1天)

#### 步驟 1: 安裝套件
```bash
cd h:\cellpose
conda create -n cellpose-active python=3.10 -y
conda activate cellpose-active
pip install cellpose[gui] scikit-learn pandas matplotlib seaborn tqdm jupyterlab
```

#### 步驟 2: 驗證安裝
```bash
python -c "import cellpose; print(f'Cellpose version: {cellpose.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### 步驟 3: 啟動 Cellpose GUI
```bash
python -m cellpose
```

---

### Phase 1: 資料準備 (第2-3天)

#### 步驟 1: 檢查現有資料
```bash
python scripts/check_data.py
```

**預期輸出：**
```
已標記資料 (Roboflow):
  - train: 15 張影像
  - valid: 3 張影像
  - test: 2 張影像

未標記資料:
  - Adult MSC: ~2000 張影像
  - Pediatric MSC: ~1500 張影像
```

#### 步驟 2: 處理大型切片影像
```bash
# 將大型全視野影像切割成小塊
python scripts/tile_large_images.py \
    --input "raw_data/01 AI 分析細胞照片/02 IX83 全通量細胞照片" \
    --output "data/tiled_images" \
    --tile_size 1024 \
    --overlap 128
```

#### 步驟 3: 轉換標註格式
```bash
# 將 COCO 格式轉換為 Cellpose 格式
python scripts/convert_coco_to_cellpose.py \
    --input "raw_data/01 AI 分析細胞照片/05 輔仁大學Roboflow" \
    --output "data/cellpose_format"
```

---

### Phase 2: 基礎模型訓練 (第4-7天)

#### 步驟 1: 訓練初始模型
```python
# notebooks/01_train_initial_model.ipynb
from cellpose import models, io

# 載入資料
train_dir = "data/cellpose_format/train"
images, labels, image_names = io.load_train_test_data(train_dir)

# 建立模型
model = models.CellposeModel(gpu=True, pretrained_model='cpsam')

# 訓練
model_path = model.train(
    train_data=images,
    train_labels=labels,
    save_path="models/",
    n_epochs=100,
    learning_rate=0.001,
    model_name="bone_marrow_initial"
)
```

#### 步驟 2: 評估基礎模型
```bash
python scripts/evaluate_model.py \
    --model "models/bone_marrow_initial" \
    --test_dir "data/cellpose_format/test"
```

#### 步驟 3: 建立細胞分類器
```python
# notebooks/02_train_classifier.ipynb
# 提取細胞特徵 -> 訓練 3 類分類器（早/中/晚期）
```

---

### Phase 3: 主動學習迭代 (第8-60天，每週1-2次迭代)

#### 啟動主動學習系統
```bash
python src/active_learning_framework.py
```

**每輪迭代流程：**

```
┌─────────────────────────────────────────┐
│  Iteration N                            │
├─────────────────────────────────────────┤
│  1. 模型預測未標記資料                    │
│  2. 計算不確定性分數                      │
│  3. 選擇 100 個最有價值的樣本              │
│  4. 📋 人工標註 (Cellpose GUI)           │
│  5. 增量訓練模型                         │
│  6. 驗證效能                             │
│  7. 檢查是否達標                         │
│     └─ 是: 結束訓練                      │
│     └─ 否: 繼續下一輪                    │
└─────────────────────────────────────────┘
```

#### 人工標註步驟（每輪迭代）

1. **開啟 Cellpose GUI**
   ```bash
   python -m cellpose
   ```

2. **載入待標註影像**
   - 影像清單位於：`active_learning_workspace/iteration_XX/to_annotate.txt`
   - 逐一開啟影像

3. **標註細胞**
   - 使用 Cellpose 工具繪製細胞輪廓
   - 為每個細胞標記類別：
     - Label 1: 早期細胞 (Early Stage)
     - Label 2: 中期細胞 (Middle Stage)
     - Label 3: 晚期細胞 (Late Stage)

4. **儲存標註**
   - 格式：`_seg.npy` (Cellpose 格式)
   - 儲存位置：`active_learning_workspace/iteration_XX/annotations/`

5. **繼續訓練**
   - 完成標註後，在終端按 Enter 繼續

---

### Phase 4: 大規模預測 (第61-70天)

#### 步驟 1: 批次預測所有資料
```bash
python scripts/batch_inference.py \
    --model "models/bone_marrow_final" \
    --input_dir "raw_data/01 AI 分析細胞照片/02 IX83 全通量細胞照片" \
    --output_dir "results/predictions" \
    --batch_size 16
```

#### 步驟 2: 統計分析
```python
# notebooks/03_statistical_analysis.ipynb
import pandas as pd
import matplotlib.pyplot as plt

# 載入預測結果
predictions = pd.read_csv("results/predictions/summary.csv")

# 分析各族群數量
stage_counts = predictions.groupby(['image_type', 'cell_stage']).size()
print(stage_counts)

# 視覺化
stage_counts.unstack().plot(kind='bar', stacked=True)
plt.title("細胞年齡分佈")
plt.xlabel("樣本類型")
plt.ylabel("細胞數量")
plt.savefig("results/cell_stage_distribution.png")
```

---

## 🔧 輔助腳本說明

### 1. 資料檢查腳本 (`scripts/check_data.py`)

```python
"""檢查資料完整性與統計"""
import os
from pathlib import Path

def check_data():
    base_path = Path("raw_data/01 AI 分析細胞照片")
    
    # 檢查已標記資料
    labeled_path = base_path / "05 輔仁大學Roboflow"
    for split in ["train", "valid", "test"]:
        split_dir = labeled_path / split
        if split_dir.exists():
            images = list(split_dir.glob("*.jpg"))
            print(f"{split}: {len(images)} 張影像")
    
    # 檢查未標記資料
    unlabeled_path = base_path / "02 IX83 全通量細胞照片"
    for subdir in ["Adult MSC", "Pediatric MSC"]:
        images = list((unlabeled_path / subdir).rglob("*.jpg"))
        print(f"{subdir}: {len(images)} 張影像")

if __name__ == "__main__":
    check_data()
```

### 2. 影像切片腳本 (`scripts/tile_large_images.py`)

```python
"""將大型全視野影像切割成小塊"""
import cv2
import numpy as np
from pathlib import Path
import argparse

def tile_image(image_path, tile_size=1024, overlap=128):
    """切割單張影像"""
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    
    stride = tile_size - overlap
    tiles = []
    
    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            tile = img[y:y+tile_size, x:x+tile_size]
            tiles.append((tile, (x, y)))
    
    return tiles

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tile_size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=128)
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for img_path in input_path.rglob("*.jpg"):
        print(f"處理: {img_path.name}")
        tiles = tile_image(img_path, args.tile_size, args.overlap)
        
        for i, (tile, (x, y)) in enumerate(tiles):
            tile_name = f"{img_path.stem}_tile_{i:04d}_x{x}_y{y}.jpg"
            cv2.imwrite(str(output_path / tile_name), tile)
    
    print(f"✅ 完成！切片儲存於: {output_path}")

if __name__ == "__main__":
    main()
```

### 3. 格式轉換腳本 (`scripts/convert_coco_to_cellpose.py`)

```python
"""將 COCO 格式轉換為 Cellpose _seg.npy 格式"""
import json
import numpy as np
from pathlib import Path
import cv2
from pycocotools import mask as maskUtils

def convert_coco_to_cellpose(coco_file, output_dir):
    """轉換單個 COCO 標註檔"""
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for img_info in coco_data['images']:
        img_id = img_info['id']
        h, w = img_info['height'], img_info['width']
        
        # 建立空白 mask
        masks = np.zeros((h, w), dtype=np.uint16)
        
        # 填入每個細胞的 mask
        cell_id = 1
        for anno in coco_data['annotations']:
            if anno['image_id'] == img_id:
                # 解析 segmentation
                # TODO: 根據實際格式解析
                cell_id += 1
        
        # 儲存為 _seg.npy
        output_file = output_dir / f"{img_info['file_name'].replace('.jpg', '_seg.npy')}"
        np.save(output_file, masks)

# 使用範例
if __name__ == "__main__":
    convert_coco_to_cellpose(
        "raw_data/01 AI 分析細胞照片/05 輔仁大學Roboflow/train/_annotations.coco.json",
        "data/cellpose_format/train"
    )
```

---

## 📊 監控訓練進度

### 1. 查看訓練歷史
```bash
python scripts/plot_training_history.py
```

### 2. 即時監控（使用 TensorBoard）
```bash
tensorboard --logdir active_learning_workspace/logs
```

### 3. 檢查當前效能
```bash
python scripts/evaluate_current_model.py
```

---

## ⚠️ 常見問題與解決

### Q1: GPU 記憶體不足
```python
# 在 active_learning_framework.py 中調整
config.batch_size = 4  # 降低 batch size
```

### Q2: 標註速度太慢
**解決方案：**
- 使用半自動標註：先用模型預測，再手動修正
- 培訓多位標註人員
- 使用更高效的標註工具（如 CVAT）

### Q3: 模型效能提升緩慢
**檢查清單：**
- [ ] 標註品質是否一致？
- [ ] 資料是否有類別不平衡？
- [ ] 是否需要更多的資料增強？
- [ ] 學習率是否合適？

---

## 📈 效能指標追蹤

### 每輪迭代記錄
```
Iteration   |  Train Size  |  Val IoU  |  Val Acc  |  Time
-----------|--------------|-----------|-----------|--------
    0      |      20      |   0.52    |   0.63    |  2h
    1      |      70      |   0.58    |   0.71    |  2.5h
    2      |     170      |   0.65    |   0.78    |  3h
    3      |     270      |   0.71    |   0.84    |  3.5h
    4      |     370      |   0.76    |   0.88    |  4h
    5      |     470      |   0.79    |   0.91    |  4.5h
```

### 目標指標
- ✅ **分割 IoU**: > 0.75
- ✅ **分類準確率**: > 0.90
- ✅ **每類 F1-score**: > 0.85

---

## 🎯 下一步行動檢查清單

### 本週任務
- [ ] 安裝 Cellpose 環境
- [ ] 執行 `check_data.py` 檢查資料
- [ ] 解壓 Roboflow 資料
- [ ] 執行影像切片腳本
- [ ] 確認 GPU 可用性

### 下週任務
- [ ] 轉換標註格式
- [ ] 訓練初始分割模型
- [ ] 訓練初始分類器
- [ ] 評估基礎效能

### 需要討論
1. **細胞分類標準**：如何區分早/中/晚期？
2. **標註團隊**：誰負責標註？需要培訓嗎?
3. **計算資源**：GPU 規格？是否需要雲端資源？
4. **時程安排**：每週可以投入多少時間標註？

---

## 📞 支援資源

- **Cellpose 文件**: https://cellpose.readthedocs.io/
- **Cellpose GitHub**: https://github.com/MouseLand/cellpose
- **Image.sc 論壇**: https://forum.image.sc/tag/cellpose
- **本專案問題**: 請建立 GitHub Issue

---

**版本**: 1.0  
**更新日期**: 2025-10-16  
**作者**: AI Assistant
