"""
影像切割腳本 - 骨髓幹細胞專案
功能:
1. 已標註資料 (Roboflow): 直接複製到 data/labeled
2. 未標註資料 (IX83): 切割成 16x16 grid (256 張小圖)，並記錄 metadata
"""

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List

from PIL import Image
from tqdm import tqdm

# 提高 PIL 影像大小限制（IX83 全視野影像非常大）
Image.MAX_IMAGE_PIXELS = None  # 移除限制


def parse_donor_info(file_path: Path) -> Dict:
    """
    從檔案路徑解析捐獻者資訊
    
    範例:
        - AA4 NP2 → 捐獻者: A4, 週期: 2
        - PA1 NP9 → 捐獻者: A1, 週期: 9
        - PB135 NP17 → 捐獻者: B135, 週期: 17
    
    說明:
        - 第一個字母 (A/P) 代表年齡組別 (Adult/Pediatric)
        - 第二個字母 (A/B) 代表捐獻者編號前綴
        - 數字代表捐獻者編號
        - NP 後面的數字代表週期 (passage number)
    
    Args:
        file_path: 影像檔案路徑
        
    Returns:
        包含 donor_id, passage_number, original_label 的字典
    """
    # 從父目錄名稱獲取資訊 (例如: AA4 NP2, PA1 NP9)
    parent_dir = file_path.parent.name
    
    donor_id = None
    passage_number = None
    
    # 解析格式: AA4 NP2, AB10 NP2, PA1 NP9, PB135 NP17
    # 捕獲群組: (A|P)(A|B)(數字) NP(數字)
    match = re.match(r'([AP])([AB])(\d+)\s+NP(\d+)', parent_dir)
    if match:
        donor_prefix = match.group(2)  # A 或 B
        donor_number = match.group(3)  # 4, 10, 1, 135
        passage = match.group(4)       # 2, 9, 17
        
        # 捐獻者編號只保留後面的字母和數字
        # AA4 → A4, AB10 → B10, PA1 → A1, PB135 → B135
        donor_id = f"{donor_prefix}{donor_number}"
        passage_number = int(passage)
    
    return {
        'donor_id': donor_id,
        'passage_number': passage_number,
        'original_label': parent_dir
    }


def tile_image_16x16(
    image_path: Path,
    output_dir: Path,
    age_group: str
) -> List[Dict]:
    """
    將影像切割成 16x16 網格 (共 256 張小圖)
    
    Args:
        image_path: 輸入影像路徑
        output_dir: 輸出目錄
        age_group: 年齡組別 ("Adult" 或 "Pediatric")
    
    Returns:
        切塊資訊列表，包含 metadata
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 開啟影像
    with Image.open(image_path) as img:
        w, h = img.size
        
        # 計算每個切塊的大小
        tile_w = w // 16
        tile_h = h // 16
        
        # 解析捐獻者資訊
        donor_info = parse_donor_info(image_path)
        
        tiles_info = []
        tile_idx = 0
        
        print(f"  處理 {image_path.name}: {w}x{h} → 16x16 grid ({tile_w}x{tile_h} per tile)")
        
        for row in range(16):
            for col in range(16):
                # 計算切塊邊界
                x_start = col * tile_w
                y_start = row * tile_h
                
                # 最後一列/行延伸到影像邊界
                x_end = w if col == 15 else x_start + tile_w
                y_end = h if row == 15 else y_start + tile_h
                
                # 切割影像
                tile = img.crop((x_start, y_start, x_end, y_end))
                
                # 生成檔名: 原檔名_r行_c列.副檔名
                # 例如: 20250220_AA4 NP2_r00_c00.tif
                tile_name = f"{image_path.stem}_r{row:02d}_c{col:02d}{image_path.suffix}"
                tile_path = output_dir / tile_name
                
                # 儲存切塊
                tile.save(tile_path)
                
                # 記錄 metadata
                tile_info = {
                    'tile_name': tile_name,
                    'original_image': image_path.name,
                    'original_path': str(image_path),
                    'age_group': age_group,
                    'donor_id': donor_info['donor_id'],
                    'passage_number': donor_info['passage_number'],
                    'original_label': donor_info['original_label'],
                    'grid_position': {
                        'row': row,
                        'col': col,
                        'total_grid': '16x16'
                    },
                    'bbox': {
                        'x': x_start,
                        'y': y_start,
                        'width': x_end - x_start,
                        'height': y_end - y_start
                    },
                    'tile_index': tile_idx
                }
                
                tiles_info.append(tile_info)
                tile_idx += 1
        
        print(f"    ✓ 生成 {len(tiles_info)} 張切塊")
    
    return tiles_info


def process_labeled_data(
    roboflow_dir: Path, 
    output_base: Path
):
    """
    處理已標註資料 (Roboflow) - 合併 train/valid/test 到 data/labeled
    """
    print("\n" + "="*60)
    print("📋 處理已標註資料 (Roboflow)")
    print("="*60)
    
    if not roboflow_dir.exists():
        print(f"⚠️  找不到 Roboflow 目錄: {roboflow_dir}")
        return
    
    output_dir = output_base / "labeled"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 合併所有 COCO 標註
    all_images = []
    all_annotations = []
    all_categories = None
    next_image_id = 1
    next_anno_id = 1
    image_id_mapping = {}  # 原始 image_id -> 新 image_id
    
    total_images = 0
    
    for split in ["train", "valid", "test"]:
        split_dir = roboflow_dir / split
        if not split_dir.exists():
            print(f"\n⚠️  跳過不存在的目錄: {split}")
            continue
        
        # 找出所有影像
        images = (
            list(split_dir.glob("*.jpg")) +
            list(split_dir.glob("*.png")) +
            list(split_dir.glob("*.tif")) +
            list(split_dir.glob("*.tiff"))
        )
        
        if not images:
            print(f"\n⚠️  {split} 中沒有找到影像")
            continue
        
        print(f"\n處理 {split} 集合: {len(images)} 張影像")
        
        # 複製影像
        for img_path in tqdm(images, desc=f"  複製 {split} 影像"):
            shutil.copy2(img_path, output_dir / img_path.name)
        
        total_images += len(images)
        
        # 讀取並合併 COCO 標註
        anno_file = split_dir / "_annotations.coco.json"
        if anno_file.exists():
            with open(anno_file, 'r', encoding='utf-8') as f:
                coco_data = json.load(f)
            
            # 設定 categories (所有 split 應該相同)
            if all_categories is None:
                all_categories = coco_data.get('categories', [])
            
            # 處理 images
            for img_info in coco_data.get('images', []):
                old_image_id = img_info['id']
                img_info['id'] = next_image_id
                image_id_mapping[old_image_id] = next_image_id
                all_images.append(img_info)
                next_image_id += 1
            
            # 處理 annotations，更新 image_id
            for anno in coco_data.get('annotations', []):
                old_image_id = anno['image_id']
                anno['id'] = next_anno_id
                anno['image_id'] = image_id_mapping.get(old_image_id, old_image_id)
                all_annotations.append(anno)
                next_anno_id += 1
            
            print(f"  ✓ 已讀取 {split} 標註: {len(coco_data.get('images', []))} 張影像, {len(coco_data.get('annotations', []))} 個標註")
    
    # 儲存合併後的 COCO 標註
    if all_images:
        merged_coco = {
            'images': all_images,
            'annotations': all_annotations,
            'categories': all_categories if all_categories else []
        }
        
        output_anno_file = output_dir / "_annotations.coco.json"
        with open(output_anno_file, 'w', encoding='utf-8') as f:
            json.dump(merged_coco, f, indent=2, ensure_ascii=False)
        
        print(f"\n  ✅ 已合併標註檔:")
        print(f"     - 總影像數: {len(all_images)}")
        print(f"     - 總標註數: {len(all_annotations)}")
        print(f"     - 類別數: {len(all_categories) if all_categories else 0}")
        print(f"     - 輸出至: {output_anno_file}")
    
    print(f"\n✅ 已標註資料處理完成 → {output_dir}")
    print(f"   總共 {total_images} 張影像")


def process_unlabeled_data(
    unlabeled_base: Path,
    output_base: Path
):
    """
    處理未標註資料 (IX83) - 切割成 16x16 grid 並記錄 metadata
    """
    print("\n" + "="*60)
    print("📸 處理未標註資料 (IX83 全視野)")
    print("="*60)
    
    if not unlabeled_base.exists():
        print(f"⚠️  找不到未標註資料目錄: {unlabeled_base}")
        return
    
    # 處理 Adult MSC 和 Pediatric MSC
    for subdir_name, age_group in [("Adult MSC", "Adult"), ("Pediatric MSC", "Pediatric")]:
        input_dir = unlabeled_base / subdir_name
        
        if not input_dir.exists():
            print(f"\n⚠️  跳過不存在的目錄: {subdir_name}")
            continue
        
        output_dir = output_base / "unlabeled" / subdir_name.replace(" ", "_")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 找出所有影像 (包含子目錄)，排除廢棄資料
        all_images = (
            list(input_dir.rglob("*.jpg")) +
            list(input_dir.rglob("*.png")) +
            list(input_dir.rglob("*.tif")) +
            list(input_dir.rglob("*.tiff"))
        )
        
        # 過濾掉廢棄資料夾中的影像
        excluded_folders = ["AI照片過密", "其他"]
        images = [
            img for img in all_images 
            if not any(excluded in str(img) for excluded in excluded_folders)
        ]
        
        if len(all_images) > len(images):
            print(f"  ⚠️  已排除 {len(all_images) - len(images)} 張影像（來自廢棄資料夾: {', '.join(excluded_folders)}）")
        
        if not images:
            print(f"\n⚠️  {subdir_name} 中沒有找到影像")
            continue
        
        print(f"\n處理 {subdir_name}:")
        print(f"  年齡組別: {age_group}")
        print(f"  影像數量: {len(images)}")
        print(f"  切割方式: 16x16 grid (每張 → 256 小圖)")
        
        all_tiles_info = []
        
        # 切割每張影像
        for img_path in images:
            try:
                tiles_info = tile_image_16x16(
                    img_path,
                    output_dir,
                    age_group
                )
                all_tiles_info.extend(tiles_info)
            except Exception as e:
                print(f"\n  ⚠️  處理 {img_path.name} 時發生錯誤: {e}")
                continue
        
        # 儲存 metadata
        metadata_file = output_dir / "tiles_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(all_tiles_info, f, indent=2, ensure_ascii=False)
        
        # 產生統計摘要
        summary = {
            'age_group': age_group,
            'total_original_images': len(images),
            'total_tiles': len(all_tiles_info),
            'tiles_per_image': 256,
            'grid_size': '16x16',
            'donors': list(set(t['donor_id'] for t in all_tiles_info if t['donor_id'])),
            'passages': sorted(set(t['passage_number'] for t in all_tiles_info if t['passage_number']))
        }
        
        summary_file = output_dir / "summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n  摘要:")
        print(f"    - 原始影像: {summary['total_original_images']} 張")
        print(f"    - 切塊總數: {summary['total_tiles']} 張")
        print(f"    - 捐獻者: {len(summary['donors'])} 位 → {summary['donors']}")
        print(f"    - 週期範圍: NP{min(summary['passages']) if summary['passages'] else 'N/A'} - NP{max(summary['passages']) if summary['passages'] else 'N/A'}")
        print(f"  ✅ {subdir_name} 完成 → {output_dir}")
        print(f"     Metadata: {metadata_file.name}")
        print(f"     Summary: {summary_file.name}")


def main():
    parser = argparse.ArgumentParser(
        description="骨髓幹細胞影像切割工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例用法:
  # 處理所有資料
  python scripts/tile_images.py
  
  # 只處理已標註資料
  python scripts/tile_images.py --labeled-only
  
  # 只處理未標註資料
  python scripts/tile_images.py --unlabeled-only
  
  # 自訂路徑
  python scripts/tile_images.py --raw-data-dir "custom/path" --output-dir "output/path"
        """
    )
    
    parser.add_argument(
        "--raw-data-dir",
        type=str,
        default="raw_data/01 AI 分析細胞照片",
        help="原始資料目錄"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="輸出目錄"
    )
    
    parser.add_argument(
        "--labeled-only",
        action="store_true",
        help="只處理已標註資料"
    )
    
    parser.add_argument(
        "--unlabeled-only",
        action="store_true",
        help="只處理未標註資料"
    )
    
    args = parser.parse_args()
    
    # 設定路徑
    raw_data_dir = Path(args.raw_data_dir)
    output_dir = Path(args.output_dir)
    
    roboflow_dir = raw_data_dir / "05 輔仁大學Roboflow"
    unlabeled_base = raw_data_dir / "02 IX83 全通量細胞照片"
    
    print("\n" + "="*60)
    print("  骨髓幹細胞影像切割工具")
    print("="*60)
    print(f"\n設定:")
    print(f"  原始資料: {raw_data_dir.absolute()}")
    print(f"  輸出目錄: {output_dir.absolute()}")
    print(f"  模式: ", end="")
    if args.labeled_only:
        print("僅處理已標註資料")
    elif args.unlabeled_only:
        print("僅處理未標註資料")
    else:
        print("處理所有資料")
    
    # 處理資料
    if not args.unlabeled_only:
        process_labeled_data(roboflow_dir, output_dir)
    
    if not args.labeled_only:
        process_unlabeled_data(unlabeled_base, output_dir)
    
    print("\n" + "="*60)
    print("  ✅ 所有處理完成！")
    print("="*60)
    print(f"\n處理後的資料位於: {output_dir.absolute()}")
    print("\n資料夾結構:")
    print("  data/")
    print("    ├── labeled/              # 已標註資料 (Roboflow 合併)")
    print("    │   ├── *.jpg/png         (所有已標註影像)")
    print("    │   └── _annotations.coco.json  (合併後的標註)")
    print("    └── unlabeled/            # 未標註資料 (IX83)")
    print("        ├── Adult_MSC/")
    print("        │   ├── *_r00_c00.tif   (256 tiles per image)")
    print("        │   ├── tiles_metadata.json")
    print("        │   └── summary.json")
    print("        └── Pediatric_MSC/")
    print("            ├── *_r00_c00.tif")
    print("            ├── tiles_metadata.json")
    print("            └── summary.json")
    print("\nLabeled data:")
    print("  - 合併 train/valid/test 為單一資料集")
    print("  - COCO 格式標註，image_id 已重新編號")
    print("\nUnlabeled data metadata 包含:")
    print("  - age_group: Adult/Pediatric")
    print("  - donor_id: A4, B10, A1, B135, etc.")
    print("  - passage_number: 2, 9, 17, etc.")
    print("  - grid_position: 16x16 網格位置")
    print("  - bbox: 原始影像中的座標")


if __name__ == "__main__":
    main()
