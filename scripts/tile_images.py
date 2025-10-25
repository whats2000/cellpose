"""
Image tiling script - Bone marrow stem cell project
Features:
1. Labeled data (Roboflow): Copy directly to data/labeled
2. Unlabeled data (IX83): Tile into 16x16 grid (256 tiles per image) and record metadata
"""

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List

from PIL import Image
from tqdm import tqdm

# Increase PIL image size limit (IX83 full-field images are very large)
Image.MAX_IMAGE_PIXELS = None  # Remove limit


def parse_donor_info(file_path: Path) -> Dict:
    """
    Parse donor information from file path
    
    Examples:
        - AA4 NP2 → Donor: A4, Passage: 2
        - PA1 NP9 → Donor: A1, Passage: 9
        - PB135 NP17 → Donor: B135, Passage: 17
    
    Explanation:
        - First letter (A/P) represents age group (Adult/Pediatric)
        - Second letter (A/B) represents donor ID prefix
        - Numbers represent donor ID
        - NP followed by numbers represent passage number
    
    Args:
        file_path: Image file path
        
    Returns:
        Dictionary containing donor_id, passage_number, original_label
    """
    # Get information from parent directory name (e.g.: AA4 NP2, PA1 NP9)
    parent_dir = file_path.parent.name
    
    donor_id = None
    passage_number = None
    
    # Parse format: AA4 NP2, AB10 NP2, PA1 NP9, PB135 NP17
    # Capture groups: (A|P)(A|B)(digits) NP(digits)
    match = re.match(r'([AP])([AB])(\d+)\s+NP(\d+)', parent_dir)
    if match:
        donor_prefix = match.group(2)  # A or B
        donor_number = match.group(3)  # 4, 10, 1, 135
        passage = match.group(4)       # 2, 9, 17
        
        # Donor ID only keeps the letters and numbers after
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
    Tile image into 16x16 grid (256 tiles total)
    
    Args:
        image_path: Input image path
        output_dir: Output directory
        age_group: Age group ("Adult" or "Pediatric")
    
    Returns:
        List of tile information containing metadata
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open image
    with Image.open(image_path) as img:
        w, h = img.size
        
        # Calculate size of each tile
        tile_w = w // 16
        tile_h = h // 16
        
        # Parse donor information
        donor_info = parse_donor_info(image_path)
        
        tiles_info = []
        tile_idx = 0
        
        print(f"  Processing {image_path.name}: {w}x{h} → 16x16 grid ({tile_w}x{tile_h} per tile)")
        
        for row in range(16):
            for col in range(16):
                # Calculate tile boundaries
                x_start = col * tile_w
                y_start = row * tile_h
                
                # Extend to image boundary for last row/column
                x_end = w if col == 15 else x_start + tile_w
                y_end = h if row == 15 else y_start + tile_h
                
                # Crop image
                tile = img.crop((x_start, y_start, x_end, y_end))
                
                # Generate filename: original_name_r[row]_c[col].extension
                # Example: 20250220_AA4 NP2_r00_c00.tif
                tile_name = f"{image_path.stem}_r{row:02d}_c{col:02d}{image_path.suffix}"
                tile_path = output_dir / tile_name
                
                # Save tile
                tile.save(tile_path)
                
                # Record metadata
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
        
        print(f"    ✓ Generated {len(tiles_info)} tiles")
    
    return tiles_info


def process_labeled_data(
    roboflow_dir: Path, 
    output_base: Path
):
    """
    Process labeled data (Roboflow) - merge train/valid/test to data/labeled
    """
    print("\n" + "="*60)
    print("📋 Processing labeled data (Roboflow)")
    print("="*60)
    
    if not roboflow_dir.exists():
        print(f"⚠️  Roboflow directory not found: {roboflow_dir}")
        return
    
    output_dir = output_base / "labeled"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Merge all COCO annotations
    all_images = []
    all_annotations = []
    all_categories = None
    next_image_id = 1
    next_anno_id = 1
    image_id_mapping = {}  # original image_id -> new image_id
    
    total_images = 0
    
    for split in ["train", "valid", "test"]:
        split_dir = roboflow_dir / split
        if not split_dir.exists():
            print(f"\n⚠️  Skipping non-existent directory: {split}")
            continue
        
        # Find all images
        images = (
            list(split_dir.glob("*.jpg")) +
            list(split_dir.glob("*.png")) +
            list(split_dir.glob("*.tif")) +
            list(split_dir.glob("*.tiff"))
        )
        
        if not images:
            print(f"\n⚠️  No images found in {split}")
            continue
        
        print(f"\nProcessing {split} set: {len(images)} images")
        
        # Copy images
        for img_path in tqdm(images, desc=f"  Copying {split} images"):
            shutil.copy2(img_path, output_dir / img_path.name)
        
        total_images += len(images)
        
        # Read and merge COCO annotations
        anno_file = split_dir / "_annotations.coco.json"
        if anno_file.exists():
            with open(anno_file, 'r', encoding='utf-8') as f:
                coco_data = json.load(f)
            
            # Set categories (should be same across all splits)
            if all_categories is None:
                all_categories = coco_data.get('categories', [])
            
            # Process images
            for img_info in coco_data.get('images', []):
                old_image_id = img_info['id']
                img_info['id'] = next_image_id
                image_id_mapping[old_image_id] = next_image_id
                all_images.append(img_info)
                next_image_id += 1
            
            # Process annotations, update image_id
            for anno in coco_data.get('annotations', []):
                old_image_id = anno['image_id']
                anno['id'] = next_anno_id
                anno['image_id'] = image_id_mapping.get(old_image_id, old_image_id)
                all_annotations.append(anno)
                next_anno_id += 1
            
            print(f"  ✓ Read {split} annotations: {len(coco_data.get('images', []))} images, {len(coco_data.get('annotations', []))} annotations")
    
    # Save merged COCO annotations
    if all_images:
        merged_coco = {
            'images': all_images,
            'annotations': all_annotations,
            'categories': all_categories if all_categories else []
        }
        
        output_anno_file = output_dir / "_annotations.coco.json"
        with open(output_anno_file, 'w', encoding='utf-8') as f:
            json.dump(merged_coco, f, indent=2, ensure_ascii=False)
        
        print(f"\n  ✅ Merged annotation file:")
        print(f"     - Total images: {len(all_images)}")
        print(f"     - Total annotations: {len(all_annotations)}")
        print(f"     - Categories: {len(all_categories) if all_categories else 0}")
        print(f"     - Output to: {output_anno_file}")
    
    print(f"\n✅ Labeled data processing completed → {output_dir}")
    print(f"   Total {total_images} images")


def process_unlabeled_data(
    unlabeled_base: Path,
    output_base: Path
):
    """
    Process unlabeled data (IX83) - tile into 16x16 grid and record metadata
    """
    print("\n" + "="*60)
    print("📸 Processing unlabeled data (IX83 full-field)")
    print("="*60)
    
    if not unlabeled_base.exists():
        print(f"⚠️  Unlabeled data directory not found: {unlabeled_base}")
        return
    
    # Process Adult MSC and Pediatric MSC
    for subdir_name, age_group in [("Adult MSC", "Adult"), ("Pediatric MSC", "Pediatric")]:
        input_dir = unlabeled_base / subdir_name
        
        if not input_dir.exists():
            print(f"\n⚠️  Skipping non-existent directory: {subdir_name}")
            continue
        
        output_dir = output_base / "unlabeled" / subdir_name.replace(" ", "_")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all images (including subdirectories), exclude discarded data
        all_images = (
            list(input_dir.rglob("*.jpg")) +
            list(input_dir.rglob("*.png")) +
            list(input_dir.rglob("*.tif")) +
            list(input_dir.rglob("*.tiff"))
        )
        
        # Filtering rules
        excluded_folders = ["AI照片過密", "其他"]  # Discarded folders
        excluded_patterns = ["Snapshot"]  # Exclude Snapshot preview images
        
        images = [
            img for img in all_images 
            if not any(excluded in str(img) for excluded in excluded_folders)
            and not any(pattern in img.name for pattern in excluded_patterns)
        ]
        
        excluded_count = len(all_images) - len(images)
        if excluded_count > 0:
            print(f"  ⚠️  {excluded_count} images excluded")
            print(f"     - Discarded folders: {', '.join(excluded_folders)}")
            print(f"     - Excluded file types: {', '.join(excluded_patterns)}")
        
        if not images:
            print(f"\n⚠️  No images found in {subdir_name}")
            continue
        
        print(f"\nProcessing {subdir_name}:")
        print(f"  Age group: {age_group}")
        print(f"  Image count: {len(images)}")
        print(f"  Tiling method: 16x16 grid (256 tiles per image)")
        
        all_tiles_info = []
        
        # Tile each image
        for img_path in images:
            try:
                tiles_info = tile_image_16x16(
                    img_path,
                    output_dir,
                    age_group
                )
                all_tiles_info.extend(tiles_info)
            except Exception as e:
                print(f"\n  ⚠️  Error processing {img_path.name}: {e}")
                continue
        
        # Save metadata
        metadata_file = output_dir / "tiles_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(all_tiles_info, f, indent=2, ensure_ascii=False)
        
        # Generate summary statistics
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
        
        print(f"\n  Summary:")
        print(f"    - Original images: {summary['total_original_images']}")
        print(f"    - Total tiles: {summary['total_tiles']}")
        print(f"    - Donors: {len(summary['donors'])} donors → {summary['donors']}")
        print(f"    - Passage range: NP{min(summary['passages']) if summary['passages'] else 'N/A'} - NP{max(summary['passages']) if summary['passages'] else 'N/A'}")
        print(f"  ✅ {subdir_name} completed → {output_dir}")
        print(f"     Metadata: {metadata_file.name}")
        print(f"     Summary: {summary_file.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Bone marrow stem cell image tiling tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Process all data
  python scripts/tile_images.py
  
  # Process only labeled data
  python scripts/tile_images.py --labeled-only
  
  # Process only unlabeled data
  python scripts/tile_images.py --unlabeled-only
  
  # Custom paths
  python scripts/tile_images.py --raw-data-dir "custom/path" --output-dir "output/path"
        """
    )
    
    parser.add_argument(
        "--raw-data-dir",
        type=str,
        default="raw_data/01 AI 分析細胞照片",
        help="Raw data directory"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory"
    )
    
    parser.add_argument(
        "--labeled-only",
        action="store_true",
        help="Process only labeled data"
    )
    
    parser.add_argument(
        "--unlabeled-only",
        action="store_true",
        help="Process only unlabeled data"
    )
    
    args = parser.parse_args()
    
    # Set paths
    raw_data_dir = Path(args.raw_data_dir)
    output_dir = Path(args.output_dir)
    
    roboflow_dir = raw_data_dir / "05 輔仁大學Roboflow"
    unlabeled_base = raw_data_dir / "02 IX83 全通量細胞照片"
    
    print("\n" + "="*60)
    print("  Bone marrow stem cell image tiling tool")
    print("="*60)
    print(f"\nSettings:")
    print(f"  Raw data: {raw_data_dir.absolute()}")
    print(f"  Output directory: {output_dir.absolute()}")
    print(f"  Mode: ", end="")
    if args.labeled_only:
        print("Process labeled data only")
    elif args.unlabeled_only:
        print("Process unlabeled data only")
    else:
        print("Process all data")
    
    # Process data
    if not args.unlabeled_only:
        process_labeled_data(roboflow_dir, output_dir)
    
    if not args.labeled_only:
        process_unlabeled_data(unlabeled_base, output_dir)
    
    print("\n" + "="*60)
    print("  ✅ All processing completed!")
    print("="*60)
    print(f"\nProcessed data located at: {output_dir.absolute()}")
    print("\nDirectory structure:")
    print("  data/")
    print("    ├── labeled/              # Labeled data (Roboflow merged)")
    print("    │   ├── *.jpg/png         (All labeled images)")
    print("    │   └── _annotations.coco.json  (Merged annotations)")
    print("    └── unlabeled/            # Unlabeled data (IX83)")
    print("        ├── Adult_MSC/")
    print("        │   ├── *_r00_c00.tif   (256 tiles per image)")
    print("        │   ├── tiles_metadata.json")
    print("        │   └── summary.json")
    print("        └── Pediatric_MSC/")
    print("            ├── *_r00_c00.tif")
    print("            ├── tiles_metadata.json")
    print("            └── summary.json")
    print("\nLabeled data:")
    print("  - Merged train/valid/test into single dataset")
    print("  - COCO format annotations, image_id renumbered")
    print("\nUnlabeled data metadata includes:")
    print("  - age_group: Adult/Pediatric")
    print("  - donor_id: A4, B10, A1, B135, etc.")
    print("  - passage_number: 2, 9, 17, etc.")
    print("  - grid_position: 16x16 grid position")
    print("  - bbox: coordinates in original image")


if __name__ == "__main__":
    main()
