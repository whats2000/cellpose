"""
Data checking script - Check integrity and statistics of bone marrow stem cell image data
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def print_header(text):
    """Print header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)


def print_section(text):
    """Print section"""
    print(f"\n{text}")
    print("-"*60)


def check_labeled_data(base_path):
    """Check labeled data (Roboflow)"""
    print_section("📋 Labeled data (Roboflow COCO format)")
    
    labeled_path = base_path / "05 輔仁大學Roboflow"
    
    if not labeled_path.exists():
        print(f"❌ Directory not found: {labeled_path}")
        return {}
    
    stats = {}
    
    for split in ["train", "valid", "test"]:
        split_dir = labeled_path / split
        if not split_dir.exists():
            print(f"⚠️  {split} directory does not exist")
            continue
        
        # Count images
        images = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png"))
        
        # Check annotation file
        anno_file = split_dir / "_annotations.coco.json"
        has_annotations = anno_file.exists()
        
        n_annotations = 0
        if has_annotations:
            try:
                with open(anno_file, 'r', encoding='utf-8') as f:
                    coco_data = json.load(f)
                    n_annotations = len(coco_data.get('annotations', []))
            except Exception as e:
                print(f"⚠️  Unable to read annotation file: {e}")
        
        print(f"  {split:6s}: {len(images):4d} images, {n_annotations:5d} annotations", 
              "✅" if has_annotations else "❌ Missing annotation file")
        
        stats[split] = {
            'n_images': len(images),
            'n_annotations': n_annotations,
            'has_annotations': has_annotations
        }
    
    total_images = sum(s['n_images'] for s in stats.values())
    total_annotations = sum(s['n_annotations'] for s in stats.values())
    
    print(f"\n  Total: {total_images} images, {total_annotations} annotations")
    
    return stats


def check_unlabeled_data(base_path):
    """Check unlabeled data"""
    print_section("📦 Unlabeled data (IX83 full-field cell photos)")
    
    unlabeled_path = base_path / "02 IX83 全通量細胞照片"
    
    if not unlabeled_path.exists():
        print(f"❌ Directory not found: {unlabeled_path}")
        return {}
    
    stats = {}
    
    for subdir_name in ["Adult MSC", "Pediatric MSC"]:
        subdir = unlabeled_path / subdir_name
        if not subdir.exists():
            print(f"⚠️  {subdir_name} directory does not exist")
            continue
        
        # Count images (recursive search)
        images = list(subdir.rglob("*.jpg")) + list(subdir.rglob("*.png")) + \
                 list(subdir.rglob("*.tif")) + list(subdir.rglob("*.tiff"))
        
        # Group by date folders
        date_folders = defaultdict(int)
        for img in images:
            # Get the nearest parent folder name
            parent = img.parent.name
            date_folders[parent] += 1
        
        print(f"\n  {subdir_name}:")
        print(f"    Total images: {len(images)}")
        print(f"    Date batches: {len(date_folders)}")
        
        # Display top batches
        sorted_folders = sorted(date_folders.items(), key=lambda x: x[1], reverse=True)
        print(f"    Main batches:")
        for folder, count in sorted_folders[:5]:
            print(f"      - {folder}: {count} images")
        
        if len(sorted_folders) > 5:
            print(f"      ... and {len(sorted_folders) - 5} more batches")
        
        stats[subdir_name] = {
            'n_images': len(images),
            'n_batches': len(date_folders),
            'batches': dict(sorted_folders)
        }
    
    total_unlabeled = sum(s['n_images'] for s in stats.values())
    print(f"\n  Total: {total_unlabeled} unlabeled images")
    
    return stats


def check_workspace():
    """Check workspace directory"""
    print_section("🔧 Workspace status")
    
    workspace = Path("active_learning_workspace")
    
    if not workspace.exists():
        print(f"  ℹ️  Workspace directory not yet created: {workspace}")
        print(f"     Will be created automatically when running active learning system")
        return
    
    # Check subdirectories
    subdirs = ["models", "annotations", "predictions", "logs"]
    for subdir in subdirs:
        subdir_path = workspace / subdir
        exists = subdir_path.exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {subdir}/")
    
    # Check iteration directories
    iterations = sorted(workspace.glob("iteration_*"))
    if iterations:
        print(f"\n  Completed iterations: {len(iterations)}")
        for iter_dir in iterations[-3:]:  # Show last 3
            print(f"    - {iter_dir.name}")


def check_environment():
    """Check Python environment"""
    print_section("🐍 Python environment check")
    
    # Python version
    python_version = sys.version.split()[0]
    print(f"  Python version: {python_version}")
    
    # Check packages
    required_packages = [
        "cellpose",
        "torch",
        "numpy",
        "cv2",
        "sklearn",
        "pandas",
        "matplotlib"
    ]
    
    print("\n  Required packages:")
    for package in required_packages:
        try:
            if package == "cv2":
                import cv2
                version = getattr(cv2, '__version__', 'installed')
            elif package == "sklearn":
                import sklearn
                version = sklearn.__version__
            else:
                mod = __import__(package)
                version = getattr(mod, '__version__', 'unknown')
            
            print(f"    ✅ {package:12s} (v{version})")
        except ImportError:
            print(f"    ❌ {package:12s} (not installed)")
    
    # GPU check
    print("\n  GPU status:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"    ✅ CUDA available")
            print(f"       Device: {torch.cuda.get_device_name(0)}")
            print(f"       Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print(f"    ⚠️  CUDA not available (will use CPU, training will be slower)")
    except Exception as e:
        print(f"    ❌ Unable to check GPU: {e}")


def estimate_resources():
    """Estimate resource requirements"""
    print_section("📊 Resource requirement estimation")
    
    print("  Estimation based on current data scale:")
    print()
    print("  Training resources:")
    print("    - GPU memory: Recommended >= 8GB")
    print("    - RAM: Recommended >= 32GB")
    print("    - Storage space: Recommended >= 500GB")
    print()
    print("  Time estimation (10 rounds of iteration):")
    print("    - Annotation per round: ~10-15 hours (100 images)")
    print("    - Training per round: ~2-4 hours (depends on GPU)")
    print("    - Total approx: 120-190 hours (about 3-4 months)")
    print()
    print("  Personnel requirements:")
    print("    - AI engineers: 1-2 people")
    print("    - Biomedical experts: 1-2 people (quality control)")
    print("    - Annotators: 2-3 people")


def generate_summary_report(labeled_stats, unlabeled_stats):
    """Generate summary report"""
    print_header("📈 Data summary report")
    
    total_labeled = sum(s['n_images'] for s in labeled_stats.values())
    total_unlabeled = sum(s['n_images'] for s in unlabeled_stats.values())
    
    print(f"""
  Labeled data: {total_labeled:5d} images
  Unlabeled data: {total_unlabeled:5d} images
  
  Labeled/Unlabeled ratio: 1:{total_unlabeled/max(total_labeled, 1):.0f}
  
  Recommendations:
  """)
    
    if total_labeled < 50:
        print("  ⚠️  Labeled data is low (<50), suggestions:")
        print("     1. Manually annotate more data first (at least 50-100 images)")
        print("     2. Or start active learning directly, but initial performance may be lower")
    elif total_labeled < 200:
        print("  ✅ Labeled data is moderate, can start active learning")
    else:
        print("  ✅ Labeled data is sufficient for training")
    
    if total_unlabeled > 10000:
        print(f"\n  ℹ️  Unlabeled data volume is large ({total_unlabeled} images)")
        print("     Consider sampling or batch processing")


def save_stats_to_file(labeled_stats, unlabeled_stats):
    """Save statistics to file"""
    stats = {
        'labeled': labeled_stats,
        'unlabeled': unlabeled_stats,
        'timestamp': str(Path('.').__str__())
    }
    
    output_file = Path("data_statistics.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Statistics saved to: {output_file}")


def main():
    """Main program"""
    print_header("Bone marrow stem cell image data checking tool")
    
    # Set base path
    base_path = Path("raw_data/01 AI 分析細胞照片")
    
    if not base_path.exists():
        print(f"\n❌ Root data directory not found: {base_path}")
        print(f"   Please ensure you are running this script in the correct project directory")
        return
    
    # Execute checks
    labeled_stats = check_labeled_data(base_path)
    unlabeled_stats = check_unlabeled_data(base_path)
    check_workspace()
    check_environment()
    estimate_resources()
    generate_summary_report(labeled_stats, unlabeled_stats)
    
    # Save statistics
    try:
        save_stats_to_file(labeled_stats, unlabeled_stats)
    except Exception as e:
        print(f"⚠️  Unable to save statistics: {e}")
    
    print("\n" + "="*60)
    print("  Check completed!")
    print("="*60 + "\n")
    
    # Next steps suggestions
    print("📋 Next steps suggestions:")
    print("  1. If environment check shows missing packages, please install them first")
    print("  2. Run scripts/tile_large_images.py to process large images")
    print("  3. Run scripts/convert_coco_to_cellpose.py to convert format")
    print("  4. Run src/active_learning_framework.py to start training\n")


if __name__ == "__main__":
    main()
