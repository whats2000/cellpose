"""
資料檢查腳本 - 檢查骨髓幹細胞影像資料的完整性與統計
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def print_header(text):
    """列印標題"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)


def print_section(text):
    """列印區段"""
    print(f"\n{text}")
    print("-"*60)


def check_labeled_data(base_path):
    """檢查已標記資料 (Roboflow)"""
    print_section("📋 已標記資料 (Roboflow COCO 格式)")
    
    labeled_path = base_path / "05 輔仁大學Roboflow"
    
    if not labeled_path.exists():
        print(f"❌ 找不到資料夾: {labeled_path}")
        return {}
    
    stats = {}
    
    for split in ["train", "valid", "test"]:
        split_dir = labeled_path / split
        if not split_dir.exists():
            print(f"⚠️  {split} 資料夾不存在")
            continue
        
        # 統計影像
        images = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png"))
        
        # 檢查標註檔
        anno_file = split_dir / "_annotations.coco.json"
        has_annotations = anno_file.exists()
        
        n_annotations = 0
        if has_annotations:
            try:
                with open(anno_file, 'r', encoding='utf-8') as f:
                    coco_data = json.load(f)
                    n_annotations = len(coco_data.get('annotations', []))
            except Exception as e:
                print(f"⚠️  無法讀取標註檔: {e}")
        
        print(f"  {split:6s}: {len(images):4d} 張影像, {n_annotations:5d} 個標註", 
              "✅" if has_annotations else "❌ 缺少標註檔")
        
        stats[split] = {
            'n_images': len(images),
            'n_annotations': n_annotations,
            'has_annotations': has_annotations
        }
    
    total_images = sum(s['n_images'] for s in stats.values())
    total_annotations = sum(s['n_annotations'] for s in stats.values())
    
    print(f"\n  總計: {total_images} 張影像, {total_annotations} 個標註")
    
    return stats


def check_unlabeled_data(base_path):
    """檢查未標記資料"""
    print_section("📦 未標記資料 (IX83 全通量細胞照片)")
    
    unlabeled_path = base_path / "02 IX83 全通量細胞照片"
    
    if not unlabeled_path.exists():
        print(f"❌ 找不到資料夾: {unlabeled_path}")
        return {}
    
    stats = {}
    
    for subdir_name in ["Adult MSC", "Pediatric MSC"]:
        subdir = unlabeled_path / subdir_name
        if not subdir.exists():
            print(f"⚠️  {subdir_name} 資料夾不存在")
            continue
        
        # 統計影像（遞迴搜尋）
        images = list(subdir.rglob("*.jpg")) + list(subdir.rglob("*.png")) + \
                 list(subdir.rglob("*.tif")) + list(subdir.rglob("*.tiff"))
        
        # 按日期資料夾分組
        date_folders = defaultdict(int)
        for img in images:
            # 取得最近的父資料夾名稱
            parent = img.parent.name
            date_folders[parent] += 1
        
        print(f"\n  {subdir_name}:")
        print(f"    總影像數: {len(images)}")
        print(f"    日期批次數: {len(date_folders)}")
        
        # 顯示前幾個批次
        sorted_folders = sorted(date_folders.items(), key=lambda x: x[1], reverse=True)
        print(f"    主要批次:")
        for folder, count in sorted_folders[:5]:
            print(f"      - {folder}: {count} 張")
        
        if len(sorted_folders) > 5:
            print(f"      ... 還有 {len(sorted_folders) - 5} 個批次")
        
        stats[subdir_name] = {
            'n_images': len(images),
            'n_batches': len(date_folders),
            'batches': dict(sorted_folders)
        }
    
    total_unlabeled = sum(s['n_images'] for s in stats.values())
    print(f"\n  總計: {total_unlabeled} 張未標記影像")
    
    return stats


def check_workspace():
    """檢查工作目錄"""
    print_section("🔧 工作目錄狀態")
    
    workspace = Path("active_learning_workspace")
    
    if not workspace.exists():
        print(f"  ℹ️  工作目錄尚未建立: {workspace}")
        print(f"     執行主動學習系統時會自動建立")
        return
    
    # 檢查子目錄
    subdirs = ["models", "annotations", "predictions", "logs"]
    for subdir in subdirs:
        subdir_path = workspace / subdir
        exists = subdir_path.exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {subdir}/")
    
    # 檢查迭代目錄
    iterations = sorted(workspace.glob("iteration_*"))
    if iterations:
        print(f"\n  已完成迭代: {len(iterations)}")
        for iter_dir in iterations[-3:]:  # 顯示最近3個
            print(f"    - {iter_dir.name}")


def check_environment():
    """檢查 Python 環境"""
    print_section("🐍 Python 環境檢查")
    
    # Python 版本
    python_version = sys.version.split()[0]
    print(f"  Python 版本: {python_version}")
    
    # 檢查套件
    required_packages = [
        "cellpose",
        "torch",
        "numpy",
        "cv2",
        "sklearn",
        "pandas",
        "matplotlib"
    ]
    
    print("\n  必要套件:")
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
            print(f"    ❌ {package:12s} (未安裝)")
    
    # GPU 檢查
    print("\n  GPU 狀態:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"    ✅ CUDA 可用")
            print(f"       裝置: {torch.cuda.get_device_name(0)}")
            print(f"       記憶體: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print(f"    ⚠️  CUDA 不可用 (將使用 CPU，訓練會較慢)")
    except Exception as e:
        print(f"    ❌ 無法檢查 GPU: {e}")


def estimate_resources():
    """估算資源需求"""
    print_section("📊 資源需求估算")
    
    print("  基於當前資料規模的估算:")
    print()
    print("  訓練資源:")
    print("    - GPU 記憶體: 建議 >= 8GB")
    print("    - RAM: 建議 >= 32GB")
    print("    - 儲存空間: 建議 >= 500GB")
    print()
    print("  時間估算 (10 輪迭代):")
    print("    - 每輪標註: ~10-15 小時 (100 張影像)")
    print("    - 每輪訓練: ~2-4 小時 (視 GPU 而定)")
    print("    - 總計約: 120-190 小時 (約 3-4 個月)")
    print()
    print("  人力需求:")
    print("    - AI 工程師: 1-2 人")
    print("    - 生物醫學專家: 1-2 人 (品質控制)")
    print("    - 標註人員: 2-3 人")


def generate_summary_report(labeled_stats, unlabeled_stats):
    """生成摘要報告"""
    print_header("📈 資料摘要報告")
    
    total_labeled = sum(s['n_images'] for s in labeled_stats.values())
    total_unlabeled = sum(s['n_images'] for s in unlabeled_stats.values())
    
    print(f"""
  已標記資料: {total_labeled:5d} 張影像
  未標記資料: {total_unlabeled:5d} 張影像
  
  標記/未標記比例: 1:{total_unlabeled/max(total_labeled, 1):.0f}
  
  建議:
  """)
    
    if total_labeled < 50:
        print("  ⚠️  已標記資料較少 (<50)，建議:")
        print("     1. 先手動標註更多資料 (至少 50-100 張)")
        print("     2. 或直接開始主動學習，但初始效能可能較低")
    elif total_labeled < 200:
        print("  ✅ 已標記資料適中，可以開始主動學習")
    else:
        print("  ✅ 已標記資料充足，適合訓練")
    
    if total_unlabeled > 10000:
        print(f"\n  ℹ️  未標記資料量龐大 ({total_unlabeled} 張)")
        print("     建議先對資料進行採樣或分批處理")


def save_stats_to_file(labeled_stats, unlabeled_stats):
    """儲存統計資料到檔案"""
    stats = {
        'labeled': labeled_stats,
        'unlabeled': unlabeled_stats,
        'timestamp': str(Path('.').__str__())
    }
    
    output_file = Path("data_statistics.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 統計資料已儲存至: {output_file}")


def main():
    """主程式"""
    print_header("骨髓幹細胞影像資料檢查工具")
    
    # 設定基礎路徑
    base_path = Path("raw_data/01 AI 分析細胞照片")
    
    if not base_path.exists():
        print(f"\n❌ 找不到資料根目錄: {base_path}")
        print(f"   請確認您在正確的專案目錄下執行此腳本")
        return
    
    # 執行各項檢查
    labeled_stats = check_labeled_data(base_path)
    unlabeled_stats = check_unlabeled_data(base_path)
    check_workspace()
    check_environment()
    estimate_resources()
    generate_summary_report(labeled_stats, unlabeled_stats)
    
    # 儲存統計
    try:
        save_stats_to_file(labeled_stats, unlabeled_stats)
    except Exception as e:
        print(f"⚠️  無法儲存統計資料: {e}")
    
    print("\n" + "="*60)
    print("  檢查完成！")
    print("="*60 + "\n")
    
    # 下一步建議
    print("📋 下一步建議:")
    print("  1. 如果環境檢查有缺少的套件，請先安裝")
    print("  2. 執行 scripts/tile_large_images.py 處理大型影像")
    print("  3. 執行 scripts/convert_coco_to_cellpose.py 轉換格式")
    print("  4. 執行 src/active_learning_framework.py 開始訓練\n")


if __name__ == "__main__":
    main()
