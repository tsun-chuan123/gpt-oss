#!/usr/bin/env python3
"""
檢查 LLaVA 1.5 微調資料的完整性
"""

import json
import os
from collections import defaultdict
from pathlib import Path

def check_finetune_data():
    data_dir = Path("/workspace/playground/data")
    json_file = data_dir / "llava_v1_5_mix665k.json"
    
    print("=" * 60)
    print("LLaVA 1.5 微調資料完整性檢查")
    print("=" * 60)
    
    # 1. 檢查 JSON 檔案
    print("\n[1] 檢查訓練註解檔案...")
    if not json_file.exists():
        print(f"  ✗ 找不到 {json_file}")
        print(f"  → 請先下載: wget https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_v1_5_mix665k.json")
        return False
    
    print(f"  ✓ 找到 {json_file.name}")
    print(f"    大小: {json_file.stat().st_size / 1024 / 1024:.1f} MB")
    
    # 2. 載入並分析資料
    print("\n[2] 載入並分析資料...")
    try:
        with open(json_file) as f:
            data = json.load(f)
        print(f"  ✓ 總樣本數: {len(data):,}")
    except Exception as e:
        print(f"  ✗ 載入失敗: {e}")
        return False
    
    # 3. 統計各資料集的圖片數量
    print("\n[3] 統計各資料集分佈...")
    dataset_counts = defaultdict(int)
    missing_images = []
    text_only_count = 0
    
    for item in data:
        # 有些樣本是純文本對話，沒有圖片
        if 'image' not in item:
            text_only_count += 1
            continue
            
        image_path = item['image']
        dataset = image_path.split('/')[0]
        dataset_counts[dataset] += 1
        
        # 檢查圖片是否存在（只檢查前100個）
        if len(missing_images) < 100:
            full_path = data_dir / image_path
            if not full_path.exists():
                missing_images.append(image_path)
    
    print(f"  視覺-語言樣本: {sum(dataset_counts.values()):,}")
    print(f"  純文本樣本: {text_only_count:,}")
    print(f"\n  資料集分佈:")
    for dataset, count in sorted(dataset_counts.items()):
        print(f"    {dataset:15s}: {count:6,} 樣本")
    
    # 4. 檢查目錄結構
    print("\n[4] 檢查目錄結構...")
    required_dirs = {
        'coco/train2017': 'COCO train2017 圖片',
        'gqa/images': 'GQA 圖片',
        'ocr_vqa/images': 'OCR-VQA 圖片',
        'textvqa/train_images': 'TextVQA 圖片',
        'vg/VG_100K': 'Visual Genome Part 1',
        'vg/VG_100K_2': 'Visual Genome Part 2'
    }
    
    all_exists = True
    for dir_path, description in required_dirs.items():
        full_path = data_dir / dir_path
        if full_path.exists():
            # 計算檔案數量
            num_files = len(list(full_path.rglob('*.jpg'))) + len(list(full_path.rglob('*.png')))
            print(f"  ✓ {description:25s}: {num_files:6,} 個檔案")
        else:
            print(f"  ✗ {description:25s}: 目錄不存在")
            all_exists = False
    
    # 5. 檢查缺失的圖片
    if missing_images:
        print(f"\n[5] 缺失圖片檢查 (抽樣檢查前100個樣本)")
        print(f"  ⚠️  發現 {len(missing_images)} 個缺失的圖片")
        print(f"  範例:")
        for img in missing_images[:5]:
            print(f"    - {img}")
        if len(missing_images) > 5:
            print(f"    ... 還有 {len(missing_images) - 5} 個")
    else:
        print(f"\n[5] 圖片完整性檢查")
        print(f"  ✓ 抽樣檢查通過 (前100個樣本)")
    
    # 6. 磁碟空間檢查
    print("\n[6] 磁碟空間使用")
    try:
        import subprocess
        result = subprocess.run(['du', '-sh', str(data_dir)], 
                              capture_output=True, text=True)
        print(f"  總使用空間: {result.stdout.split()[0]}")
    except:
        pass
    
    # 總結
    print("\n" + "=" * 60)
    print("檢查總結")
    print("=" * 60)
    
    if all_exists and not missing_images:
        print("\n✓ 所有資料準備完成！可以開始微調訓練。")
        print("\n下一步:")
        print("  1. 確保預訓練模型已完成: ./checkpoints/llava-gptoss-20b-pretrain")
        print("  2. 賦予執行權限: chmod +x scripts/v1_5/finetune_gptoss.sh")
        print("  3. 開始微調訓練: ./scripts/v1_5/finetune_gptoss.sh")
        return True
    else:
        print("\n⚠️  資料不完整，請完成以下步驟:")
        if not all_exists:
            print("\n缺少的資料集目錄:")
            for dir_path, description in required_dirs.items():
                full_path = data_dir / dir_path
                if not full_path.exists():
                    print(f"  - {description}")
            print("\n請執行: ./scripts/v1_5/download_finetune_data.sh")
        
        if missing_images:
            print("\n有圖片檔案缺失，請確保:")
            print("  - 所有資料集都已完整下載")
            print("  - 目錄結構正確")
            print("  - 圖片檔案格式正確 (jpg/png)")
        
        return False

if __name__ == "__main__":
    check_finetune_data()
