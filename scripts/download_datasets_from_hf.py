#!/usr/bin/env python3
"""
从 Hugging Face 下载 DermAgent 所需的皮肤病数据集到正确的位置
- ISIC2019 -> /root/DermAgent/data/isic2019/
- SCIN -> /root/DermAgent/data/scin/official_mirror/
- SD-198 -> /root/DermAgent/data/sd198/sd-198/
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download

# DermAgent 项目的数据目录
DATA_DIR = Path("/root/DermAgent/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_isic2019():
    """下载 ISIC 2019 数据集"""
    print("\n[1/3] 下载 ISIC 2019 数据集...")
    print("  目标路径: /root/DermAgent/data/isic2019/")
    print("  需要文件: ISIC_2019_Training_Metadata.csv, ISIC_2019_Training_Input/")

    target_dir = DATA_DIR / "isic2019"
    target_dir.mkdir(parents=True, exist_ok=True)

    # ISIC 2019 推荐的 Hugging Face 仓库
    repo_id = "marmal88/skin_cancer"

    try:
        print(f"  从 {repo_id} 下载...")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
        )
        print(f"  ✓ 成功下载 ISIC2019")

        # 验证关键文件
        metadata_file = target_dir / "ISIC_2019_Training_Metadata.csv"
        if metadata_file.exists():
            print(f"  ✓ 验证: {metadata_file} 存在")
        else:
            print(f"  ⚠ 警告: 未找到 ISIC_2019_Training_Metadata.csv")

    except Exception as e:
        print(f"  ✗ 下载失败: {e}")
        print("  请手动下载:")
        print("    - Kaggle: https://www.kaggle.com/c/isic-2019/data")
        print("    - ISIC Archive: https://challenge.isic-archive.com/data/#2019")
        print("    - 或使用: huggingface-cli download --repo-type dataset marmal88/skin_cancer --local-dir /root/DermAgent/data/isic2019")


def download_scin():
    """下载 SCIN 数据集"""
    print("\n[2/3] 下载 SCIN 数据集...")
    print("  目标路径: /root/DermAgent/data/scin/official_mirror/")
    print("  需要文件: scin_cases.csv, images/")

    target_dir = DATA_DIR / "scin" / "official_mirror"
    target_dir.mkdir(parents=True, exist_ok=True)

    # SCIN 推荐的 Hugging Face 仓库
    repo_id = "Nahrawy/SCIN"

    try:
        print(f"  从 {repo_id} 下载...")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
        )
        print(f"  ✓ 成功下载 SCIN")

        # 验证关键文件
        metadata_file = target_dir / "scin_cases.csv"
        if metadata_file.exists():
            print(f"  ✓ 验证: {metadata_file} 存在")
        else:
            print(f"  ⚠ 警告: 未找到 scin_cases.csv")

    except Exception as e:
        print(f"  ✗ 下载失败: {e}")
        print("  请手动下载:")
        print("    - GitHub: https://github.com/xiaoxuegao499/SCIN-dataset")
        print("    - 或使用: huggingface-cli download --repo-type dataset Nahrawy/SCIN --local-dir /root/DermAgent/data/scin/official_mirror")


def download_sd198():
    """下载 SD-198 数据集"""
    print("\n[3/3] 下载 SD-198 数据集...")
    print("  目标路径: /root/DermAgent/data/sd198/sd-198/")
    print("  需要文件: classes.txt, images.txt, image_class_labels.txt, images/")

    target_dir = DATA_DIR / "sd198" / "sd-198"
    target_dir.mkdir(parents=True, exist_ok=True)

    # SD-198 推荐的 Hugging Face 仓库
    repo_id = "zjysteven/SD-198"

    try:
        print(f"  从 {repo_id} 下载...")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
        )
        print(f"  ✓ 成功下载 SD-198")

        # 验证关键文件
        for filename in ["classes.txt", "images.txt", "image_class_labels.txt"]:
            file_path = target_dir / filename
            if file_path.exists():
                print(f"  ✓ 验证: {filename} 存在")
            else:
                print(f"  ⚠ 警告: 未找到 {filename}")

    except Exception as e:
        print(f"  ✗ 下载失败: {e}")
        print("  请手动下载:")
        print("    - 或使用: huggingface-cli download --repo-type dataset zjysteven/SD-198 --local-dir /root/DermAgent/data/sd198/sd-198")


def main():
    print("=== 从 Hugging Face 下载 DermAgent 数据集 ===")
    print(f"目标目录: {DATA_DIR}")
    print()

    download_isic2019()
    download_scin()
    download_sd198()

    print("\n=== 下载完成 ===")
    print("\n当前数据集状态:")
    os.system(f"ls -lh {DATA_DIR}")

    print("\n验证数据集结构:")
    print("\nISIC2019:")
    os.system(f"ls -lh {DATA_DIR}/isic2019/ 2>/dev/null | head -10")
    print("\nSCIN:")
    os.system(f"ls -lh {DATA_DIR}/scin/official_mirror/ 2>/dev/null | head -10")
    print("\nSD-198:")
    os.system(f"ls -lh {DATA_DIR}/sd198/sd-198/ 2>/dev/null | head -10")


if __name__ == "__main__":
    main()
