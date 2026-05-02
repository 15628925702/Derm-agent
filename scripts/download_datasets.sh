#!/bin/bash
# 从 Hugging Face 下载 DermAgent 所需的三个数据集
# ISIC2019, SCIN, SD-198

set -e

DATA_DIR="/root/DermAgent/data"
mkdir -p "$DATA_DIR"

echo "=== 从 Hugging Face 下载 DermAgent 数据集 ==="
echo "目标目录: $DATA_DIR"
echo ""

# ============================================
# 1. ISIC 2019 数据集
# ============================================
echo "[1/3] 下载 ISIC 2019 数据集..."
echo "  目标路径: $DATA_DIR/isic2019/"
echo "  需要文件: ISIC_2019_Training_Metadata.csv, ISIC_2019_Training_Input/"

mkdir -p "$DATA_DIR/isic2019"

if command -v huggingface-cli &> /dev/null; then
    echo "  使用 huggingface-cli 下载..."
    huggingface-cli download --repo-type dataset marmal88/skin_cancer \
        --local-dir "$DATA_DIR/isic2019" || {
        echo "  ✗ 下载失败"
        echo "  备选方案:"
        echo "    - Kaggle: https://www.kaggle.com/c/isic-2019/data"
        echo "    - ISIC Archive: https://challenge.isic-archive.com/data/#2019"
    }

    # 验证
    if [ -f "$DATA_DIR/isic2019/ISIC_2019_Training_Metadata.csv" ]; then
        echo "  ✓ 验证: ISIC_2019_Training_Metadata.csv 存在"
    else
        echo "  ⚠ 警告: 未找到 ISIC_2019_Training_Metadata.csv"
    fi
else
    echo "  ⚠ 未安装 huggingface-cli"
    echo "  请运行: pip install -U huggingface_hub"
    echo "  或使用 Python 脚本: python3 /root/DermAgent/scripts/download_datasets_from_hf.py"
fi

# ============================================
# 2. SCIN 数据集
# ============================================
echo ""
echo "[2/3] 下载 SCIN 数据集..."
echo "  目标路径: $DATA_DIR/scin/official_mirror/"
echo "  需要文件: scin_cases.csv, images/"

mkdir -p "$DATA_DIR/scin/official_mirror"

if command -v huggingface-cli &> /dev/null; then
    echo "  使用 huggingface-cli 下载..."
    huggingface-cli download --repo-type dataset Nahrawy/SCIN \
        --local-dir "$DATA_DIR/scin/official_mirror" || {
        echo "  ✗ 下载失败"
        echo "  备选方案:"
        echo "    - GitHub: https://github.com/xiaoxuegao499/SCIN-dataset"
    }

    # 验证
    if [ -f "$DATA_DIR/scin/official_mirror/scin_cases.csv" ]; then
        echo "  ✓ 验证: scin_cases.csv 存在"
    else
        echo "  ⚠ 警告: 未找到 scin_cases.csv"
    fi
fi

# ============================================
# 3. SD-198 数据集
# ============================================
echo ""
echo "[3/3] 下载 SD-198 数据集..."
echo "  目标路径: $DATA_DIR/sd198/sd-198/"
echo "  需要文件: classes.txt, images.txt, image_class_labels.txt, images/"

mkdir -p "$DATA_DIR/sd198/sd-198"

if command -v huggingface-cli &> /dev/null; then
    echo "  使用 huggingface-cli 下载..."
    huggingface-cli download --repo-type dataset zjysteven/SD-198 \
        --local-dir "$DATA_DIR/sd198/sd-198" || {
        echo "  ✗ 下载失败"
        echo "  备选方案: 联系论文作者获取数据集"
    }

    # 验证
    for file in "classes.txt" "images.txt" "image_class_labels.txt"; do
        if [ -f "$DATA_DIR/sd198/sd-198/$file" ]; then
            echo "  ✓ 验证: $file 存在"
        else
            echo "  ⚠ 警告: 未找到 $file"
        fi
    done
fi

# ============================================
# 总结
# ============================================
echo ""
echo "=== 下载完成 ==="
echo ""
echo "当前数据集状态:"
ls -lh "$DATA_DIR"

echo ""
echo "验证数据集结构:"
echo ""
echo "ISIC2019:"
ls -lh "$DATA_DIR/isic2019/" 2>/dev/null | head -10 || echo "  目录为空或不存在"

echo ""
echo "SCIN:"
ls -lh "$DATA_DIR/scin/official_mirror/" 2>/dev/null | head -10 || echo "  目录为空或不存在"

echo ""
echo "SD-198:"
ls -lh "$DATA_DIR/sd198/sd-198/" 2>/dev/null | head -10 || echo "  目录为空或不存在"
