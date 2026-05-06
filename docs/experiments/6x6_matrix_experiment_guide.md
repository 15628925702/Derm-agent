# 6x6 矩阵实验运行指南

## 概述

已创建支持四卡并行的 6x6 矩阵实验框架（6个模型 × 6个数据集，每个30 case bootstrap + 30 case对比）。

## 核心特性

1. **分层抽样**: bootstrap 和对比实验都使用标签均匀的分层抽样
2. **四卡并行**: 每个模型独立占用一个GPU，最多4个模型并行运行
3. **失败容错**: 单个实验失败不影响其他实验继续运行
4. **实时进度**: 通过 `summary.tsv` 和日志文件实时查看进度
5. **独立端口**: 每个模型使用独立端口，避免冲突

## 文件说明

### 1. 分层抽样脚本
**文件**: `scripts/bootstrap_stratified_30case.py`

生成标签均匀的分层抽样索引：

```bash
python scripts/bootstrap_stratified_30case.py \
  --split-id ham10000_balanced_v1 \
  --data-root /root/DermAgent/data \
  --split-name train \
  --n-samples 30 \
  --seed 42 \
  --output bootstrap_indices.json
```

### 2. 6x6 矩阵并行运行器
**文件**: `scripts/run_6x6_matrix_parallel.sh`

主实验脚本，支持：
- 6个模型: qwen, medgemma, skinvl, llama, hulumed, dermatollama
- 6个数据集: ham10000, isic2019, pad20, scin, sd198, xiangya
- 每个模型顺序跑完6个数据集后停止，释放GPU
- 最多4个模型并行（对应4个GPU）

### 3. 测试脚本
**文件**: `scripts/test_6x6_matrix.sh`

生成烟雾测试版本（1个模型 × 2个数据集 × 3 cases）

## 运行指令

### 完整 6x6 实验（推荐）

```bash
cd /root/DermAgent

# 运行完整 6x6 矩阵实验
bash scripts/run_6x6_matrix_parallel.sh
```

**预计时间**: 
- 每个 case 约 2-5 分钟
- 每个数据集 30 bootstrap + 30 compare = 60 cases
- 每个模型 6 个数据集 = 360 cases
- 6 个模型并行（4卡，分两批）= 约 12-30 小时

### 自定义配置

```bash
# 修改 bootstrap 和 compare 的 case 数量
export BOOTSTRAP_COUNT=50
export COMPARE_COUNT=50

# 指定输出目录
export OUTPUT_ROOT=/root/DermAgent/outputs/my_6x6_experiment

bash scripts/run_6x6_matrix_parallel.sh
```

### 烟雾测试（推荐先运行）

```bash
cd /root/DermAgent

# 生成并运行烟雾测试（1模型 × 2数据集 × 3 cases）
bash scripts/test_6x6_matrix.sh
# 然后运行生成的烟雾测试脚本
bash scripts/run_6x6_matrix_smoke.sh
```

## 监控进度

### 1. 实时日志
```bash
# 主运行日志
tail -f outputs/6x6_matrix_*/runner.log

# 特定模型+数据集的日志
tail -f outputs/6x6_matrix_*/logs/qwen_ham10000_bootstrap.log
tail -f outputs/6x6_matrix_*/logs/qwen_ham10000_compare.log
```

### 2. 进度摘要
```bash
# 查看实时进度表格
watch -n 10 'column -t -s $'"'"'\t'"'"' outputs/6x6_matrix_*/summary.tsv'

# 或直接查看
cat outputs/6x6_matrix_*/summary.tsv
```

**summary.tsv 格式**:
```
model       dataset    status  bootstrap_status  compare_status  start_time            end_time              duration_sec  log_file
qwen        ham10000   OK      OK                OK              2026-05-05T03:00:00Z  2026-05-05T04:30:00Z  5400         logs/qwen_ham10000_*.log
medgemma    isic2019   FAILED  OK                FAILED          2026-05-05T03:00:00Z  2026-05-05T04:00:00Z  3600         logs/medgemma_isic2019_*.log
```

### 3. GPU 使用情况
```bash
watch -n 5 nvidia-smi
```

## 输出结构

```
outputs/6x6_matrix_YYYYMMDD_HHMMSS/
├── runner.log                          # 主运行日志
├── summary.tsv                         # 进度摘要表格
├── logs/                               # 详细日志
│   ├── qwen_server.log
│   ├── qwen_ham10000_bootstrap.log
│   ├── qwen_ham10000_compare.log
│   └── ...
├── qwen/                               # 每个模型的输出
│   ├── ham10000/
│   │   ├── bootstrap/                  # bootstrap 运行结果
│   │   ├── compare/                    # 对比实验结果
│   │   ├── policy/                     # 学习到的策略
│   │   ├── split_states/               # 分割状态
│   │   ├── ham10000_split.json         # 数据集分割定义
│   │   └── bootstrap_indices.json      # 分层抽样索引
│   ├── isic2019/
│   └── ...
├── medgemma/
└── ...
```

## 故障排查

### 模型启动失败
```bash
# 检查模型服务器日志
cat outputs/6x6_matrix_*/logs/qwen_server.log

# 手动测试模型启动
bash scripts/start_qwen_server.sh /root/DermAgent
curl http://127.0.0.1:8000/v1/models
```

### 端口冲突
```bash
# 检查端口占用
ss -ltnp | grep -E '8000|8010|8011|8012|8013|8014'

# 停止所有模型服务
bash scripts/switch_model_server.sh stop
```

### GPU 内存不足
```bash
# 检查 GPU 使用
nvidia-smi

# 减少并行模型数量（修改脚本中的批次大小）
# 或减少每个实验的 case 数量
export BOOTSTRAP_COUNT=15
export COMPARE_COUNT=15
```

### 单个实验失败
实验失败不会中断整体运行，可以：
1. 查看 `summary.tsv` 找到失败的模型+数据集组合
2. 查看对应的日志文件定位问题
3. 修复后单独重跑该组合（手动调用 bootstrap 和 compare 脚本）

## 模型-GPU 映射

```
GPU 0: qwen (port 8000), hulumed (port 8013)
GPU 1: medgemma (port 8010), dermatollama (port 8014)
GPU 2: skinvl (port 8011)
GPU 3: llama (port 8012)
```

第一批并行: qwen, medgemma, skinvl, llama (GPU 0-3)
第二批并行: hulumed, dermatollama (GPU 0-1)

## 数据集配置

| 数据集 | Split ID | 策略 | 数据路径 |
|--------|----------|------|----------|
| ham10000 | ham10000_balanced_v1 | stratified_by_dx_group | data/ham10000 |
| isic2019 | isic2019_contiguous_v1 | contiguous_by_metadata_index | data/isic2019 |
| pad20 | pad_ufes_20_contiguous_v1 | contiguous_by_metadata_index | data/pad_ufes_20 |
| scin | scin_contiguous_v1 | contiguous_by_metadata_index | data/scin |
| sd198 | sd198_balanced_v1 | stratified_by_sd198_label | data/sd198 |
| xiangya | xiangya_sft_grouped_v1 | stratified_by_xiangya_grouped_label | data/sft数据 |

## 注意事项

1. **运行前确保**:
   - 所有数据集已下载到 `data/` 目录
   - 所有模型启动脚本可用
   - 有足够的磁盘空间（每个实验约 1-5GB）

2. **中断恢复**:
   - 当前版本不支持断点续传
   - 如需中断，按 Ctrl+C，脚本会尝试优雅停止
   - 重新运行会创建新的输出目录

3. **资源占用**:
   - 4个GPU会同时运行
   - 每个模型占用 10-40GB GPU 内存
   - 确保系统内存充足（建议 64GB+）

4. **分层抽样**:
   - 对于已经 balanced 的数据集（ham10000_balanced_v1, sd198_balanced_v1），直接随机抽样
   - 对于 contiguous 数据集，会按标签分组后分层抽样
   - 保证每个标签都有代表性样本
