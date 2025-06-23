
## ⚙️ Setup

### Install Environment via Anaconda (Recommended)
```bash
conda create -n dinoseg python=3.10
conda activate dinoseg
pip install -r requirements.txt
cd detectron2
pip install -e .
```

## 💫 Traning

```bash
  sh script/train.sh
```

## 💫 Test

```bash
  sh script/test.sh
```

## 💫 Inference

```bash
  sh script/infer.sh
```

## 实验记录

## Backbone: dinov2

| version | description | pretrained | bsz | lr       | epoches | size     | data   | loss              | val           |
|---------|-------------|------------|-----|----------|---------|----------|--------|-------------------|---------------|
| v0      | baseline    | dinov2_s   | 12  | 1.00E-04 | 100     | 512x512  | ADE20K | ce_loss           | MIoU: 29.51   |
| v1      | 增加 dice_loss | v0         | 12  | 1.00E-05 | 100     | 512x512  | ADE20K | ce_loss, dice_loss| MIoU: 26.16   |
| v2      | 冻结 backbone | dinov2_s   | 12  | 1.00E-04 | 100     | 512x512  | ADE20K | ce_loss           | MIoU: 32.95   |
| v3      | 冻结 backbone, 增加 dice_loss | dinov2_s | 12  | 1.00E-04 | 100 | 512x512 | ADE20K | ce_loss, dice_loss| MIoU: 32.22   |
| v4      | 冻结 backbone，增加金字塔池化模块作为 neck，聚合 backbone 生成的**四层特征图**，集中语义信息 | dinov2_s | 12 | 1.00E-04 | 100 | 512x512 | ADE20K | ce_loss | **MIoU: 35.21** |
| v5      | 冻结 backbone，以直接相加的方式聚合 backbone 生成的**四层特征图**，集中语义信息 | dinov2_s | 12 | 1.00E-04 | 100 | 512x512 | ADE20K | ce_loss | MIoU: 33.12 |
| v6      | 冻结 backbone，增加金字塔池化模块作为 neck，聚合 backbone 生成的**十二层特征图**，集中语义信息 | dinov2_s | 12 | 1.00E-04 | 100 | 512x512 | ADE20K | ce_loss | MIoU: 35.21 |






