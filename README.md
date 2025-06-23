
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

### Backbone: dinov2

| version | description                            | pretrained | bsz | lr       | epoches | size     | data   | loss              | val           |
|---------|----------------------------------------|------------|-----|----------|---------|----------|--------|-------------------|---------------|
| v0      | baseline                               | dinov2_s   | 12  | 1.00E-04 | 100     | 512x512  | ADE20K | ce_loss           | MIoU: 29.51   |
| v1      | +dice_loss (based on v0)               | v0         | 12  | 1.00E-05 | 100     | 512x512  | ADE20K | ce_loss, dice_loss| MIoU: 26.16   |
| v2      | freeze backbone                        | dinov2_s   | 12  | 1.00E-04 | 100     | 512x512  | ADE20K | ce_loss           | MIoU: 32.95   |
| v3      | freeze backbone, +dice_loss            | dinov2_s   | 12  | 1.00E-04 | 100     | 512x512  | ADE20K | ce_loss, dice_loss| MIoU: 32.22   |
| v4      | freeze + pyramid neck (4-layer fusion) | dinov2_s   | 12  | 1.00E-04 | 100     | 512x512  | ADE20K | ce_loss           | **MIoU: 35.21** |
| v5      | freeze + direct sum (4-layer fusion)   | dinov2_s   | 12  | 1.00E-04 | 100     | 512x512  | ADE20K | ce_loss           | MIoU: 33.12   |
| v6      | freeze + pyramid neck (12-layer fusion)| dinov2_s   | 12  | 1.00E-04 | 100     | 512x512  | ADE20K | ce_loss           | MIoU: 35.21   |







