
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

| version | description                            | val           |
|---------|-------------------------------------|---------------|
| v0      | baseline                            | MIoU: 29.51   |
| v1      | +dice_loss (based on v0)            | MIoU: 26.16   |
| v2      | freeze backbone                     | MIoU: 32.95   |
| v3      | freeze backbone, +dice_loss          | MIoU: 32.22   |
| v4      | freeze + pyramid neck (4-layer fusion)| **MIoU: 35.21** |
| v5      | freeze + direct sum (4-layer fusion) | MIoU: 33.12   |
| v6      | freeze + pyramid neck (12-layer fusion)| MIoU: 35.21   |








