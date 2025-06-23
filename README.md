
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

| version | description                                              | val           |
|---------|----------------------------------------------------------|---------------|
| v0      | baseline                                                 | MIoU: 29.51   |
| v1      | 增加dice_loss                                           | MIoU: 26.16   |
| v2      | 冻结backbone                                            | MIoU: 32.95   |
| v3      | 冻结backbone，增加dice_loss                              | MIoU: 32.22   |
| v4      | 冻结backbone，增加金字塔池化模块作为neck，聚合backbone生成的四层特征图，集中语义信息 | **MIoU: 35.21** |
| v5      | 冻结backbone，以直接相加的方式聚合backbone生成的四层特征图，集中语义信息  | MIoU: 33.12   |
| v6      | 冻结backbone，增加金字塔池化模块作为neck，聚合backbone生成的十二层特征图，集中语义信息 | MIoU: 35.21   |








