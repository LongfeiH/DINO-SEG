
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


