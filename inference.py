'''
Inference for Composition-1k Dataset.

Run:
python inference.py \
    --config-dir path/to/config
    --checkpoint-dir path/to/checkpoint
    --inference-dir path/to/inference
    --data-dir path/to/data
'''
import os
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import functional as F
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import default_argument_parser
import cv2
import random
import numpy as np
from utils.visualization import colorize_label
import warnings
warnings.filterwarnings('ignore')
os.environ['MASTER_ADDR'] = '127.0.0.1'  # 这里用本机地址，分布式多机需改为主节点IP
os.environ['MASTER_PORT'] = str(random.randint(1024, 65535)) 
torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=1, rank=0)


#model and output
def matting_inference(
    config_dir='',
    checkpoint_dir='',
    inference_dir='',
):
    #initializing model
    cfg = LazyConfig.load(config_dir)
    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    model.eval()
    DetectionCheckpointer(model).load(checkpoint_dir)

    #initializing dataset
    dataloader = instantiate(cfg.dataloader.test)
    
    #inferencing
    os.makedirs(inference_dir, exist_ok=True)

    for data in tqdm(dataloader):
        H, W = data['hw'][0].item(), data['hw'][1].item()
        output = model(data)
        output = torch.argmax(output, dim=1)[0]
        output = cv2.resize(output.cpu().numpy().astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        output = colorize_label(output)
        cv2.imwrite(os.path.join(inference_dir, data['image_name'][0][:-4]+'.png') ,output)
        torch.cuda.empty_cache()

if __name__ == '__main__':
    #add argument we need:
    parser = default_argument_parser()
    parser.add_argument('--config-dir', type=str, required=True)
    parser.add_argument('--checkpoint-dir', type=str, required=True)
    parser.add_argument('--inference-dir', type=str, required=True)
    
    args = parser.parse_args()
    matting_inference(
        config_dir = args.config_dir,
        checkpoint_dir = args.checkpoint_dir,
        inference_dir = args.inference_dir,
    )