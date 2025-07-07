from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from detectron2.config import LazyCall as L
from torch.utils.data.distributed import DistributedSampler
from data import GetData, DataGenerator
from detectron2 import model_zoo
from detectron2.solver import WarmupParamScheduler
from fvcore.common.param_scheduler import MultiStepParamScheduler
from modeling import DINOMattePromptDiverseV2, Detail_Capture_DINO_V2, MultiLayerPPMFusion, SE_SegHead, CBAM_SegHead
from easydict import EasyDict
from evaluation import Evaluator

opts = EasyDict(
    batch_size=12,
    num_workers=12,
    lr=1e-4,
    num_gpu = 1,
    epoches = 50,
    data_num=20000,
    crop_size = (512, 512),
    output_dir="./output/SEG_9",
    init_checkpoint=None,
    losses = ["ce_loss"],
    frozen = True,
)
opts.max_iter = int(opts.data_num / opts.batch_size / opts.num_gpu * opts.epoches)
opts.val_step = int(opts.max_iter / 10)

# Model
model = L(DINOMattePromptDiverseV2)(
    patch_size=14,
    emb_dim=384,
    select_list=[2, 5, 8, 11],
    neck = MultiLayerPPMFusion(num_layers=4, in_dim=384),
    decoder=L(Detail_Capture_DINO_V2)(),
)

# Dataloader
train_dataset = DataGenerator(
    data=GetData(
        img_dir='/mnt/e/Semantic Segmentation/ADEChallengeData2016/images/training',
        label_dir='/mnt/e/Semantic Segmentation/ADEChallengeData2016/annotations/training'
    ),
    crop_size = opts.crop_size,
    phase = 'train'
)

test_dataset = DataGenerator(
    data=GetData(
        img_dir='/mnt/e/Semantic Segmentation/ADEChallengeData2016/images/validation',
        label_dir='/mnt/e/Semantic Segmentation/ADEChallengeData2016/annotations/validation'
    ),
    crop_size = opts.crop_size,
    phase = 'test'
)

dataloader = OmegaConf.create()
dataloader.train = L(DataLoader)(
    dataset=train_dataset,
    batch_size=opts.batch_size,
    shuffle=False,
    num_workers=opts.num_workers,
    pin_memory=True,
    sampler=L(DistributedSampler)(
        dataset=train_dataset,
    ),
    drop_last=True
)

dataloader.test = L(DataLoader)(
    dataset=test_dataset,
    batch_size = 1,
    shuffle=False,
    pin_memory=True,
)

dataloader.evaluator = L(Evaluator)()

# Training Setting
train = EasyDict(
    output_dir=opts.output_dir,
    init_checkpoint=opts.init_checkpoint,
    max_iter=opts.max_iter,
    amp=dict(enabled=True),  # options for Automatic Mixed Precision
    ddp=dict(
        broadcast_buffers=True,
        find_unused_parameters=False,
        fp16_compression=True,
    ),
    checkpointer=dict(period=opts.val_step, max_to_keep=100),  # options for PeriodicCheckpointer
    eval_period=opts.val_step,
    log_period=10,
    device="cuda"
)

# Optimizer
optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.lr=opts.lr

# LR
lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.05],
        milestones=[int(opts.max_iter * 0.3), int(opts.max_iter * 0.6)],
        num_updates=opts.max_iter,
    ),
    warmup_length=500 / opts.max_iter,
    warmup_factor=0.001,
)