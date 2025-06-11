from detectron2.engine import AMPTrainer
import torch
import time
from criterion import SegmentationCriterion

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

class MattingTrainer(AMPTrainer):
    def __init__(self, model, data_loader, optimizer, grad_scaler=None, opts=None):
        super().__init__(model, data_loader, optimizer, grad_scaler=None)
        self.opts = opts
        self.data_loader_iter = iter(cycle(self.data_loader))
        self.criterion = SegmentationCriterion(opts.losses)

    def run_step(self):
        """
        Implement the AMP training logic.
        """
        assert self.model.training, "[AMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        #matting pass
        start = time.perf_counter()        
        data = next(self.data_loader_iter)
        data_time = time.perf_counter() - start
        label = data['label'].cuda()

        with autocast():
            output = self.model(data)
            loss_dict = self.criterion(output, label)
            if isinstance(loss_dict, torch.Tensor):
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        self.optimizer.zero_grad()
        self.grad_scaler.scale(losses).backward()

        self._write_metrics(loss_dict, data_time)

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()