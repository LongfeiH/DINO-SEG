import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from sklearn.metrics import adjusted_rand_score

class SegmentationCriterion(nn.Module):
    def __init__(self,losses):
        super(SegmentationCriterion, self).__init__()
        self.losses = losses

    def loss_gradient_penalty(self, sample_map ,preds, targets):
        preds = preds['phas']
        targets = targets['phas']

        #sample_map for unknown area
        scale = sample_map.shape[0]*200704/torch.sum(sample_map)

        #gradient in x
        sobel_x_kernel = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]).type(dtype=preds.type())
        delta_pred_x = F.conv2d(preds, weight=sobel_x_kernel, padding=1)
        delta_gt_x = F.conv2d(targets, weight=sobel_x_kernel, padding=1)

        #gradient in y 
        sobel_y_kernel = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]]).type(dtype=preds.type())
        delta_pred_y = F.conv2d(preds, weight=sobel_y_kernel, padding=1)
        delta_gt_y = F.conv2d(targets, weight=sobel_y_kernel, padding=1)

        #loss
        loss = (F.l1_loss(delta_pred_x*sample_map, delta_gt_x*sample_map)* scale + \
            F.l1_loss(delta_pred_y*sample_map, delta_gt_y*sample_map)* scale + \
            0.01 * torch.mean(torch.abs(delta_pred_x*sample_map))* scale +  \
            0.01 * torch.mean(torch.abs(delta_pred_y*sample_map))* scale)

        return dict(loss_gradient_penalty=loss)

    def loss_pha_laplacian(self, preds, targets):
        assert 'phas' in preds and 'phas' in targets
        loss = laplacian_loss(preds['phas'], targets['phas'])

        return dict(loss_pha_laplacian=loss)

    def l1_loss(self, preds, targets):
    
        loss = F.l1_loss(preds, targets)
        
        return dict(l1_loss=loss)

    def sim_loss(self, preds, targets):
        
        sim = preds['sim']
        phas = targets['phas']
        phas = F.interpolate(phas, scale_factor=1/14)
        phas = (phas>0).float()
        numerator = (sim * phas).sum()
        denominator = (sim * phas).sum() + ( sim * (1 - phas)).sum()
        # denominator = (sim * pred + (1 - sim) * pred).sum()
        loss = -torch.log(numerator / denominator)
        return dict(sim_loss=loss*0.05)

    def trans_loss(self, sample_map, preds, targets):
        pred_trimap = preds['trans']
        # trans_map = sample_map.bool() & (targets['phas']<0.9)  & (targets['phas']>0)
        trans_map = targets['phas']>0

        sample_map = F.interpolate(trans_map.float(), size=(pred_trimap.shape[-2], pred_trimap.shape[-1])).float()
        loss = nn.BCEWithLogitsLoss()(pred_trimap,sample_map)
        return dict(trans_loss=loss)

    def trans_loss_seg(self, sample_map, preds, targets):
        pred_trimap = preds['trans']
        trans_map = sample_map.bool() 
        sample_map = F.interpolate(trans_map.float(), size=(pred_trimap.shape[-2], pred_trimap.shape[-1])).float()
        loss = nn.BCEWithLogitsLoss()(pred_trimap,sample_map)
        return dict(trans_loss=loss)

    def ce_loss(self, pred, targets):
        loss = F.cross_entropy(pred, targets.squeeze(1).long())
        return dict(ce_loss=loss)
    
    
    def dice_loss(self, pred, targets, num_classes=151, ignore_index=None, eps=1e-6):
        """
        Compute Dice Loss for multi-class segmentation.
        
        Args:
            pred (Tensor): [B, C, H, W], raw logits from the model.
            target (Tensor): [B, H, W], ground truth labels with values in [0, num_classes-1].
            num_classes (int): number of segmentation classes (151 for ADE20K).
            ignore_index (int or None): optional class index to ignore in loss (e.g., 0 for background).
            eps (float): small constant to avoid division by zero.
            
        Returns:
            Dice loss (scalar)
        """
        pred = F.softmax(pred, dim=1)  # [B, C, H, W]
        targets = targets.squeeze(1).long()
        target_onehot = F.one_hot(targets, num_classes=num_classes)  # [B, H, W, C]
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()   # [B, C, H, W]

        # Mask ignored class if specified
        if ignore_index is not None:
            valid_mask = (targets != ignore_index).unsqueeze(1)  # [B, 1, H, W]
            pred = pred * valid_mask
            target_onehot = target_onehot * valid_mask

        # Calculate Dice
        intersection = (pred * target_onehot).sum(dim=(0, 2, 3))
        union = pred.sum(dim=(0, 2, 3)) + target_onehot.sum(dim=(0, 2, 3))

        dice = (2. * intersection + eps) / (union + eps)
        loss = 1 - dice

        # Optionally exclude ignored class from final loss
        if ignore_index is not None:
            loss[ignore_index] = 0
            loss = loss.sum() / (num_classes - 1)
        else:
            loss = loss.mean()
        
        return dict(dice_loss=loss)

    
    def forward(self, preds, targets):
        losses = dict()
        for k in self.losses:
            losses.update(getattr(self, k)(preds, targets))
        return losses


#-----------------Laplacian Loss-------------------------#
def laplacian_loss(pred, true, max_levels=5):
    kernel = gauss_kernel(device=pred.device, dtype=pred.dtype)
    pred_pyramid = laplacian_pyramid(pred, kernel, max_levels)
    true_pyramid = laplacian_pyramid(true, kernel, max_levels)
    loss = 0
    for level in range(max_levels):
        loss += (2 ** level) * F.l1_loss(pred_pyramid[level], true_pyramid[level])
    return loss / max_levels

def laplacian_pyramid(img, kernel, max_levels):
    current = img
    pyramid = []
    for _ in range(max_levels):
        current = crop_to_even_size(current)
        down = downsample(current, kernel)
        up = upsample(down, kernel)
        diff = current - up
        pyramid.append(diff)
        current = down
    return pyramid

def gauss_kernel(device='cpu', dtype=torch.float32):
    kernel = torch.tensor([[1,  4,  6,  4, 1],
                        [4, 16, 24, 16, 4],
                        [6, 24, 36, 24, 6],
                        [4, 16, 24, 16, 4],
                        [1,  4,  6,  4, 1]], device=device, dtype=dtype)
    kernel /= 256
    kernel = kernel[None, None, :, :]
    return kernel

def gauss_convolution(img, kernel):
    B, C, H, W = img.shape
    img = img.reshape(B * C, 1, H, W)
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')
    img = F.conv2d(img, kernel)
    img = img.reshape(B, C, H, W)
    return img

def downsample(img, kernel):
    img = gauss_convolution(img, kernel)
    img = img[:, :, ::2, ::2]
    return img

def upsample(img, kernel):
    B, C, H, W = img.shape
    out = torch.zeros((B, C, H * 2, W * 2), device=img.device, dtype=img.dtype)
    out[:, :, ::2, ::2] = img * 4
    out = gauss_convolution(out, kernel)
    return out

def crop_to_even_size(img):
    H, W = img.shape[2:]
    H = H - H % 2
    W = W - W % 2
    return img[:, :, :H, :W]