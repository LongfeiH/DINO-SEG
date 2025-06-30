import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os

from detectron2.structures import ImageList



class DINOMattePromptDiverseV2(nn.Module):
    def __init__(self, neck, decoder, patch_size=14, emb_dim=384, select_list=[]):
        super(DINOMattePromptDiverseV2, self).__init__()
        self.backbone = torch.hub.load('dinov2', 'dinov2_vits14', source='local')
        self.embedding_size = emb_dim
        self.patch_size = patch_size
        self.select_list = select_list
        self.neck = neck
        self.decoder = decoder
    
    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, data):
        images, H, W = self.preprocess_inputs(data)
        h, w = int(images.shape[2] / self.patch_size), int(images.shape[3] / self.patch_size)

        features = self.backbone.forward_custom(images, select_list=self.select_list)
        x = self.neck(features, h, w)
        outputs = self.decoder(x, images)

        return outputs[:,:,:H,:W]
        




    def preprocess_inputs(self, data):
        """
        Normalize, pad and batch the input images.
        """
        images = data["image"].cuda()

        _, _, H, W = images.shape
        os=14
        if images.shape[-1]%os!=0 or images.shape[-2]%os!=0:
            new_H = (os-images.shape[-2]%os) + H
            new_W = (os-images.shape[-1]%os) + W
            new_images = torch.zeros((images.shape[0], images.shape[1], new_H, new_W)).to('cuda')
            new_images[:,:,:H,:W] = images[:,:,:,:]
            del images
            images = new_images
            
        return images, H, W