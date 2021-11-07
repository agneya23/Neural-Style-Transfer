import os
import numpy as np 
import torch 
import torchvision

imsize = 512

class StyleTransfer(torch.nn.Module):
    
    def __init__(self, style, content, pastiche, device):
        super().__init__()
        self.style = style.to(device)
        self.content = content.to(device)
        self.pastiche = torch.nn.Parameter(pastiche.data)
        self.pastiche.to(device)
        self.pastiche.requires_grad_(True)
        
        self.content_layers = ['conv4', 'conv7']
        self.style_layers = ['conv2', 'conv4', 'conv7', 'conv10']
        self.alpha = 1
        self.beta = 850000
        
        self.net = torchvision.models.vgg19(pretrained=True).to(device)
        for param in self.net.parameters():
            param.requires_grad = False
        
        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD([self.pastiche], lr=0.06, momentum=0.9)
