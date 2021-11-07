import os
import numpy as np
import torch
import torchvision
from Utils import *

def train(model, style, content, pastiche, i):
    
    style_pass = style
    content_pass = content
    pastiche_pass = pastiche
    
    model.optimizer.zero_grad()
    
    content_loss = 0
    style_loss = 0
    
    imsize = 512
    
    j = 0
    for layer in model.net.features:
        if isinstance(layer, torch.nn.Conv2d):
            j += 1
            name = "conv" + str(j)
            pastiche_pass, content_pass, style_pass = layer.forward(pastiche_pass), layer.forward(content_pass), layer.forward(style_pass)
            
            if name in model.content_layers:
                content_loss += model.loss(pastiche_pass, content_pass)
            
            if name in model.style_layers:
                pastiche_gram, style_gram = GramMatrix(pastiche_pass), GramMatrix(style_pass)
                style_loss += model.loss(pastiche_gram, style_gram)
           
        if isinstance(layer, torch.nn.ReLU):
            name = "relu" + str(j)
            layer = torch.nn.ReLU(inplace=False)
            pastiche_pass, content_pass, style_pass = layer.forward(pastiche_pass), layer.forward(content_pass), layer.forward(style_pass)
        
        if isinstance(layer, torch.nn.MaxPool2d):
            name = "maxpool" + str(j)
            pastiche_pass, content_pass, style_pass = layer.forward(pastiche_pass), layer.forward(content_pass), layer.forward(style_pass)
    
    total_loss = model.alpha * content_loss + model.beta * style_loss
    total_loss.backward(retain_graph=False)
    
    if i % 100 == 0:
        print("Total Loss: {}".format(total_loss.data))

    model.optimizer.step()
    return pastiche
