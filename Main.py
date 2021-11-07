from Model import *
from Utils import *
from Train import *

num_epochs = 500

def main(style, content, pastiche, device):
    style_transfer = StyleTransfer(style, content, pastiche, device)
    for i in range(num_epochs):
        output = train(style_transfer, style_transfer.style, style_transfer.content, style_transfer.pastiche, i)
    output = output.cpu()
    return output
