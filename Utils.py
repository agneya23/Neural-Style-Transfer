import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

imsize = 512

transform = transforms.Compose([
             transforms.Resize([imsize, imsize]),
             transforms.ToTensor()
            ])

def GramMatrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

def image_loader(image_name):
    image = Image.open(image_name)
    image = Variable(transform(image))
    image = image.unsqueeze(0)
    return image
