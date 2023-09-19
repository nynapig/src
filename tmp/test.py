import torchvision.models as models
import torch
import torch.nn as nn
model = torch.load("VGG16_testBest.pt") 



feature = torch.nn.Sequential(*list(model.childer())[:])
print(feature)