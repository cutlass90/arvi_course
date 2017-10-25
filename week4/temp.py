from config import config as c
from resnet50 import ResNetGenerator

model = ResNetGenerator(c)
model.summary()