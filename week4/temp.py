from config import config as c
from resnet50 import ResNetGenerator, ResNetDiscriminator

# model = ResNetGenerator(c)
model = ResNetDiscriminator(c)
# model.summary()