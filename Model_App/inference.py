import torch
import torchvision.transforms as transforms
from PIL import Image

from dataset import transform
from model import create_model
from config import PRETRAINED_MODEL




model = create_model()
model.load_state_dict(torch.load(PRETRAINED_MODEL))

model.eval()

image = Image.open("path/to/image.jpg")
image = transform(image).unsqueeze(0)

with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    print(f'Predicted class: {predicted.item()}')