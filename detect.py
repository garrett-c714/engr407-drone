import warnings
import urllib3

"""
warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)
warnings.filterwarnings("ignore", message="Some weights of the model.*")
"""

import torch
import torchvision.transforms as T
from transformers import DetrForObjectDetection
from transformers import logging
from PIL import Image

logging.set_verbosity_error()

# Load Model
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
#checkpoint = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth")
#print(checkpoint)
#model.load_state_dict(checkpoint["model"])

#exit()

model.eval()

# Transformation for input images
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image = Image.open("object-tests/test-desk.jpeg")
image_tensor = transform(image).unsqueeze(0)

# Perform inference
with torch.no_grad():
    outputs = model(image_tensor)

# Remove batch dimension
pred_boxes = outputs['pred_boxes'][0]
print(outputs.keys())
pred_logits = outputs['logits'][0]

print(pred_boxes)
print(pred_logits)
    
