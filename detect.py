import torch
import numpy as np
import torchvision.transforms as T
from transformers import DetrForObjectDetection
from transformers import logging
from PIL import Image

logging.set_verbosity_error()

def get_normalized_image(image_path):
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def get_detections():

    # Load Model
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    config = model.config
    class_labels = config.id2label
    class_labels[91]="unknown"

    model.eval()

    # Transformation for input images
    image_path = "object-tests/test-beach.jpeg"
    image_tensor = get_normalized_image(image_path)

    # Perform inference
    with torch.no_grad():
        outputs = model(image_tensor)
        
        pred_boxes = outputs['pred_boxes'][0]
        pred_logits = outputs['logits'][0]

        # Only take detections that are a high enough confidence
        confidence_threshold = 0.5
        high_confidence_indices = (pred_logits.softmax(dim=-1).max(dim=-1).values > confidence_threshold)

        pred_boxes = pred_boxes[high_confidence_indices]
        pred_logits = pred_logits[high_confidence_indices]

    # Highest scoring class within each bounding box
    predicted_classes = torch.argmax(pred_logits, dim=1)

    # Confidence scores for each box
    confidence_scores = pred_logits.max(dim=1)[0].cpu().numpy()

    # Confidence -> Probability
    confidence_probs = 1 / (1 + np.exp(-confidence_scores))

    detections = []
    for i in range(len(pred_boxes)):
        xmin, ymin, xmax, ymax = pred_boxes[i]
        class_index = predicted_classes[i].item()
        detections.append({
            "box": (float(xmin), float(ymin), float(xmax), float(ymax)),
            "class_index": class_index,
            "class": class_labels[class_index],
            "confidence": confidence_probs[i].item()
        })

    return {"image": image_path, "detections": detections}



def main():
    out = get_detections()
    print(out['detections'])
    for d in out['detections']:
        if d["class"] != "unknown":
            print(d)


if __name__ == "__main__":
    main()
    
