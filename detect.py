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
    #checkpoint = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth")
    #print(checkpoint)
    #model.load_state_dict(checkpoint["model"])

    #exit()

    model.eval()

    # Transformation for input images
    image_path = "object-tests/test-desk.jpeg"
    image_tensor = get_normalized_image(image_path)

    # Perform inference
    with torch.no_grad():
        outputs = model(image_tensor)
        
        pred_boxes = outputs['pred_boxes'][0]
        pred_logits = outputs['logits'][0]

        confidence_threshold = 0.5
        high_confidence_indices = (pred_logits.softmax(dim=-1).max(dim=-1).values > confidence_threshold)

        pred_boxes = pred_boxes[high_confidence_indices]
        pred_logits = pred_logits[high_confidence_indices]

    print(pred_boxes)
    print(pred_logits)

    print(len(pred_boxes))
    print(len(pred_logits))


    # Highest scoring class within each bounding box
    predicted_classes = torch.argmax(pred_logits, dim=1)
    print(len(predicted_classes))

    # Confidence scores for each box
    confidence_scores, _ = pred_logits.softmax(dim=2).max(dim=2)
    confidence_scores = confidence_scores.cpu().numpy()

    # Confidence -> Probability
    confidence_probs = 1 / (1 + np.exp(-confidence_scores))

    '''
    print(pred_boxes[0])
    print(predicted_classes[0])
    print(confidence_scores[0])
    '''

    print(len(predicted_classes[0]))
    print(len(pred_boxes[0]))
    print(len(confidence_probs[0]))


    detections = []
    # Only take detections that are a high enough confidence
    predicted_classes = predicted_classes[0]
    pred_boxes = pred_boxes[0]
    confidence_probs = confidence_probs[0]

    for i in range(min(len(predicted_classes), len(pred_boxes), len(confidence_probs))):
        pass

    '''
    confidence_threshold = 0.5
    detections = []
    for box, scores, classes in zip(pred_boxes[0], confidence_probs, predicted_classes):
        for score, class_index in zip(scores, classes):
            if score > confidence_threshold:
                xmin, ymin, xmax, ymax = box

                detections.append({
                    "box": (float(xmin), float(ymin), float(xmax), float(ymax)),
                    "class_index": class_index.item(),
                    "confidence": score.item()                                   
                })
    '''


    # return {"image": image_path, "boxes": pred_boxes, "logits": pred_logits}
    return {"image": image_path, "detections": detections}



def main():
    out = get_detections()
    print(out['detections'])


if __name__ == "__main__":
    main()
    
