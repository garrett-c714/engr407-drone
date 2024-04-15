import os

import matplotlib.pyplot as plt 
import matplotlib.patches as patches

from detect import get_detections

def markup_image(processing_output):
    image_path = processing_output['image']
    detections = processing_output['detections']

    fig, ax = plt.subplots(1)
    image = plt.imread(image_path)

    ax.imshow(image)

    image_height, image_width = image.shape[:2]

    for d in detections:
        box = d["box"]
        xmin = int(box[0] * image_width)
        ymin = int(box[1] * image_height)
        xmin+=20
        ymin+=20
        xmax = xmin - 50
        ymax = ymin - 50

        rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=1, edgecolor="r", facecolor="none")

        '''
        if d["class"] == "person":
            ax.add_patch(rect)
        '''
        ax.add_patch(rect)

    return plt


def save_image(plt, output_path, output_name):
    plt.savefig(os.path.joinpath(output_path, f"{output_name}.png"))



def get_markup_figure(image_path):
    processed_output = get_detections(image_path)
    plt = markup_image(processed_output)
    return plt