import matplotlib as plt 
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
        xmax = xmin + 20
        ymax = ymin + 20

        rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=1, edgecolor="r", facecolor="none")

        if d["class"] == "person":
            ax.add_patch(rect)


    return plt


def save_image(plt, output_name):
    plt.savefig(f"out/{output_name}.png")



def get_markup_image(image_path):
    processed_output = get_detections()
    plt = markup_image(processed_output)
    return plt