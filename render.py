import os

from image import get_markup_figure, save_image
from video import split_video, stitch_video

INPUT_VIDEO_PATH = ""
SPLIT_IMAGES_LOC = ""
OUTPUT_PATH = ""
OUTPUT_VIDEO_NAME = ""

def main():
    split_video(INPUT_VIDEO_PATH, SPLIT_IMAGES_LOC)
    i = 0
    for image in os.listdir(SPLIT_IMAGES_LOC):
        plt = get_markup_figure(os.path.joinpath(SPLIT_IMAGES_LOC, image))
        save_image(plt, OUTPUT_PATH, i)
        i += 1
    stitch_video(SPLIT_IMAGES_LOC, SPLIT_IMAGES_LOC, OUTPUT_VIDEO_NAME)