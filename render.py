import os
import glob

from image import get_markup_figure, save_image
from video import split_video, stitch_video

INPUT_VIDEO_PATH = "./videos/beach_quiet_shore.mp4"
SPLIT_IMAGES_LOC = "./out/test-full"
OUTPUT_PATH = "./out/test-full-processed"
OUTPUT_VIDEO_NAME = "test-full.avi"

def main():
    files = glob.glob(f"{SPLIT_IMAGES_LOC}/*")
    for f in files:
        os.remove(f)
    frames_read, fps = split_video(INPUT_VIDEO_PATH, SPLIT_IMAGES_LOC)
    i = 0
    for image in os.listdir(SPLIT_IMAGES_LOC):
        if image == ".DS_Store":
            continue

        plt = get_markup_figure(os.path.join(SPLIT_IMAGES_LOC, image))
        save_image(plt, OUTPUT_PATH, i)
        i += 1
    stitch_video(OUTPUT_PATH, OUTPUT_PATH, OUTPUT_VIDEO_NAME, fps)



if __name__ == "__main__":
    main()