import os
import cv2

from PIL import Image

# path = "./object-tests"

def resize_images(path):
    mean_width = 0
    mean_height = 0

    num_images = len(os.listdir(path))
    print(num_images)


    for f in os.listdir(path):
        if f == ".DS_Store":
            continue

        image = Image.open(os.path.join(path, f))
        width, height = image.size
        mean_width += width
        mean_height += height


    mean_width = int(mean_width / num_images)
    mean_height = int(mean_height / num_images)


    for f in os.listdir(path):
        if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png"):
            image = Image.open(os.path.join(path, f))

            width, height = image.size

            # Resize image to the mean width and height
            resized_image = image.resize((mean_width, mean_height), Image.LANCZOS).convert("RGB")
            f_split = f.split('.')
            resized_image.save(os.path.join(path, f"{f_split[0]}_r.jpeg"), 'JPEG', quality=95)


def get_file_number(f):
    return int(f.split("_")[0])


def generate_video(image_folder, video_folder, video_name, fps):
    # FIXME: passed-in arguments
    '''
    image_folder = path
    video_folder = "./videos"
    video_name = "test_video.avi"
    '''

    images = [img for img in os.listdir(image_folder)
              if img.endswith("_r.jpg") or 
              img.endswith("_r.jpeg") or 
              img.endswith("_r.png")] 
    
    images = sorted(images, key=get_file_number)
    
    #print(images)

    frame = cv2.imread(os.path.join(image_folder, images[0]))

    height, width, layers = frame.shape

    video = cv2.VideoWriter(f"{video_folder}/" + video_name, 0, fps, (width, height))

    #for image in images:
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

    for image in images:
        os.remove(os.path.join(image_folder, image))



def split_video(video, output_folder):
    vidcap = cv2.VideoCapture(video)
    success, image = vidcap.read()

    fps = vidcap.get(cv2.CAP_PROP_FPS)

    # print("Got here")
    # print(success)

    i = 0
    while success:
        cv2.imwrite(os.path.join(output_folder, f"{i}.jpg"), image)
        success, image = vidcap.read()
        i += 1

    return i, fps


def stitch_video(image_folder, video_folder, video_name, fps):
    resize_images(image_folder)
    generate_video(image_folder, video_folder, video_name, fps)

    

    