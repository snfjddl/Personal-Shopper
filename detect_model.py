from keras_retinanet import models

from PIL import Image

import numpy as np
import pandas as pd
import json
import cv2


def label_color(label):
    if label < len(colors):
        return colors[label]
    # else:
    #     warnings.warn('Label {} has no color, returning default.'.format(label))
    #     return (0, 255, 0)

colors = [
    [31, 0, 255],
    [0, 159, 255],
    [255, 95, 0],
    [255, 19, 0],
    [255, 0, 0],
    [255, 38, 0],
    [0, 255, 25],
    [255, 0, 133],
    [255, 172, 0],
    [108, 0, 255],
    [0, 82, 255],
    [0, 255, 6],
    [255, 0, 152],
    [223, 0, 255],
    [12, 0, 255],
    [0, 255, 178],
    [108, 255, 0],
    [184, 0, 255],
    [255, 0, 76],
    [146, 255, 0],
    [51, 0, 255],
    [0, 197, 255],
    [255, 248, 0],
    [255, 0, 19],
    [255, 0, 38],
    [89, 255, 0],
    [127, 255, 0],
    [255, 153, 0],
    [0, 255, 255],
    [0, 255, 216],
    [0, 255, 121],
    [255, 0, 248],
    [70, 0, 255],
    [0, 255, 159],
    [0, 216, 255],
    [0, 6, 255],
    [0, 63, 255],
    [31, 255, 0],
    [255, 57, 0],
    [255, 0, 210],
    [0, 255, 102],
    [242, 255, 0],
    [255, 191, 0],
    [0, 255, 63],
    [255, 0, 95],
    [146, 0, 255],
    [184, 255, 0],
    [255, 114, 0],
    [0, 255, 235],
    [255, 229, 0],
    [0, 178, 255],
    [255, 0, 114],
    [255, 0, 57],
    [0, 140, 255],
    [0, 121, 255],
    [12, 255, 0],
    [255, 210, 0],
    [0, 255, 44],
    [165, 255, 0],
    [0, 25, 255],
    [0, 255, 140],
    [0, 101, 255],
    [0, 255, 82],
    [223, 255, 0],
    [242, 0, 255],
    [89, 0, 255],
    [165, 0, 255],
    [70, 255, 0],
    [255, 0, 172],
    [255, 76, 0],
    [203, 255, 0],
    [204, 0, 255],
    [255, 0, 229],
    [255, 133, 0],
    [127, 0, 255],
    [0, 235, 255],
    [0, 255, 197],
    [255, 0, 191],
    [0, 44, 255],
    [50, 255, 0]
]

def compute_resize_scale(image_shape, min_side=800, max_side=1333):
    (rows, cols, _) = image_shape

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    return scale


def resize_image(img, min_side=800, max_side=1333):
    # compute scale to resize the image
    scale = compute_resize_scale(img.shape, min_side=min_side, max_side=max_side)

    # resize the image with the computed scale
    img = cv2.resize(img, None, fx=scale, fy=scale)

    return img, scale


def preprocess_image(x, mode='caffe'):
    # mostly identical to "https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py"
    # except for converting RGB -> BGR since we assume BGR already

    # covert always to float32 to keep compatibility with opencv
    x = x.astype(np.float32)

    if mode == 'tf':
        x /= 127.5
        x -= 1.
    elif mode == 'caffe':
        x -= [103.939, 116.779, 123.68]

    return x

def draw_caption(image, box, caption):
    """ Draws a caption above the box in an image.

    # Arguments
        image   : The image to draw on.
        box     : A list of 4 elements (x1, y1, x2, y2).
        caption : String containing the text to draw.
    """
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] + 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] + 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

THRES_SCORE = 0.6
def draw_detections(image, boxes, scores, labels):
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        if score < THRES_SCORE:
            break
        color = label_color(label)
        b = box.astype(int)
        draw_box(image, b, color=color)
        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(image, b, caption)


def read_image_bgr(path):
    """ Read an image in BGR format.
    Args
        path: Path to the image.
    """
    # We deliberately don't use cv2.imread here, since it gives no feedback on errors while reading the image.
    image = np.asarray(Image.open(path).convert('RGB'))

    return image[:, :, ::-1].copy()

def draw_box(image, box, color, thickness=2):
    """ Draws a box on an image with a given color.

    # Arguments
        image     : The image to draw on.
        box       : A list of 4 elements (x1, y1, x2, y2).
        color     : The color of the box.
        thickness : The thickness of the lines to draw a box with.
    """
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)

labels = json.load(open(r'C:\Users\tjrdl\Downloads\final_web\detect_labels.txt'))

ANNOTATIONS_FILE = r'C:\Users\tjrdl\Downloads\final_web\annotations.csv'
CLASSES_FILE = r'C:\Users\tjrdl\Downloads\final_web\classes.csv'

labels_to_names = pd.read_csv(
  CLASSES_FILE,
  header=None
).T.loc[0].to_dict()

model = models.load_model(r'C:\Users\tjrdl\Downloads\final_web\final_detection_model.h5')
model = models.convert_model(model)


def detect_img(path):
    image = read_image_bgr(path)
    # image = read_image_bgr('test1.jpeg')

    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

    # correct for image scale
    boxes /= scale

    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.4:
            break

        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)

    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, draw)
    # path_list = path.split('\\')
    # return_path = '/'+path_list[-3]+'/'+path_list[-2]+'/'+path_list[-1]
    # return return_path
