import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class FaceDetector(object):
    """Cascade Classifier (based on cv2.CascadeClassifier).

    Args:
        face_cascade (str): path to face cascade xml file.
    """

    def __init__(self, face_cascade):
        self.detector = cv2.CascadeClassifier(face_cascade)
        self.forward = self.detector.detectMultiScale

    def __call__(self, image):
        assert len(image.shape) <= 2, 'only support gray image input.'
        return self.forward(image)


def draw_bboxes(image, bboxes):
    """draw bboxes into image."""
    for bbox in bboxes:
        x, y, w, h = bbox
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 2, 255), 2)

    return image


def imshow(image, backend='pillow'):
    """backend agnostic image show.

    Args:
        image (numpy.ndarry): The image needs to save.
        save_path (str): The image storage path.
        backend (str): The storage operation package backend.
    """
    assert backend in ['cv2', 'pillow', 'matplotlib']
    if backend == 'cv2':
        cv2.imshow('face detection', image)
    elif backend == 'pillow':
        Image.fromarray(image).show()
    elif backend == 'matplotlib':
        plt.imshow(image)
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def imwrite(image, save_path, backend='pillow'):
    """backend agnostic image write

    Args:
        image (numpy.ndarry): The image needs to save.
        save_path (str): The image storage path.
        backend (str): The storage operation package backend.
    """
    assert backend in ['cv2', 'pillow', 'matplotlib']
    assert isinstance(image, np.ndarray)
    if backend == 'cv2':
        cv2.imwrite(save_path, image)
    elif backend == 'pillow':
        Image.fromarray(image).save(save_path)
    elif backend == 'matplotlib':
        plt.imshow(image)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path)


def parse_args():
    parser = argparse.ArgumentParser('face detection')
    parser.add_argument('image_path', help='The image input path.')
    parser.add_argument(
        '-s',
        '--show',
        action='store_true',
        help='Whether to visulize result.')
    parser.add_argument(
        '-p',
        '--show-path',
        type=str,
        help='The storage path of visulization path.')

    return parser.parse_args()


def main():
    args = parse_args()
    image_path = args.image_path
    show = args.show
    show_path = args.show_path

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    detector = FaceDetector('week2/haarcascade_frontalface_default.xml')

    bboxes = detector(gray_image)
    image = draw_bboxes(image, bboxes)

    if show:
        imshow(image)
    if show_path:
        imwrite(image, show_path)


if __name__ == '__main__':
    main()
