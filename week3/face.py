import argparse
import os.path as osp

import cv2
import face_recognition


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


def draw_bboxes(image, bboxes, text='cyanlaser'):
    """draw bboxes into image."""
    for bbox in bboxes:
        x, y, w, h = bbox
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (2, 255, 255), 2)
        image = cv2.putText(image, text, (x + 20, y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 2, 2), 2)

    return image


def extract_detections(image, backend='face-recognition', if_draw=True):
    """extract face detection from image.

    backend:
        1. opencv cascade detector;
        2. face-recognition package;
    """
    assert backend in ['face-recognition', 'opencv']

    if isinstance(image, str):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # face detection main part
    if backend == 'face-recognition':
        face_locations = face_recognition.face_locations(image)
        bboxes = []
        for face_location in face_locations:
            y1, x1, y2, x2 = face_location
            bbox = (x1, y1, x2 - x1, y2 - y1)
            bboxes.append(bbox)
    elif backend == 'opencv':
        detector = FaceDetector('haarcascade_frontalface_default.xml')
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        bboxes = detector(gray_image)

    if if_draw:
        image = draw_bboxes(image, bboxes, text='F**k')

    return bboxes, image


def video_process(cap, writer, if_show=False, backend='face-recognition'):
    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        _, frame = extract_detections(frame, backend=backend)

        if if_show:
            # Display the resulting frame
            cv2.imshow('frame', frame)
        writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()


def parse_args():
    parser = argparse.ArgumentParser('face detection')
    parser.add_argument('input_path', help='The image or video input path.')
    parser.add_argument(
        '-s',
        '--show',
        action='store_true',
        help='Whether to visulize result.')
    parser.add_argument(
        '-p',
        '--show-path',
        default='results.mp4',
        type=str,
        help='The storage path of visulization path.')

    return parser.parse_args()


def main():
    args = parse_args()
    path = args.input_path
    suffix = osp.splitext(path)[1]

    if suffix in ['.jpg', '.png', '.tif']:
        pass
    elif suffix in ['.avi', '.mp4']:
        cap = cv2.VideoCapture(path)
        # video codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # raw video fps
        fps = cap.get(cv2.CAP_PROP_FPS)
        # raw frame width and height
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.show_path, fourcc, fps,
                                 (width, height))  # 写入视频
        video_process(cap, writer, args.show)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
