import argparse
import os

import mmcv

from mmedit.apis import init_model, restoration_inference
from mmedit.core import tensor2img


def parse_args():
    parser = argparse.ArgumentParser(description='Restoration demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('img_path', help='path to input image file')
    parser.add_argument('save_path', help='path to save restoration result')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if not os.path.isfile(args.img_path):
        raise ValueError('It seems that you did not input a valid '
                         '"image_path". Please double check your input, or '
                         'you may want to use "restoration_video_demo.py" '
                         'for video restoration.')

    model = init_model(args.config, args.checkpoint)

    output = restoration_inference(model, args.img_path)
    output = tensor2img(output)

    mmcv.imwrite(output, args.save_path)


if __name__ == '__main__':
    main()
