from yolact import Yolact
from utils.augmentations import FastBaseTransform
from utils.functions import SavePath
from layers.output_utils import postprocess

from data import cfg, set_cfg

import torch
import torch.backends.cudnn as cudnn
import argparse
import random
import cv2

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='YOLACT COCO Evaluation')
    parser.add_argument('--trained_model',
                        default='weights/yolact_base_54_800000.pth', type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument('--image', default=None, type=str,
                        help='A path to an image to use for display.')


    global args
    args = parser.parse_args(argv)


if __name__ == '__main__':
    parse_args()

    model_path = SavePath.from_str(args.trained_model)
    args.config = model_path.model_name + '_config'
    print('Config not specified. Parsed %s from the file name.\n' % args.config)
    set_cfg(args.config)

    print('Loading model...', end='')
    net = Yolact(model_path=args.trained_model)
    print(' Done.')

    net.infer(path_in=args.image)


