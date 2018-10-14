#!/usr/bin/env python
# --------------------------------------------------------
# --------------------------------------------------------

"""Test an object detection network on an image database."""

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import _init_paths
from core.test import test_net
from core.config import cfg, cfg_from_file, cfg_from_list, set_global_cfg
from datasets.factory import get_repo_imdb
import caffe
import argparse
import pprint
import time, os, sys


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test an Object Detection network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--vis', dest='vis', help='visualize detections',
                        action='store_true')
    parser.add_argument('--num_dets', dest='max_per_image',
                        help='max number of detections per image',
                        default=100, type=int)
    parser.add_argument('--rotate', dest='rotate',
                        help='how much should we rotate each image?',
                        default=0, type=int)
    # params for model to which active learning is applied
    parser.add_argument('--al_def', dest='al_def',
                        help='model prototxt to which active learning is applied',
                        default=None, type=str)
    parser.add_argument('--al_net', dest='al_net',
                        help='model weights to which active learning is applied',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    set_global_cfg("TEST")

    cfg.GPU_ID = args.gpu_id
    cfg.ROTATE_IMAGE = args.rotate

    print('Using config:')
    pprint.pprint(cfg)

    while not os.path.exists(args.caffemodel) and args.wait:
        print('Waiting for {} to exist...'.format(args.caffemodel))
        time.sleep(10)

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]

    print(args.imdb_name)
    imdb = get_repo_imdb(args.imdb_name)
    imdb.competition_mode(args.comp_mode)
    if not cfg.TEST.OBJ_DET.HAS_RPN and cfg.TASK == 'object_detection':
        imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)

    al_net = None
    if args.al_net is not None and args.al_def is not None:
        al_net = caffe.Net(args.al_def,caffe.TEST, weights=args.al_net)
        al_net.name = "al_"+os.path.splitext(os.path.basename(args.al_def))[0]

    test_net(net, imdb, max_per_image=args.max_per_image, vis=args.vis, al_net=al_net)
    
'''
argparse.ArgumentParser
Input: (description='Test an Object Detection network'), Output: parser

caffe.Net
input: (args.prototxt, args.caffemodel, caffe.TEST), Output: net

os.path.splitext
input: (os.path.basename(args.caffemodel)), output: net.name

get_repo_imdb
input: (args.imdb_name), output: imdb
'''
    
