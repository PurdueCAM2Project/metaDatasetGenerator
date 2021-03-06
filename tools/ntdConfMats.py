#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")

import _init_paths
from core.train import get_training_roidb
from core.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, loadDatasetIndexDict,iconicImagesFileFormat
from datasets.factory import get_repo_imdb
from datasets.ds_utils import load_mixture_set,print_each_size,computeTotalAnnosFromAnnoCount,cropImageToAnnoRegion,roidbSampleHOG,roidbSampleImage
import os.path as osp
import datasets.imdb
import argparse
import pprint
import numpy as np
import matplotlib.pyplot as plt
import sys,os,cv2,pickle,uuid
# pytorch imports
from datasets.pytorch_roidb_loader import RoidbDataset
from numpy import transpose as npt
from utils.misc import *
from ntd.ntd_utils import *
from ntd.hog_svm import plot_confusion_matrix, extract_pyroidb_features,appendHOGtoRoidb,split_data, scale_data,train_SVM,findMaxRegions, make_confusion_matrix

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test loading a mixture dataset')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--setID', dest='setID',
                        help='which 8 digit ID to read from',
                        default=['11111111'],nargs='*',type=str)
    parser.add_argument('--repeat', dest='repeat',
                        help='which repeat to read from',
                        default=[1],nargs='*',type=int)
    parser.add_argument('--size', dest='size',
                        help='which size to read from',
                        default=[250],nargs='*',type=int)
    parser.add_argument('--save', dest='save',
                        help='save some samples with bboxes visualized?',
                        action='store_true')
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--modelRaw', dest='modelRaw',
                        help='give the path to a fit model',
                        default=[], nargs='*', type=str)
    parser.add_argument('--modelCropped', dest='modelCropped',
                       help='give the path to a fit model',
                        default=[], nargs='*', type=str)

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
    if not args.randomize:
        np.random.seed(cfg.RNG_SEED)
    print('Using config:')
    pprint.pprint(cfg)

    cfg.DEBUG = False
    cfg.uuid = str(uuid.uuid4())
    ntdGameInfo = {}
    ntdGameInfo['trainSize'] = 1000
    ntdGameInfo['testSize'] = 1000
    ntdGameInfo['Raw'] = {}
    ntdGameInfo['Cropped'] = {}
    ntdGameInfo['Raw']['trainSize'] = 300
    ntdGameInfo['Raw']['testSize'] = 300
    ntdGameInfo['Cropped']['trainSize'] = 300
    ntdGameInfo['Cropped']['testSize'] = 300

    modelParams = {}
    modelParams['modelType'] = "dl"
    modelParams['dl_arch'] = "vgg16"
    modelParams['modelType'] = "svm"

    setID_l = args.setID
    repeat_l = args.repeat
    size_l = args.size

    cmRaw_l = []
    cmCropped_l = []
    cmDiff_l = []
    for setID in setID_l:
        for repeat in repeat_l:
            for size in size_l:


                ntdGameInfo['setID'] = setID
                ntdGameInfo['repeat'] = repeat
                ntdGameInfo['size'] = size
                print("REAPEAT {}".format(repeat))

                # force size to be 5000 so at least 1000 images are in each mixedset
                # HACK
                roidbTrDict,roidbTeDict,roidbTrDict1k,roidbTeDict1k,dsHasTest,annoSizes = prepareMixedDataset(setID,repeat,5000)
                ntdGameInfo["dsHasTest"] = dsHasTest
                print(dsHasTest)
                print(annoSizes)


                modelParams['modelFn'] = None
                if len(args.modelRaw) > repeat: modelParams['modelFn'] = args.modelRaw[repeat]
                roidbTr = flattenRoidbDict(roidbTrDict)
                roidbTe = flattenRoidbDict(roidbTeDict)
                print("roidb length of:\ntrain: {}\ntest: {}\n".format(len(roidbTr),len(roidbTe)))
                cmRaw,modelRaw = genConfRaw(modelParams, roidbTr, roidbTe, ntdGameInfo)

                modelParams['modelFn'] = None
                if len(args.modelCropped) > repeat: modelParams['modelFn'] = args.modelCropped[repeat]
                roidbTr = flattenRoidbDict(roidbTrDict1k)
                roidbTe = flattenRoidbDict(roidbTeDict1k)
                print("roidb length of:\ntrain: {}\ntest: {}\n".format(len(roidbTr),len(roidbTe)))
                cmCropped,modelCropped = genConfCropped(modelParams, roidbTr, roidbTe, ntdGameInfo)

                cmDiff = cmRaw - cmCropped
                saveNtdConfMats(cmRaw,cmCropped,ntdGameInfo)
                plotNtdConfMats(cmRaw,cmCropped,cmDiff,ntdGameInfo)
                cmRaw_l.append(cmRaw)
                cmCropped_l.append(cmCropped)
                cmDiff_l.append(cmDiff)

    saveNtdSummaryStats(cmRaw_l,cmCropped_l,cmDiff_l)
    print("\n\n -=-=-=- uuid: {} -=-=-=- \n\n".format(cfg.uuid))
   
