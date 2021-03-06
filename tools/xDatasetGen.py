#!/usr/bin/env python
# -*- coding: utf-8 -*-

import _init_paths
from core.train import get_training_roidb
from core.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, loadDatasetIndexDict
from datasets.factory import get_repo_imdb
from datasets.ds_utils import load_mixture_set,print_each_size,computeTotalAnnosFromAnnoCount,cropImageToAnnoRegion,roidbSampleHOG,roidbSampleImage
import numpy as np
import os.path as osp
import argparse,sys,os
import pprint
pp = pprint.PrettyPrinter(indent=4)

clsToSet = loadDatasetIndexDict()
paperFriendlySets = ["COCO","ImageNet","VOC","Caltech","INRIA","SUN","KITTI","CAM2"]

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='File for transforming x-dataset-gen results into a table. ')
    parser.add_argument('--type', dest='tableType',
                        help='which format of table to print',
                        default='txt', type=str)
    parser.add_argument('--color', dest='color',
                        help='print with coloring?',
                        action='store_true')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def readXDatasetResults(fn):
    lines = []
    xdata = {}
    with open(fn,"r") as f:
        header = f.readline().strip().split(',')[1:]
        print(header)
        for line in f.readlines():
            l = line.strip().split(',')
            xdata[l[0]] = dict(zip(header,map(float,l[1:])))
    return xdata

def computePercentDiff(actual,theoretical):
    return (actual - theoretical)/np.abs(theoretical)

def xDataToApDiff(xdata):
    aps = dict.fromkeys(xdata.keys(),None)
    apDiff = dict.fromkeys(xdata.keys(),None)
    for trSet,xResults in xdata.items():
        apDiff[trSet] = dict.fromkeys(xResults.keys(),None)
        for teSet,perf in xResults.items():
            if trSet == teSet:
                aps[trSet] = perf
                break

    for trSet,xResults in xdata.items():
        for teSet,perf in xResults.items():
            apDiff[trSet][teSet] = computePercentDiff(perf,aps[trSet])
    return apDiff,aps

def elementValues(diff):
    maxColorValue = 40
    colorValue = int(np.abs(diff) * maxColorValue)
    if colorValue > maxColorValue: colorValue = maxColorValue
    if diff < 0:
        color = "red"
        sign = "-"
    elif diff == 0:
        color = "gray"
        sign = ""
        colorValue = 20
    else:
        color = "blue"
        sign = "+"
    return color,sign,colorValue

def createLatexTable(xdata,params):
    outputFile = osp.join("./output/faster_rcnn/","xDatasetGen.txt")
    apDiff,aps = xDataToApDiff(xdata)
    diffMat = np.zeros((len(apDiff.keys()),len(apDiff['VOC'].keys())))
    outputStr = "Base AP & \\multicolumn{1}{c||}{\\backslashbox{Training on:}{Testing on:}} & "
    for idx,name in enumerate(paperFriendlySets):
        
        if idx == (len(paperFriendlySets) - 1):
            outputStr += "\\multicolumn{{1}}{{c|}}{{\\makebox[3em]{{{:s}}}}}".format(name)
        else:
            outputStr += "\\multicolumn{{1}}{{c}}{{\\makebox[3em]{{{:s}}}}}".format(name)
            outputStr += " & "

    outputStr += " & \multicolumn{1}{c|}{\makebox[3em]{$l_1$-norm}}"
    outputStr += "\\\\\n\\hline\n\\hline\n"
    outputStr += "\\arrayrulecolor{light-gray}\n"
    trSetCount = 0
    for name in paperFriendlySets:
        if name not in apDiff.keys(): continue
        outputStr += "{:2.4f} & {:s} & ".format(aps[name],name)
        trSetCount+=1
        for idx,teSet in enumerate(paperFriendlySets):
            diff = apDiff[name][teSet]
            tableValue = round(np.abs(diff) * 100,0)
            diffMat[trSetCount-1,idx] = diff
            color,sign,colorValue = elementValues(diff)
            if params['color']:
                outputStr += "\\cellcolor{{{0:s}!{1:d}}}{2:s}{3:2.0f}\%".format(color,
                                                                                colorValue,
                                                                                sign,tableValue)
            else:
                outputStr += "{0:s}{1:2.0f}\%".format(sign,tableValue)

            if idx == (len(paperFriendlySets) - 1):
                outputStr += " & {{\\bf {0:2.0f}\% }}".format(
                    sum(np.abs(apDiff[name].values()))*100)

            if idx == (len(paperFriendlySets) - 1) and trSetCount != len(aps):
                outputStr += "\\\\\n\\cline{3-11}\n"
            elif idx == (len(paperFriendlySets) - 1) and trSetCount == len(aps):
                outputStr += "\\\\\n\\arrayrulecolor{black}\n"                
                outputStr += "\\hline\n"
            else:
                outputStr += " & "


    print(diffMat)
    outputStr += " & Total & "
    colSums = np.mean(diffMat,axis=0)
    print(colSums)
    for idx,col in enumerate(colSums):
        tableValue = round(np.abs(col) * 100,0)
        color,sign,colorValue = elementValues(col)
        if params['color']:
            outputStr += "\\cellcolor{{{0:s}!{1:d}}}{2:s}{3:2.0f}\%".format(color,
                                                                            colorValue,
                                                                            sign,tableValue)
        else:
            outputStr += "{0:s}{1:2.0f}\%".format(sign,tableValue)

        if idx < (len(colSums)-1):
            outputStr += "&"
    outputStr += "& \\\\\n\\hline\n"
    
    print(outputStr)
            
    pp.pprint(apDiff)

def createTxtTable(xdata):
    outputFile = osp.join("./output/faster_rcnn/","xDatasetGen.txt")
    apDiff = [ [] for _ in range(len(xdata))]
    # for trSet,xResults in xdata.items():
    #     for teSet,perf in xResults.items():
    #         if trSet == teSet:
                

if __name__ == '__main__':

    args = parse_args()
    print('Called with args:')
    print(args)
    params = {}
    params['color'] = args.color

    fn = osp.join("./output/faster_rcnn/","xDatasetGen.txt")
    xdata = readXDatasetResults(fn)
    pp.pprint(xdata)

    if args.tableType == "latex":
        createLatexTable(xdata,params)
    elif args.tableType == "txt":
        createTxtTable(xdata)
        
