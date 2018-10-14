# --------------------------------------------------------
# CBAL
# Written by Kent Gauen
# --------------------------------------------------------

"""Method for loading the datasets by name."""

__sets = []

import os, glob
import numpy as np
from core.config import cfg, cfg_from_file, cfg_from_list
from datasets.repo_imdb import RepoImdb
from datasets.coco import coco

__sets.append("pascal_voc_2012")
__sets.append("pascal_voc_2007")
__sets.append("pascal_voc")
__sets.append("imagenet")
__sets.append("imagenet_cls")
__sets.append("cam2")
__sets.append("caltech")
__sets.append("kitti")
__sets.append("inria")
__sets.append("coco")
__sets.append("sun")
__sets.append("mnist")

def get_repo_imdb(name,path_to_imageSets=None):
    """Get an imdb (image database) by name."""
    print(name)
    cfg.CALLING_DATASET_NAME = name
    di = name.split("-")
    if len(di) != 3:
        raise KeyError('Dataset name [{}] is not correct length'.format(name))
    for __set in __sets:
        if di[0] == __set:
            print(di)
            return RepoImdb(di[0],di[1],di[2],path_to_imageSets=path_to_imageSets)
            # if __set == "coco":
            #     return coco(di[0],di[1],di[2])
            # else:
            #     return RepoImdb(di[0],di[1],di[2])
    raise KeyError('Unknown dataset: {}'.format(name))


def list_imdbs():
    """List all registered imdbs."""
    return __sets
