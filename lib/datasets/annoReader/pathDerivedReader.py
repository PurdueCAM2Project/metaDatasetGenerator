
# --------------------------------------------------------
# CBAL
# Written by Kent Gauen
# --------------------------------------------------------

import os,re
import os.path as osp
import PIL
import numpy as np
import scipy.sparse
from core.config import cfg,cfgData
from easydict import EasyDict as edict

class pdReader(object):
    """Image database."""

    def __init__(self, annoPath, classes , datasetName, setID,useDiff = True,
                 convertToPerson = None, convertIdToCls = None,
                 is_image_index_flattened=False, splitIndex=1):
        """
        __init__ function for annoReader [annotationReader]

        """
        self._annoPath = annoPath
        self._classes = classes
        self._datasetName = datasetName
        self._setID = setID
        self.num_classes = len(classes)
        self.useDiff = useDiff
        self._classToIndex = self._create_classToIndex(classes)
        self._convertToPerson = convertToPerson
        self._convertIdToCls = convertIdToCls
        self._is_image_index_flattened = is_image_index_flattened
        self._splitIndex = 1

    def _create_classToIndex(self,classes):
        return dict(zip(classes, range(self.num_classes)))
        
    def load_annotation(self,index):
        """
        load annotations depending on how the annotation should be loaded

        """
        return self._load_derived_cls_annotation(index)

    def _load_derived_cls_annotation(self, index):
        filename = os.path.join(self._annoPath, index + '.txt')
        id_str = index.split("/")[self._splitIndex]
        cls_name = int(self._convertIdToCls[cls_str])
        # ensure cls is a classname
        assert cls_name in self._classes, "class {} is not in available class list".format(cls)
        cls = self._classToIndex[cls]
        return {'gt_classes': np.array([cls],dtype=np.uint8),
                'flipped' : False,
                'set' : self._setID}

