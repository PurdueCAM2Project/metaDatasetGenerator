
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
import xml.etree.ElementTree as ET


class txtReader(object):
    """Image database."""

    def __init__(self, annoPath, classes , datasetName, setID,
                 bboxOffset = 0,useDiff = True, cleanRegex = None,
                 convertToPerson = None, convertIdToCls = None, anno_type='box',
                 is_image_index_flattened=False):
        """
        __init__ function for annoReader [annotationReader]

        """
        self._annoPath = annoPath
        self._bboxOffset = bboxOffset
        self._datasetName = datasetName
        self._setID = setID
        self.num_classes = len(classes)
        self.useDiff = useDiff
        self._classToIndex = self._create_classToIndex(classes)
        self._convertIdToCls = convertIdToCls
        self._convertToPerson = convertToPerson
        self._anno_type = anno_type
        self._is_image_index_flattened = is_image_index_flattened
        if cleanRegex is not None: self._cleanRegex = cleanRegex # used for INRIA
        else:
            self._cleanRegex = r"(?P<cls>[0-9]+) (?P<xmin>[0-9]*\.[0-9]*) (?P<ymin>[0-9]*\.[0-9]*) (?P<xmax>[0-9]*\.[0-9]*) (?P<ymax>[0-9]*\.[0-9]*)"
        
    def _create_classToIndex(self,classes):
        return dict(zip(classes, range(self.num_classes)))
        
    def load_annotation(self,index):
        """
        load annotations depending on how the annotation should be loaded

        """
        if self._anno_type is 'box':
            if self._is_image_index_flattened:
                return self._load_single_txt_box_annotation(index)
            else:
                return self._load_txt_box_annotation(index)
        elif self._anno_type is 'cls': return self._load_txt_cls_annotation(index)

    def _load_txt_cls_annotation(self, index):
        filename = os.path.join(self._annoPath, index + '.txt')
        annos = []
        with open(filename,"r") as f:
            annos = f.readlines()
            return {'gt_classes': np.array(annos,dtype=np.uint8),
                    'boxes': [None],
                    'flipped' : False,
                    'set' : self._setID}

    def _load_txt_box_annotation(self, index):
        """
        requires the following format @ 
        <PATH_TO_ANNOTATIONS>/*txt
        """

        filename = os.path.join(self._annoPath, index + '.txt')
        annos = []
        with open(filename,"r") as f:
            annos = f.readlines()

        num_objs = 0 
        if self._cleanRegex is not None:
            for idx,line in enumerate(annos):
                m = re.match(self._cleanRegex,line)
                if m is not None:
                    if self._find_cls(m.groupdict()) != -1:
                        num_objs += 1
        else:
            num_objs = len(annos)
                    
        # reformat into the dictionary
        boxes = np.zeros((num_objs, 4), dtype=np.int16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        ix = 0

        for line in annos:
            m = re.match(self._cleanRegex,line)
            if m is not None:
                mgd = m.groupdict()
                x1,y1,x2,y2 = self._extract_bounding_box(mgd)
                cls = self._find_cls(mgd)
                if cls == -1:
                    continue
                boxes[ix, :] = [x1, y1, x2, y2]
                gt_classes[ix] = cls
                overlaps[ix, cls] = 1.0
                seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
                ix += 1

        overlaps = scipy.sparse.csr_matrix(overlaps)

        if cfg.OBJ_DET.BBOX_VERBOSE:
            bbox = {'boxes' : boxes,
                    'gt_classes': gt_classes,
                    'gt_overlaps' : overlaps,
                    'flipped' : False,
                    'seg_areas' : seg_areas,
                    'set' : self._setID}
        else:
            bbox = {'boxes' : boxes,
                    'gt_classes': gt_classes,
                    'flipped': False,
                    'set' : self._setID}
        return bbox

    def _load_single_txt_box_annotation(self, index):
        """
        requires the following format @ 
        <PATH_TO_ANNOTATIONS>/*txt
        """

        bbox_index = int(index.split('_')[-1])
        image_index = '_'.join(index.split('_')[:-1])
        filename = os.path.join(self._annoPath, image_index + '.xml')
        annos = []
        with open(filename,"r") as f:
            annos = f.readlines()

        num_objs = 0 
        if self._cleanRegex is not None:
            for idx,line in enumerate(annos):
                m = re.match(self._cleanRegex,line)
                if m is not None:
                    if self._find_cls(m.groupdict()) != -1:
                        num_objs += 1
        else:
            num_objs = len(annos)
                    
        num_objs = 1 # by definition of the function
        # reformat into the dictionary
        boxes = np.zeros((num_objs, 4), dtype=np.int16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        ix = 0
        for line in annos:
            m = re.match(self._cleanRegex,line)
            if m is not None:
                mgd = m.groupdict()
                x1,y1,x2,y2 = self._extract_bounding_box(mgd)
                cls = self._find_cls(mgd)
                if cls == -1:
                    continue
                if ix != bbox_index:
                    ix += 1
                    continue
                boxes[0, :] = [x1, y1, x2, y2]
                gt_classes[0] = cls
                overlaps[0, cls] = 1.0
                seg_areas[0] = (x2 - x1 + 1) * (y2 - y1 + 1)
                break

        overlaps = scipy.sparse.csr_matrix(overlaps)

        if cfg.OBJ_DET.BBOX_VERBOSE:
            bbox = {'boxes' : boxes,
                    'gt_classes': gt_classes,
                    'gt_overlaps' : overlaps,
                    'flipped' : False,
                    'seg_areas' : seg_areas,
                    'set' : self._setID}
        else:
            bbox = {'boxes' : boxes,
                    'gt_classes': gt_classes,
                    'flipped': False,
                    'set' : self._setID}
        return bbox

    def _find_cls(self,mgd):
        if "cls" in mgd.keys():
            cls = mgd['cls'].lower().strip()

            if re.match(r"[0-9]+",cls) is None:
                cls = self.mangle_cls(cls)
            else:
                cls = int(cls)
                # remove the index if in class list
                if cls not in self._classToIndex.values():
                    cls = -1
        else:
            cls = self._classToIndex["person"]
        return cls

    def mangle_cls(self,cls):

        # used for imagenet mapping
        if self._convertIdToCls is not None: cls = self._convertIdToCls[cls]

        # check if we need to convert annotation class to "person"
        if self._convertToPerson is not None and cls in self._convertToPerson:
            cls = self._classToIndex["person"]
        # remove all not in classToIndex
        elif cls in self._classToIndex.keys():
            cls = self._classToIndex[cls]
        else:
            return -1
        return cls

    def _extract_bounding_box(self,mgd):
        if 'xmin' in mgd:
            x1 = float(mgd['xmin'])
            y1 = float(mgd['ymin'])
            x2 = float(mgd['xmax'])
            y2 = float(mgd['ymax'])
        elif 'center_x' in mgd:
            w = float(mgd['width'])
            h = float(mgd['height'])
            cx = float(mgd['center_x'])
            cy = float(mgd['center_y'])
            x1 = cx - w/2
            x2 = cx + w/2
            y1 = cy - h/2
            y2 = cy + h/2
        else:
            raise ValueError("extract bounding box fails!")
        x1,y1,x2,y2 = self._handle_caltech_helps_vs_gauenk(*[x1, y1, x2, y2])
        return x1,y1,x2,y2

    def _handle_caltech_helps_vs_gauenk(self,x1,y1,x2,y2):
        if "helps" in cfg.PATH_YMLDATASETS and "caltech" in cfgData.EXP_DATASET: 
            x1 = x1 * 640
            y1 = y1 * 480
            x2 = x2 * 640
            y2 = y2 * 480
        return x1,y1,x2,y2
        

