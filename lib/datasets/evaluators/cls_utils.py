import numpy as np
import pickle,sys
from datasets import ds_utils
from core.config import cfg,cfgData
import os.path as osp

def cocoID_to_base(cocoID):
    if "COCO" in cocoID:
        index = int(re.match(".*(?P<id>[0-9]{12})",cocoID).groupdict()['id'])
    else:
        index = cocoID
    return index

def parse_anno(filename,load_annotation,classes):
    # convert if coco
    index = cocoID_to_base(filename)
    anno = load_annotation(index)
    objects = []
    for idx,gt_classes in enumerate(anno['gt_classes']):
        obj_struct = {}
        obj_struct['anno'] = filename
        obj_struct['cls'] = int(anno['gt_classes'][idx])
        obj_struct['cls_id'] = idx
        objects.append(obj_struct)
    return objects

def load_cls_groundTruth(classname,cachedir,imagesetfile,annopath,annoReader,indexToCls):
    # first load gt
    if not osp.isdir(cachedir):
        os.mkdir(cachedir)
    print(cachedir)
    cachefile = osp.join(cachedir, 'annots.pkl')

    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not osp.isfile(cachefile):
        # load annots
        annos = {}
        for i, imagename in enumerate(imagenames):
            annos[imagename] = parse_anno(imagename,annoReader,indexToCls)
            if i % 100 == 0:
                print 'Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames))
        # save
        print 'Saving cached annotations to {:s}'.format(cachefile)
        with open(cachefile, 'w') as f:
            pickle.dump(annos, f)
    else:
        # load
        print("cachefile @ {:s}".format(cachefile))
        with open(cachefile, 'r') as f:
            annos = pickle.load(f)

    return imagenames, annos

def load_groundTruth(classname,cachedir,imagesetfile,annopath,\
                     _load_annotation,_classes):
    if cfg.SUBTASK == 'tp_fn':
        records = ds_utils.loadEvaluationRecords(classname)
        imagenames,probs = zip(*records.items())
    elif cfg.SUBTASK in ["default","al_subset"]:
        imagenames,annos = load_cls_groundTruth(classname,cachedir,imagesetfile,annopath,\
                                                _load_annotation,_classes)
    return imagenames, annos

def extractClassGroundTruth(imagenames,annos,classname):
    if cfg.SUBTASK == 'tp_fn':
        image_ids,class_probs,npos,nneg = extractImageIdsAndProbs(imagenames,annos,True)
        class_probs_dict = dict(zip(image_ids,class_probs))
    elif cfg.SUBTASK in ["default","al_subset"]:
        image_ids,class_probs,npos,nneg = extractImageIdsAndProbs(imagenames,annos,False)
        class_probs_dict = dict(zip(image_ids,class_probs))
    else:
        print("ERROR: unknown cfg.Subtask {} in [lib/datasets/evaluate/cls_utils.py: extractClassGroundTruth]".format(cfg.SUBTASK))
        sys.exit()
    return image_ids,class_probs_dict,npos,nneg

def loadModelCls(detpath,classname):
    # read the file with the model detections
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    # interpret each line from the file
    if cfg.SUBTASK == "tp_fn":
        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        items = np.array([ [ round(float(y),3) for y in x[1:] ] for x in splitlines])
    elif cfg.SUBTASK in ["default","al_subset"]:
        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        items = np.array([int(x[1]) for x in splitlines]) # index @ 1 is the guessed class
    else:
        print("ERROR: unknown cfg.Subtask {} in [lib/datasets/evaluate/cls_utils.py: loadModelCls]".format(cfg.SUBTASK))

        
    # sort the detections by confidence for scoring    
    if cfg.SUBTASK == "tp_fn":
        if items.shape[0] == 1 or len(items.shape) == 1:
            sorted_ind = np.argsort(-items)
            sorted_scores = np.sort(-items)
            image_ids = [image_ids[x] for x in sorted_ind]

    return image_ids, items

def extractImageIdsAndProbs(imagenames,annos,flatten):
    if type(annos) is dict:
        return extractImageIdsAndProbs_annoDict(imagenames,annos,flatten)
    elif type(annos) is list:
        return extractImageIdsAndProbs_annoList(imagenames,annos,flatten)
    
def extractImageIdsAndProbs_annoDict(imagenames,annos,flatten):
    npos = 0
    nneg = 0
    gt_classes = []
    for image_id in imagenames:
        anno = annos[image_id]
        if flatten is False and cfg.SUBTASK in ["default","al_subset"]:
            gt_classes.append(anno[0]['cls'])
    return imagenames,np.array(gt_classes),npos,nneg
    
def extractImageIdsAndProbs_annoList(imagenames,annos,flatten):
    npos = 0
    nneg = 0
    gt_classes = []
    image_ids = []
    for image_id,anno in zip(imagenames,annos):
        for item in anno:
            if flatten:
                image_id_new = image_id + "_{}".format(str(n_image_items))
            else:
                image_id_new = image_id
            image_ids.append(image_id_new)
            gt_classes.append(item)
            npos += (item == 1)
            nneg += (item == 0)
            n_image_items += 1
    return image_ids,np.array(gt_classes),npos,nneg


def compute_metrics(ovthresh,image_ids,model_probs,gt_classes,num_classes):
    if cfg.SUBTASK == "tp_fn":
        return compute_metrics_v1(ovthresh,image_ids,model_probs,gt_classes)
    elif cfg.SUBTASK in ["default","al_subset"]:
        return compute_metrics_v2(ovthresh,image_ids,model_probs,gt_classes,num_classes)
    else:
        print("ERROR: unknown cfg.Subtask {} in [lib/datasets/evaluate/cls_utils.py: loadModelCls]".format(cfg.SUBTASK))
        sys.exit()

def compute_metrics_v2(ovthresh,image_ids,model_output_cls,gt_classes,num_classes):
    nd = len(image_ids)
    correct = np.zeros((nd,len(ovthresh))) # "false negative"
    for d in range(nd):
        CLS_GT = gt_classes[image_ids[d]] # should be size 1
        for idx in range(len(ovthresh)):

            # thresh = ovthresh[idx]
            # # get a guess
            # guess = 0
            # if prob >= thresh: guess = 1
            # record if correct
            guess = model_output_cls[d]
            if guess == CLS_GT: correct[d,idx] = 1
    acc = np.mean(correct,axis=0)
    print(acc)
    print("Accuracy: {}".format(acc[0]))
    print("DONE")
    sys.exit()
    return correct,None,None,None


def compute_metrics_v1(ovthresh,image_ids,model_probs,gt_classes):
    nd = len(image_ids)
    tp = np.zeros((nd,len(ovthresh))) # "true positive"
    tn = np.zeros((nd,len(ovthresh))) # "true negative"
    fp = np.zeros((nd,len(ovthresh))) # "false positive"
    fn = np.zeros((nd,len(ovthresh))) # "false negative"
    print(len(gt_classes.keys()))
    print(gt_classes)
    print(image_ids)

    for d in range(nd):
        CLS_GT = gt_classes[image_ids[d]] # should be size 1
        if model_probs[d].size == 1:
            prob = model_probs[d]
        else:
            prob = model_probs[d, :].astype(float) # should be size 1
        if cfg._DEBUG.datasets.evaluators.bbox_utils: print("[compute_TP_FP]",bb,BBGT)
        
        for idx in range(len(ovthresh)):
            thresh = ovthresh[idx]
            assert (0 < thresh and thresh < 1), "thresh in (0,1)"

            guess = 0
            if prob >= thresh: guess = 1

            if guess == CLS_GT and guess == 1: tp[d,idx] = 1
            if guess == CLS_GT and guess == 0: tn[d,idx] = 1
            if guess != CLS_GT and guess == 1: fp[d,idx] = 1
            if guess != CLS_GT and guess == 0: fn[d,idx] = 1


    return tp, tn, fp, fn










