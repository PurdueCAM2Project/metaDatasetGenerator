"""
This evalutates classification methods
"""

import os,sys
import os.path as osp
import PIL
import numpy as np
import scipy.sparse
from core.config import cfg,cfgData
from easydict import EasyDict as edict
from cls_utils import *


class classificationEvaluator(object):
    """Image database."""

    def __init__(self, datasetName, classes, compID, salt, cacheDir,
                 imageSetPath, imageIndex, annoPath,load_annotation,
                 onlyCls=None):
        self._datasetName = datasetName
        self._classes = classes
        self._comp_id = compID
        self._salt = salt
        self._cachedir = cacheDir
        self.image_index = imageIndex
        self._annoPath = annoPath
        self._load_annotation = load_annotation
        self._imageSetPath = imageSetPath
        self._imageSet = imageSetPath.split("/")[-1].split(".")[0] # "...asdf/imageSet.txt"
        self._onlyCls = onlyCls

    def evaluate_detections(self, classification_object, output_dir):
        # num_outpus = 
        # if self._classes != num_outputs:
        #     raise ValueError("ERROR: the classes and output size don't match!")
        all_probs = classification_object['all_probs']
        self._pathResults = output_dir
        if cfg.SUBTASK == "tp_fn":
            self._write_results_file(all_probs)
            self._do_python_eval(output_dir)
        elif cfg.SUBTASK in ["default","al_subset"]:
            self._write_softmax_results_file(all_probs)
            self._do_cls_python_eval(output_dir)
        else:
            print("ERROR: cfg.SUBTASK = {} is unknown".format(cfg.SUBTASK))
            sys.exit()

    def _write_softmax_results_file(self, all_probs):
        filename = self._get_results_file_template().format("all")
        max_indices = np.argmax(all_probs,axis=0)
        max_values = np.max(all_probs,axis=0)
        with open(filename, 'wt') as f:
            for image_index,max_index,max_value in zip(self.image_index,max_indices,max_values):
                f.write('{:s} {:d} {:.3f}\n'.format(image_index,max_index,max_value))

    def _write_results_file(self, all_probs):
        for cls_ind, cls in enumerate(self._classes):
            count = 0
            skip_count = 0
            det_count = 0
            if cls == '__background__':
                continue
            print('Writing {} {} results file'.format(cls,self._datasetName))
            filename = self._get_results_file_template().format(cls)
            print(filename)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    if cfg._DEBUG.datasets.evaluators.bboxEvaluator: print(im_ind,index)
                    probs = all_probs[cls_ind][im_ind]
                    f.write('{:s}'.format(index))
                    if type(probs) is float or type(probs) is int:
                        f.write(' {:.3f}\n'.format(probs))
                    else:
                        for k in xrange(probs.shape[0]):
                            f.write(' {:.3f}'.format(probs[k]))
                        f.write('\n')

    def _do_cls_python_eval(self, output_dir = 'output',rotations = None):
        annopath = self._annoPath + "/{:s}"
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        cls = "all"
        detfile = self._get_results_file_template().format(cls)
        suffix = self._get_experiment_suffix_template().format(cls)
        acc, rec, prec, ovthresh = self.cls_eval(
            detfile, annopath, self._imageSetPath, cls, self._cachedir, suffix, \
            ovthresh=0.5, use_07_metric=use_07_metric, rotations = rotations)
        print("\n\n\n\n\n")
        print("IDK what to do from here")
        print("\n\n\n\n\n")
        sys.exit()
        infix = "faster-rcnn"
        if cfgData.MODEL:
            infix = cfgData.MODEL
        if cfg.ROTATE_IMAGE != -1:
            infix += "_{}".format(cfg.ROTATE_IMAGE)
        results_filename = "./results_{}_{}_{}.txt".format(infix,self._datasetName, self._salt)
        results_fd = open(results_filename,"w")
        for kdx in range(len(ovthresh)):
            #print('{0:.3f}@{1:.2f}'.format(ap[kdx],ovthresh[kdx]))
            print('Mean AP = {:.4f} @ {:.2f}'.format(np.mean(aps[:,kdx]),ovthresh[kdx]))
        print('~~~~~~~~')
        print('Results:')
        count_ = 1
        sys.stdout.write('{0:>15} (#):'.format("class AP"))
        results_fd.write('{0:>15} (#):'.format("class AP"))
        for thsh in ovthresh:
            sys.stdout.write("\t{:>5}{:.3f}".format("@",thsh))
            results_fd.write("\t{:>5}{:.3f}".format("@",thsh))
        sys.stdout.write("\n")
        results_fd.write("\n")
        for ap in aps:
            sys.stdout.write('{:>15} ({}):'.format(self._classes[count_],count_))
            results_fd.write('{:>15} ({}):'.format(self._classes[count_],count_))
            for kdx in range(len(ovthresh)):
                sys.stdout.write('\t{0:>10.5f}'.format(ap[kdx],ovthresh[kdx]))
                results_fd.write('\t{0:>10.5f}'.format(ap[kdx],ovthresh[kdx]))
            sys.stdout.write('\n')
            results_fd.write('\n')
            count_ +=1
        sys.stdout.write('{:>15}:'.format("mAP"))
        results_fd.write('{:>15}:'.format("mAP"))
        for kdx in range(len(ovthresh)):
            sys.stdout.write('\t{:10.5f}'.format(np.mean(aps[:,kdx])))
            results_fd.write('\t{:10.5f}'.format(np.mean(aps[:,kdx])))
            #print('{0:.3f}@{1:.2f}'.format(ap[kdx],ovthresh[kdx]))
            #print('mAP @ {:.2f}: {:.5f} '.format(ovthresh[kdx],np.mean(aps[:,kdx])))
        sys.stdout.write('\n')
        results_fd.write('\n')
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('--------------------------------------------------------------')
        print('')
        results_fd.close()

        # remove the results since I don't know how to "not write" with the results_fd variable.
        if not cfg.WRITE_RESULTS: os.remove(results_filename)
        
    def _do_python_eval(self, output_dir = 'output',rotations = None):
        annopath = self._annoPath + "/{:s}"
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not osp.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            if self._onlyCls is not None and cls != self._onlyCls:
                continue
            detfile = self._get_results_file_template().format(cls)
            suffix = self._get_experiment_suffix_template().format(cls)
            rec, prec, ap, ovthresh = self.cls_eval(
                detfile, annopath, self._imageSetPath, cls, self._cachedir, suffix, \
                ovthresh=0.5, use_07_metric=use_07_metric, rotations = rotations)
            aps += [ap]
        aps = np.array(aps)
        infix = "faster-rcnn"
        if cfgData.MODEL:
            infix = cfgData.MODEL
        if cfg.ROTATE_IMAGE != -1:
            infix += "_{}".format(cfg.ROTATE_IMAGE)
        results_filename = "./results_{}_{}_{}.txt".format(infix,self._datasetName, self._salt)
        results_fd = open(results_filename,"w")
        for kdx in range(len(ovthresh)):
            #print('{0:.3f}@{1:.2f}'.format(ap[kdx],ovthresh[kdx]))
            print('Mean AP = {:.4f} @ {:.2f}'.format(np.mean(aps[:,kdx]),ovthresh[kdx]))
        print('~~~~~~~~')
        print('Results:')
        count_ = 1
        sys.stdout.write('{0:>15} (#):'.format("class AP"))
        results_fd.write('{0:>15} (#):'.format("class AP"))
        for thsh in ovthresh:
            sys.stdout.write("\t{:>5}{:.3f}".format("@",thsh))
            results_fd.write("\t{:>5}{:.3f}".format("@",thsh))
        sys.stdout.write("\n")
        results_fd.write("\n")
        for ap in aps:
            sys.stdout.write('{:>15} ({}):'.format(self._classes[count_],count_))
            results_fd.write('{:>15} ({}):'.format(self._classes[count_],count_))
            for kdx in range(len(ovthresh)):
                sys.stdout.write('\t{0:>10.5f}'.format(ap[kdx],ovthresh[kdx]))
                results_fd.write('\t{0:>10.5f}'.format(ap[kdx],ovthresh[kdx]))
            sys.stdout.write('\n')
            results_fd.write('\n')
            count_ +=1
        sys.stdout.write('{:>15}:'.format("mAP"))
        results_fd.write('{:>15}:'.format("mAP"))
        for kdx in range(len(ovthresh)):
            sys.stdout.write('\t{:10.5f}'.format(np.mean(aps[:,kdx])))
            results_fd.write('\t{:10.5f}'.format(np.mean(aps[:,kdx])))
            #print('{0:.3f}@{1:.2f}'.format(ap[kdx],ovthresh[kdx]))
            #print('mAP @ {:.2f}: {:.5f} '.format(ovthresh[kdx],np.mean(aps[:,kdx])))
        sys.stdout.write('\n')
        results_fd.write('\n')
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('--------------------------------------------------------------')
        print('')
        results_fd.close()

        # remove the results since I don't know how to "not write" with the results_fd variable.
        if not cfg.WRITE_RESULTS: os.remove(results_filename)

    def _get_results_file_template(self):
        # example: VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._comp_id + "_" + self._get_experiment_suffix_template() + ".txt"
        path = osp.join(self._pathResults,filename)
        return path

    def _get_experiment_suffix_template(self):
        return 'cls_{:s}'

    def cls_eval(self,detpath,
                  annopath,
                  imagesetfile,
                  classname,
                  cachedir,
                  suffix,
                  ovthresh=0.5,
                  use_07_metric=False,
                  rotations = None):
        """rec, prec, ap = bbox_eval(detpath,
                                    annopath,
                                    imagesetfile,
                                    classname,
                                    [ovthresh],
                                    [use_07_metric])
        Top level function that does the BBOX evaluation.
        detpath: Path to detections
            detpath.format(classname) should produce the detection results file.
        annopath: Path to annotations
            annopath.format(imagename) should be the xml annotations file.
        imagesetfile: Text file containing the list of images, one image per line.
        classname: Category name (duh)
        cachedir: Directory for caching the annotations
        [ovthresh]: Overlap threshold (default = 0.5)
        [use_07_metric]: Whether to use BBOX07's 11 point AP computation
            (default False)
        """
        # assumes detections are in detpath.format(classname)
        # assumes annotations are in annopath.format(imagename)
        # assumes imagesetfile is a text file with each line an image name
        # cachedir caches the annotations in a pickle file

        # first load gt
        imagenames, annos = load_groundTruth(classname,cachedir,imagesetfile,annopath,
                                             self._load_annotation,self._classes)

        # extract gt objects for this class
        imagenames, gt_class_probs, npos, nneg = extractClassGroundTruth(imagenames,annos,classname)
        print("npos: {}".format(npos))
        print("nneg: {}".format(nneg))
        
        gt_image_ids = imagenames
        # read dets from model
        image_ids, model_probs = loadModelCls(detpath,classname)

        # number of detections (nd) from the model
        nd = len(image_ids)
        print("# of detections",nd)

        ovthresh = [0.5,0.75,0.95]
        tp, tn, fp, fn = compute_metrics(ovthresh,image_ids,model_probs,gt_class_probs,len(self._classes))
        # print(np.sum(tp))
        n_tp = np.sum(tp,axis=0)
        n_tn = np.sum(tn,axis=0)
        n_fp = np.sum(fp,axis=0)
        n_fn = np.sum(fn,axis=0)
        tpr = n_tp / (n_tp + n_fn)
        tnr = n_tn / (n_tn + n_fp)
        ppv = n_tp / (n_tp + n_fp)
        npv = n_tn / (n_tn + n_fn)
        fpr = n_fp / (n_fp + n_tn)
        fnr = n_fn / (n_fn + n_tp)
        print("n_tp",n_tp)
        print("n_tn",n_tn)
        print("n_fp",n_fp)
        print("n_fn",n_fn)
        print("----------------------------------------------")
        print("tpr",tpr)
        print("tnr",tnr)
        print("----------------------------------------------")
        print("ppv",ppv)
        print("npv",npv)
        print("----------------------------------------------")
        print("fpr",fpr)
        print("fnr",fnr)
        print("----------------------------------------------")

        if cfg._DEBUG.datasets.evaluators.bboxEvaluator: print(rec,prec)
        print("TODO: what do return?")
        return tp,tn,fp,fn









