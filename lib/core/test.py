# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an imdb (image database)."""

from core.config import cfg, get_output_dir
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
import argparse
from utils.timer import Timer
from utils.misc import getRotationScale,toRadians,getRotationInfo,print_net_activiation_data,save_image_with_border,createAlReportHeader,transformField,openAlResultsCsv,startAlReport,computeEntopy
import numpy as np
import cv2
import caffe
from fast_rcnn.nms_wrapper import nms
import cPickle
from utils.blob import im_list_to_blob,blob_list_im,save_blob_list_to_file
from datasets.ds_utils import cropImageToAnnoRegion
from alcls_data_layer.minibatch import get_minibatch as alcls_get_minibatch
import os
import matplotlib.pyplot as plt

cfg._DEBUG.core.test = False

def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []
    im_rotate_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale_x = float(target_size) / float(im_size_min)
        im_scale_y = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale_x * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale_x = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
            im_scale_y = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        if cfg.SSD == True:
            im_scale_x = float(cfg.SSD_img_size) / float(im_shape[1])
            im_scale_y = float(cfg.SSD_img_size) / float(im_shape[0])

        im = cv2.resize(im_orig, None, None, fx=im_scale_x, fy=im_scale_y,
                        interpolation=cv2.INTER_LINEAR)
        M = None
        if cfg._DEBUG.core.test: print("[pre-process] im.shape",im.shape)
        if cfg.ROTATE_IMAGE != -1:
        #if cfg.ROTATE_IMAGE !=  0:
            rows,cols = im.shape[:2]
            if cfg._DEBUG.core.test: print("cols,rows",cols,rows)
            rotationMat, scale = getRotationInfo(cfg.ROTATE_IMAGE,cols,rows)
            im = cv2.warpAffine(im,rotationMat,(cols,rows),scale)
            im_rotate_factors.append([cfg.ROTATE_IMAGE,cols,rows,im_shape])
        if cfg.SSD == True:
            im_scale_factors.append([im_scale_x,im_scale_y])
        else:
            im_scale_factors.append(im_scale_x)
        if cfg._DEBUG.core.test: print("[post-process] im.shape",im.shape)
        processed_ims.append(im)

    # if cfg.TASK == 'object_detection':
    #     for target_size in cfg.TEST.SCALES:
    #         im_scale_x = float(target_size) / float(im_size_min)
    #         im_scale_y = float(target_size) / float(im_size_min)
    #         # Prevent the biggest axis from being more than MAX_SIZE
    #         if np.round(im_scale_x * im_size_max) > cfg.TEST.MAX_SIZE:
    #             im_scale_x = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    #             im_scale_y = float(cfg.TEST.MAX_SIZE) / float(im_size_max)

    #         if cfg.SSD == True:
    #             im_scale_x = float(cfg.SSD_img_size) / float(im_shape[1])
    #             im_scale_y = float(cfg.SSD_img_size) / float(im_shape[0])
    #         im = cv2.resize(im_orig, None, None, fx=im_scale_x, fy=im_scale_y,
    #                         interpolation=cv2.INTER_LINEAR)
    #         M = None
    #         if cfg._DEBUG.core.test: print("[pre-process] im.shape",im.shape)
    #         if cfg.ROTATE_IMAGE != -1:
    #         #if cfg.ROTATE_IMAGE !=  0:
    #             rows,cols = im.shape[:2]
    #             if cfg._DEBUG.core.test: print("cols,rows",cols,rows)
    #             rotationMat, scale = getRotationInfo(cfg.ROTATE_IMAGE,cols,rows)
    #             im = cv2.warpAffine(im,rotationMat,(cols,rows),scale)
    #             im_rotate_factors.append([cfg.ROTATE_IMAGE,cols,rows,im_shape])
    #         if cfg.SSD == True:
    #             im_scale_factors.append([im_scale_x,im_scale_y])
    #         else:
    #             im_scale_factors.append(im_scale_x)
    #         if cfg._DEBUG.core.test: print("[post-process] im.shape",im.shape)
    #         processed_ims.append(im)
    # elif cfg.TASK == 'classification':
    #     for target_size in cfg.TEST.SCALES:
    #         newSize = (cfg.CROPPED_IMAGE_SIZE,cfg.CROPPED_IMAGE_SIZE)
    #         # if im_orig.shape[0] > 1: print("HOLLA! The batch size for testing is > 1. Not good.")
    #         # NOTE: we assume "im_orig" is just ONE image with (#pixels_across,#pixel_verticle,3 colors)
    #         im = cv2.resize(im_orig, newSize, None,interpolation=cv2.INTER_LINEAR)
    #         processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors),im_rotate_factors


def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors,im_rotate_factors = _get_image_blob(im)
    if not cfg.TEST.OBJ_DET.HAS_RPN and cfg.TASK == 'object_detection':
        blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors,im_rotate_factors

def im_detect(net, im, boxes=None, image_id="",isImBlob=False):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals or None (for RPN)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    if cfg._DEBUG.core.test: print("im.shape",im.shape)
    # print("[./core/test.py: im_detect]")

    if isImBlob:
        blobs = im
        im_scales = None
        im_rotates = None
        if 'scales' in im.keys(): im_scales = im['scales']
        if 'rotates' in im.keys(): im_rotates = im['rotates']
    else:
        blobs, im_scales, im_rotates = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.OBJ_DET.DEDUP_BOXES > 0 and not cfg.TEST.OBJ_DET.HAS_RPN and cfg.TASK == 'object_detection':
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.OBJ_DET.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]

    if cfg.TEST.OBJ_DET.HAS_RPN and cfg.SSD is False:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)

    # reshape network inputs
    if cfg.SSD is False:
        net.blobs['data'].reshape(*(blobs['data'].shape))

    if cfg.TEST.OBJ_DET.HAS_RPN and cfg.SSD is False:
        net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    elif cfg.SSD is False and cfg.TASK == 'object_detection':
        net.blobs['rois'].reshape(*(blobs['rois'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    if cfg.TEST.OBJ_DET.HAS_RPN and cfg.SSD is False:
        forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    elif cfg.SSD is False and cfg.TASK == 'object_detection':
        forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)

    if cfg._DEBUG.core.test:
        im_shape = forward_kwargs["data"].shape
        rimg = blob_list_im(forward_kwargs["data"])
        fn = "input_image_{}_{}.png".format(cfg.ROTATE_IMAGE,image_id)
        #save_image_with_border(fn,rimg[0],rotation=im_rotates[0])
    if len(cfg.SAVE_ACTIVITY_VECTOR_BLOBS) > 0:
        forward_kwargs['blobs'] = cfg.SAVE_ACTIVITY_VECTOR_BLOBS

    blobs_out = net.forward(**forward_kwargs)

    if cfg.TEST.OBJ_DET.HAS_RPN and cfg.SSD is False and cfg.TASK == 'object_detection':
        assert len(im_scales) == 1, "Only single-image batch implemented"
        rois = net.blobs['rois'].data.copy()
        # unscale back to raw image space
        boxes = rois[:, 1:5] / im_scales[0]
    elif cfg.SSD is True and cfg.TASK == 'object_detection':
        #assert len(im_scales) == 1, "Only single-image batch implemented"
        rois = net.blobs['detection_out'].data.copy()
        # unscale back to raw image space
        boxes = rois[0,0,:, 3:] * cfg.SSD_img_size
        boxes[:,0] = boxes[:,0] / im_scales[0][0]
        boxes[:,1] = boxes[:,1] / im_scales[0][1]
        boxes[:,2] = boxes[:,2] / im_scales[0][0]
        boxes[:,3] = boxes[:,3] / im_scales[0][1]

    if cfg.TEST.OBJ_DET.SVM:
        # use the raw scores before softmax under the assumption they
        # were trained as linear SVMs
        scores = net.blobs['cls_score'].data
    elif cfg.SSD is False or cfg.TASK == 'classification':
        # use softmax estimated probabilities
        scores = blobs_out[cfg.CLS_PROBS] # default is "cls_prob"
    elif cfg.SSD is True and cfg.TASK == 'object_detection':
        scores = np.zeros((len(boxes),201))
        for row in range(len(rois[0,0,:,2])):
            scores[row,rois[0,0,row, 1].astype(int)] = rois[0,0,row, 2]

    pred_boxes = []
    if cfg.TEST.OBJ_DET.BBOX_REG and cfg.SSD is False and cfg.TASK == 'object_detection':
        # Apply bounding-box regression deltas
        box_deltas = blobs_out['bbox_pred']
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)
    elif cfg.SSD is True and cfg.TASK == 'object_detection':
        box_deltas = np.zeros((len(boxes),804)) ##CHANGE IF DIFF NUM OF CLASS
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)
    elif cfg.TASK == 'object_detection':
        # Simply repeat the boxes, once for each class
        print("\n\n\n\nDANGER DANGER! WE SHOULD NOT USE THIS\n\n\n\n\n")
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.OBJ_DET.DEDUP_BOXES > 0 and not cfg.TEST.OBJ_DET.HAS_RPN and cfg.TASK == 'object_detection':
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]

    # we don't want to fix the original image orientation since the bboxes
    # are predicted for the rotated version. instead we correct this issue
    # by rotating the groundtruth
    # # rotate boxes back to the original image orientation...
    # if cfg.ROTATE_IMAGE:
    #     # i think it's just one right now...
    #     M = im_rotates[0]
    #     Minv = cv2.invertAffineTransform(M)
    # print(pred_boxes.shape)

    activity_vectors = {}
    if len(cfg.SAVE_ACTIVITY_VECTOR_BLOBS) > 0:
        for blob_name in cfg.SAVE_ACTIVITY_VECTOR_BLOBS:
            # if blob_name not in activity_vectors.keys(): activity_vectors[blob_name] = []
            # above is always true since dict should start empty
            # here we know it is only one image id.
            activity_vectors[blob_name] = blobs_out[blob_name].astype(np.float32, copy=True)

    return scores, pred_boxes, im_rotates, activity_vectors


def vis_detections(im, class_name, dets, thresh=0.3):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
    for i in xrange(np.minimum(10, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > thresh:
            plt.cla()
            plt.imshow(im)
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='g', linewidth=3)
                )
            plt.title('{}  {:.3f}'.format(class_name, score))
            plt.show()

def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in xrange(num_classes):
        for im_ind in xrange(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            # CPU NMS is much faster than GPU NMS when the number of boxes
            # is relative small (e.g., < 10k)
            # TODO(rbg): autotune NMS dispatch
            keep = nms(dets, thresh, force_cpu=True)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes

def loadImage(imdb,image_path,image_index,bbox,al_net,idx):
    isImBlob = False
    if imdb.is_image_index_flattened and al_net is None:
        raw_img = cv2.imread(image_path)
        img = cropImageToAnnoRegion(raw_img,bbox)
    elif imdb.is_image_index_flattened and al_net is not None:
        sampleRoidb = [imdb.roidb[idx]]
        sampleRoidb[0]['image'] = image_path
        sampleRecords = []
        numClasses = len(cfg.DATASETS.CLASSES)
        input_data = alcls_get_minibatch(sampleRoidb,sampleRecords,al_net,numClasses)
        img = {}
        img['data'] = input_data['data']
        isImBlob = True
    else:
        img = cv2.imread(image_path)
    return img,isImBlob

def test_net(net, imdb, max_per_image=100, thresh=1/80., vis=False, al_net=None):
    """Test a Fast R-CNN network on an image database."""
    roidb = imdb.roidb

    """
    TODO: the image id's don't align because we load in the entire image -- not a cropped image.
    We need to load cropped images.
    """

    
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    ds_av = {}
    if cfg.TASK == 'object_detection':
        all_boxes = [[[] for _ in xrange(num_images)]
                     for _ in xrange(imdb.num_classes)]
        all_items = all_boxes
    elif cfg.TASK == 'classification':
        all_probs = [[-1 for _ in xrange(num_images)]
                     for _ in xrange(imdb.num_classes)]
        all_items = all_probs

    output_dir = get_output_dir(imdb, net)

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}
    
    # information to generate Active Learning Report
    fidAlReport = None
    pctErrorRed = None
    if cfg.ACTIVE_LEARNING.REPORT:
        pctErrorRed = openAlResultsCsv()
        fidAlReport = startAlReport(imdb,net)


    im_rotates_all = dict.fromkeys(imdb.image_index)

    print("num_images: {}".format(num_images))
    for i in xrange(num_images):
        # filter out any ground truth boxes
        if cfg.TEST.OBJ_DET.HAS_RPN or cfg.TASK != 'object_detection':
            box_proposals = None

        # always send in ['boxes'][0] since we want the first box in the index; we are flattened if we use 'boxes'
        im,isImBlob = loadImage(imdb,imdb.image_path_at(i),imdb.image_index[i],imdb.roidb[i]['boxes'][0],al_net,i)
        # save_blob_list_to_file(im,None,vis=True)

        _t['im_detect'].tic()
        scores, boxes, im_rotates, activity_vectors = im_detect(net, im, box_proposals,\
                                                                imdb.image_index[i],isImBlob=isImBlob)
        # print("image id: {}".format(imdb.image_index[i]))
        # print_net_activiation_data(net,["data","conv1_2","rpn_cls_prob_reshape","rois"])

        if cfg._DEBUG.core.test: print(imdb.image_index[i])
        _t['im_detect'].toc()

        _t['misc'].tic()
        # print("boxes.shape",boxes.shape)
        # sys.exit()

        # skip j = 0, because it's the background class
        im_rotates_all[imdb.image_index_at(i)] = im_rotates

        if len(cfg.SAVE_ACTIVITY_VECTOR_BLOBS) > 0:
            aggregateAV(ds_av,activity_vectors,imdb.image_index[i])
        if cfg.TASK == 'object_detection':
            aggregateDetections(imdb,scores,boxes,all_items,thresh,i,im,vis,max_per_image)
        elif cfg.TASK == 'classification':
            aggregateClassification(imdb,scores,all_items,i)

        if cfg.ACTIVE_LEARNING.REPORT:
            recordImageForAlReport(imdb,scores,activity_vectors,fidAlReport,i,pctErrorRed)
            
        _t['misc'].toc()
        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time)

    if cfg.SAVE_ACTIVITY_VECTOR_BLOBS:
        dirn = cfg.GET_SAVE_ACTIVITY_VECTOR_BLOBS_DIR()
        print("activity vectors saved @")
        for blob_name,av in ds_av.items():
            fn = os.path.join(dirn,"{}.pkl".format(blob_name))
            print(fn)
            with open(fn, 'wb') as f:
                cPickle.dump(av, f, cPickle.HIGHEST_PROTOCOL)

    save_dict = {}
    if cfg.TASK == 'object_detection':
        save_dict["all_boxes"] = all_items
        save_dict["im_rotates_all"] = im_rotates_all
        det_file = os.path.join(output_dir, 'detections.pkl')
    elif cfg.TASK == 'classification':
        save_dict["all_probs"] = all_items
        save_dict["im_rotates_all"] = im_rotates_all
        det_file = os.path.join(output_dir, 'probs.pkl')

    with open(det_file, 'wb') as f:
        cPickle.dump(save_dict, f, cPickle.HIGHEST_PROTOCOL)

    print 'Evaluating detections'
    imdb.evaluate_detections(save_dict, output_dir)
    
def aggregateDetections(imdb,scores,boxes,all_boxes,thresh,i,im,vis,max_per_image):

    for j in xrange(1, imdb.num_classes):
        inds = np.where(scores[:, j] > thresh)[0]
        cls_scores = scores[inds, j]
        cls_boxes = boxes[inds, j*4:(j+1)*4]
        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
            .astype(np.float32, copy=False)
        keep = nms(cls_dets, cfg.TEST.OBJ_DET.NMS)
        cls_dets = cls_dets[keep, :]
        if vis:
            vis_detections(im, imdb.classes[j], cls_dets)
        all_boxes[j][i] = cls_dets

    # Limit to max_per_image detections *over all classes*
    if max_per_image > 0:
        image_scores = np.hstack([all_boxes[j][i][:, -1]
                                  for j in xrange(1, imdb.num_classes)])
        if len(image_scores) > max_per_image:
            image_thresh = np.sort(image_scores)[-max_per_image]
            for j in xrange(1, imdb.num_classes):
                keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                all_boxes[j][i] = all_boxes[j][i][keep, :]


def aggregateAV(ds_av,activity_vectors,image_id):
    for blob_name,blob_av in activity_vectors.items():
        if blob_name not in ds_av.keys(): ds_av[blob_name] = {}
        ds_av[blob_name][image_id] = blob_av

def aggregateClassification(imdb,scores,all_items,i):
    scores = np.squeeze(scores)
    if imdb.num_classes == 1:
        all_items[0][i] = float(scores[0])
    else:
        for j in xrange(0, imdb.num_classes):
            all_items[j][i] = float(scores[j])
    
def recordImageForAlReport(imdb,scores,activity_vectors,fidAlReport,i,pctErrorRed):
    scores = np.squeeze(scores)
    image_index = imdb.image_index[i]
    imageStr = "{},".format(image_index)
    for score in scores:
        imageStr += "{:.5f},".format(score)
    scoreEntropy = computeEntopy(scores)
    imageStr += "{:.5f},".format(scoreEntropy)
    for idx,av_blobNamed in enumerate(cfg.SAVE_ACTIVITY_VECTOR_BLOBS):
        avValue = transformField(activity_vectors[idx])
        imageStr += "{:.5f},".format(avValue)
    imageStr += "{:.5f}\n".format(pctErrorRed[image_index]['ave'])
    fidAlReport.write(imageStr)

    


