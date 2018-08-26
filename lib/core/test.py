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
from utils.misc import getRotationScale,toRadians,getRotationInfo,print_net_activiation_data
import numpy as np
import cv2
import caffe
from fast_rcnn.nms_wrapper import nms
import cPickle
from utils.blob import im_list_to_blob
import os

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
            im_rotate_factors.append([cfg.ROTATE_IMAGE,cols,rows])
        if cfg.SSD == True:
            im_scale_factors.append([im_scale_x,im_scale_y])
        else:
            im_scale_factors.append(im_scale_x)
        if cfg._DEBUG.core.test: print("[post-process] im.shape",im.shape)
        processed_ims.append(im)

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
    if not cfg.TEST.OBJ_DET.HAS_RPN:
        blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors,im_rotate_factors

def im_detect(net, im, boxes=None):
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
    blobs, im_scales, im_rotates = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.OBJ_DET.DEDUP_BOXES > 0 and not cfg.TEST.OBJ_DET.HAS_RPN:
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
    elif cfg.SSD is False:
        net.blobs['rois'].reshape(*(blobs['rois'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    if cfg.TEST.OBJ_DET.HAS_RPN and cfg.SSD is False:
        forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    elif cfg.SSD is False:
        forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)

    blobs_out = net.forward(**forward_kwargs)

    if cfg.TEST.OBJ_DET.HAS_RPN and cfg.SSD is False:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        rois = net.blobs['rois'].data.copy()
        # unscale back to raw image space
        boxes = rois[:, 1:5] / im_scales[0]
    elif cfg.SSD is True:
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
    elif cfg.SSD is False:
        # use softmax estimated probabilities
        scores = blobs_out['cls_prob']
    elif cfg.SSD is True:
        scores = np.zeros((len(boxes),201))
        for row in range(len(rois[0,0,:,2])):
            scores[row,rois[0,0,row, 1].astype(int)] = rois[0,0,row, 2]

    if cfg.TEST.OBJ_DET.BBOX_REG and cfg.SSD is False:
        # Apply bounding-box regression deltas
        box_deltas = blobs_out['bbox_pred']
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)
    elif cfg.SSD is True:
        box_deltas = np.zeros((len(boxes),804)) ##CHANGE IF DIFF NUM OF CLASS
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.OBJ_DET.DEDUP_BOXES > 0 and not cfg.TEST.OBJ_DET.HAS_RPN:
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

    return scores, pred_boxes, im_rotates

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

def test_net(net, imdb, max_per_image=100, thresh=1/80., vis=False):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, net)

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    if not cfg.TEST.OBJ_DET.HAS_RPN:
        roidb = imdb.roidb

    im_rotates_all = dict.fromkeys(imdb.image_index)

    for i in xrange(num_images):
        # filter out any ground truth boxes
        if cfg.TEST.OBJ_DET.HAS_RPN:
            box_proposals = None
        else:
            # The roidb may contain ground-truth rois (for example, if the roidb
            # comes from the training or val split). We only want to evaluate
            # detection on the *non*-ground-truth rois. We select those the rois
            # that have the gt_classes field set to 0, which means there's no
            # ground truth.
            box_proposals = roidb[i]['boxes'][roidb[i]['gt_classes'] == 0]

        im = cv2.imread(imdb.image_path_at(i))
        _t['im_detect'].tic()
        scores, boxes, im_rotates = im_detect(net, im, box_proposals)
        # print("image id: {}".format(imdb.image_index[i]))
        # print_net_activiation_data(net,["data","conv1_2","rpn_cls_prob_reshape","rois"])

        if cfg._DEBUG.core.test: print(imdb.image_index[i])
        _t['im_detect'].toc()

        _t['misc'].tic()
        # skip j = 0, because it's the background class
        im_rotates_all[imdb.image_index_at(i)] = im_rotates

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
        _t['misc'].toc()

        # print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
        #       .format(i + 1, num_images, _t['im_detect'].average_time,
        #               _t['misc'].average_time)

    detection_object = {}
    detection_object["all_boxes"] = all_boxes
    detection_object["im_rotates_all"] = im_rotates_all

    det_file = os.path.join(output_dir, 'detections.pkl')
    print(det_file)
    with open(det_file, 'wb') as f:
        cPickle.dump(detection_object, f, cPickle.HIGHEST_PROTOCOL)

    # print(len(all_boxes))
    # for i in range(len(all_boxes)):
    #     n = 0
    #     for j in range(len(all_boxes[i])):
    #         n += len(all_boxes[i][j])
    #     print("{}: {}".format(i,n))

    #return all_boxes,output_dir
    print 'Evaluating detections'
    imdb.evaluate_detections(detection_object, output_dir)
    
