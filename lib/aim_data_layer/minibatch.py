# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""

import numpy as np
import numpy.random as npr
import cv2,sys
from core.config import cfg
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
from fast_rcnn.nms_wrapper import nms
from utils.blob import prep_im_for_blob, im_list_to_blob, _get_cropped_image_blob, save_blob_list_to_file, _get_blobs_from_roidb,getRawImageBlob,getCroppedImageBlob,createInfoBlob
from utils.misc import evaluate_image_detections
from datasets.ds_utils import cropImageToAnnoRegion
import matplotlib.pyplot as plt

def get_minibatch(roidb, records, al_net, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                    size=num_images)
    assert(cfg.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.TRAIN.BATCH_SIZE)

    # Get the input image blob, formatted for caffe
    # raw_im_data, raw_im_scales = getRawImageBlob(roidb, records, random_scale_inds)
    raw_im_data, raw_im_scales = getCroppedImageBlob(roidb, records, random_scale_inds)
    cropped_im_data, cropped_im_scales = getCroppedImageBlob(roidb, records, random_scale_inds)
    
    # prepare the image info for creating the image with activations
    raw_im_info = createInfoBlob(raw_im_data,raw_im_scales)
    cropped_im_info = createInfoBlob(cropped_im_data,cropped_im_scales)
    dataBlob = cropped_im_info['data']

    # create the image augmented with activations
    avImageBlob = prepareAlImageBlobFromRawImageBlob(raw_im_info,cropped_im_info,al_net)

    # visual inspection of output
    # save_blob_list_to_file(im_blobs,[elem['image_id'] for elem in roidb])

    # prepare for input into the classifier
    blobs = {'data': dataBlob}
    blobs = {'avImage': avImageBlob}
    blobs['labels'] = np.array(records)

    return blobs
    
def prepareAlImageBlobFromRawImageBlob(raw_im_info,cropped_im_info,al_net):

    raw_im_blob_shape = raw_im_info['data'].shape
    cropped_im_blob_shape = cropped_im_info['data'].shape

    # get the activations for each image
    batch_size = raw_im_blob_shape[0]
    output = network_forward_pass(al_net,raw_im_info,raw_im_info['scales'])

    # grab the activation values
    avImageBlob = np.zeros((batch_size,cfg.COLOR_CHANNEL,
                       cfg.AL_IMAGE_SIZE,cfg.AL_IMAGE_SIZE))
    imgSize = cfg.AL_IMAGE_SIZE
    avBatch = []
    for idx in range(cfg.BATCH_SIZE):
        avList = []
        for layer_name in cfg.AL_CLS.LAYERS:
            activation_values = output[idx][layer_name].copy()
            avList.append(activation_values.ravel())
        avImageBlob[idx,:,:,:] = formatActivationValueList(avList,imgSize)
    return avImageBlob

def formatActivationValueList(avList,imgSize):
    goalLength = imgSize * imgSize * cfg.COLOR_CHANNEL
    nLayers = len(avList)
    # get equal amount from each
    avListEqual = []
    for layerAv in avList:
        avListEqual.append(layerAv[:goalLength//nLayers])
    # aggregate into one numpy array
    avList = np.concatenate(avListEqual).ravel()
    # fill with zeros    
    toFill = goalLength - avList.size
    assert toFill >= 0, "[aim_data_layer/minibatch.py: formatActivationValueList] too many activation values. goal: {} v.s. current: {}".format(goalLength,avList.size)
    appendZeros = np.zeros(toFill)
    avImage = np.concatenate([avList,appendZeros])
    # reshape nad return
    return avImage.reshape(cfg.COLOR_CHANNEL,imgSize,imgSize)

def network_forward_pass(al_net,input_blobs,im_scales):
    output = []
    blobs = {}
    for idx in range(input_blobs['data'].shape[0]):
        im_blob = input_blobs['data'][idx,:][np.newaxis,:]
        blobs['data'] = im_blob
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[idx]]],
            dtype=np.float32)
        al_net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
        al_net.blobs['data'].reshape(*(blobs['data'].shape))
        forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
        forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
        forward_kwargs['blobs'] = cfg.AL_CLS.LAYERS
        al_out = al_net.forward(**forward_kwargs)
        output.append(al_out.copy())
    return output

def get_records(roidb,al_net):
    # this is KILLED because the FLATTENED roidb only contains ONE annotation per image
    # since each "image_id" is associated with (ORIGINAL_IMAGE_ID)_(BOUNDING_BOX_INDEX)
    # THEREFORE, we CAN'T evalutate in REAL TIME.

    # ^ wait we can since we only need the GT for the single box we care about...
    # lol i think this is okay but we don't use it so save issue for later.

    # print("[alcls_data_layer/minibatch.py: get_records]")

    # do forward pass of raw image
    im_blobs, im_scales, im_rotates = _get_blobs_from_roidb(roidb,None)
    output = network_forward_pass(al_net,im_blobs,im_scales)

    # extract relevant info
    box_deltas = output['bbox_pred']
    scores = output['cls_prob']
    rois = al_net.blobs['rois'].data.copy()
    im_shape = (cfg.CROPPED_IMAGE_SIZE,cfg.CROPPED_IMAGE_SIZE,cfg.COLOR_CHANNEL)
    # print(rois.shape)

    # compute the bounding boxes
    pre_boxes = rois[:, 1:5] / im_scales[0] # unscale back to raw image space
    pred_boxes = bbox_transform_inv(pre_boxes, box_deltas)
    boxes = clip_boxes(pred_boxes, im_shape)
    
    # print(boxes.shape)
    # print(scores.shape)
    # print(scores)

    #sort via scores
    max_per_image = 200
    sorted_ind = np.argsort(-scores)
    sorted_scores = np.sort(-scores)
    # print(sorted_scores)

    # cut off @ prob_thresh
    prob_thresh = 1./80.
    j = 1 # = cfg.PERSON_INDEX
    inds = np.where(scores[:, j] > prob_thresh)[0]
    cls_scores = scores[inds, j]
    cls_boxes = boxes[inds, j*4:(j+1)*4]
    
    # apply NMS
    cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                 .astype(np.float32, copy=False)
    keep = nms(cls_dets, prob_thresh, force_cpu=True)
    cls_dets = cls_dets[keep[:200], :] # cut off @ 200

    # evaluate the detections
    correct = evaluate_image_detections(roidb,cls_dets)

    return correct

def fillImageWithActivationValues(activation_values,im_blob,
                                  cropped_im_blob_shape,filledLimits):
    """
    CLEARLY, if we continue to pursue this project we MUST have a better
    program to fill the blob
    """
    
    av_vector = activation_values.ravel()

    # we need to scale the av_vector to within [0,1]... for now
    av_vector += min(av_vector)
    av_vector /= max(av_vector)

    start_x_im = filledLimits[0]
    stop_x_im = cfg.AL_IMAGE_SIZE
    start_y_im = filledLimits[1]
    stop_y_im = cropped_im_blob_shape[2]

    start_av = 0
    stop_av = (stop_x_im - start_x_im) * (stop_y_im - start_y_im) * 3
    # print(av_vector.shape)
    # print(im_blob[:,:,start_y_im:stop_y_im,start_x_im:stop_x_im].shape)
    # print(stop_av,len(av_vector))
    if stop_av > len(av_vector):
        # print("stop_av > len(av_vector)")
        # fix y; change x 
        len_av = len(av_vector)
        stop_x_im = len_av // (3 * (stop_y_im - start_y_im)) + start_x_im
        stop_av = (stop_y_im - start_y_im) * (stop_x_im - start_x_im) * 3
    im_blob[:,:,start_y_im:stop_y_im,start_x_im:stop_x_im] = av_vector[start_av:stop_av].reshape(1,3,stop_y_im - start_y_im,stop_x_im - start_x_im)
    return stop_y_im,stop_x_im
    

def _sample_rois(roidb, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # label = class RoI has max overlap with
    labels = roidb['max_classes']
    overlaps = roidb['max_overlaps']
    rois = roidb['boxes']

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(overlaps >= cfg.TRAIN.CLS.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(
                fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((overlaps < cfg.TRAIN.CLS.BG_THRESH_HI) &
                       (overlaps >= cfg.TRAIN.CLS.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image,
                                        bg_inds.size)
    # Sample foreground regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(
                bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    overlaps = overlaps[keep_inds]
    rois = rois[keep_inds]

    bbox_targets, bbox_inside_weights = _get_bbox_regression_labels(
            roidb['bbox_targets'][keep_inds, :], num_classes)

    return labels, overlaps, rois, bbox_targets, bbox_inside_weights

