# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Blob helper functions."""

from core.config import cfg
from datasets.ds_utils import cropImageToAnnoRegion
import numpy as np
import cv2
import matplotlib.pyplot as plt

def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)

    for i in xrange(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
        img = blob[i]
    
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob

def blob_list_im(blobs):
    """Convert a list of blobs into a images

    Assumes blobs are already prepared (means are NOT subtracted tho, BGR order, ...).
    """
    max_shape = np.array([blob.shape for blob in blobs]).max(axis=0)
    num_blobs = len(blobs)
    if len(max_shape) == 1:
        sqrt_shape = np.sqrt(max_shape[0]/3)
        max_shape = [0,sqrt_shape,sqrt_shape]
    imgs = np.zeros((num_blobs, 3, max_shape[1], max_shape[2]),
                    dtype=np.float32)
    for i in xrange(num_blobs):
        blob = blobs[i]
        imgs[i, :, 0:max_shape[1], 0:max_shape[2]] = blob.reshape(imgs[i].shape)
    # Move channels (axis 1) to axis 3
    # Axis order will become: (batch elem, height, width, channel)
    channel_swap = (0, 2, 3, 1)
    imgs = imgs.transpose(channel_swap)
    return imgs

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors,im_rotate_factors = _get_image_blob(im,None)
    # elif cfg.TASK == 'classification':
    #     blobs['data'], im_scale_factors,im_rotate_factors = _get_cropped_image_blob(im)
    if not cfg.TEST.OBJ_DET.HAS_RPN and cfg.TASK == 'object_detection':
        blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors,im_rotate_factors

def _get_blobs_from_roidb(roidb, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors,im_rotate_factors = _get_image_blob_from_roidb(roidb,None)
    # elif cfg.TASK == 'classification':
    #     blobs['data'], im_scale_factors,im_rotate_factors = _get_cropped_image_blob(im)
    if not cfg.TEST.OBJ_DET.HAS_RPN and cfg.TASK == 'object_detection':
        blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors,im_rotate_factors

def _get_image_blob_from_roidb(roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    # we dont return rotation currently
    num_images = len(roidb)
    if scale_inds is None:
        scale_inds = np.zeros(num_images).astype(np.uint8)
    processed_ims = []
    im_scales = []
    for i in xrange(num_images):
        im = cv2.imread(roidb[i]['image'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                        cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales, None

def _get_image_blob(im,scale_inds):
    """
    scale_inds included to work between core/test.py 
    and [cls_data_layer/minibatch.py,
    roi_data_layer/minibatch.py,
    alcls_data_layer/minibatch.py]
    """

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

    if cfg.TASK == 'object_detection':
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
    elif cfg.TASK == 'classification':
        newSize = (cfg.CROPPED_IMAGE_SIZE,cfg.CROPPED_IMAGE_SIZE)
        im = cv2.resize(im_orig, newSize, None,interpolation=cv2.INTER_LINEAR)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors),im_rotate_factors

def _get_raw_image_blob(roidb, records, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    return getRawCroppedImageBlob(roidb, records, scale_inds,False)

def _get_cropped_image_blob(roidb, records, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    return getRawCroppedImageBlob(roidb, records, scale_inds,True)

def getRawImageBlob(roidb,records,scale_inds):
    return getRawCroppedImageBlob(roidb,records,scale_inds,False)
    
def getCroppedImageBlob(roidb,records,scale_inds):
    return getRawCroppedImageBlob(roidb,records,scale_inds,True)

def getRawCroppedImageBlob(roidb,records,scale_inds,getCropped):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for idx, roi in enumerate(roidb):
        im = cv2.imread(roi['image'])
        if roi['flipped']:
            im = im[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[0]
        cimg = im
        if getCropped:
            cimg = cropImageToAnnoRegion(im,roi['boxes'][0]) # always the first box since we are *flattened*
        target_size = cfg.TRAIN.SCALES[0]
        cimg, cimg_scale = prep_im_for_blob(cimg, cfg.PIXEL_MEANS, target_size,
                                        cfg.TRAIN.MAX_SIZE)
        cimg_scale = 1.0 # why always "1.0"??
        processed_ims.append(cimg)
        im_scales.append(cimg_scale)
    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales
    
def prep_im_for_blob(im, pixel_means, target_size, max_size):
    """Mean subtract and scale an image for use in a blob."""
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)

    return im, im_scale

def prep_im_for_vae_blob(im, pixel_means, target_size, max_size):
    # disregard the asepct ratio
    # assume target size for all axis
    """Mean subtract and scale an image for use in a blob."""
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape
    #print("[utils/prep_im_for_vae_blob]: im.shape",im.shape)
    im_scale = [0,0]
    im_scale[0] = float(target_size) / float(im_shape[0])
    im_scale[0] = float(target_size) / float(im_shape[1])

    im = cv2.resize(im, (target_size,target_size),
                    interpolation=cv2.INTER_LINEAR)

    return im, im_scale

def save_blob_list_to_file(blob_list,append_str_l,vis=False):
    print("[./utils/blob.py: save_blob_list_to_file]: saving images")
    imgs = blob_list_im(blob_list)
    useAppendStr = append_str_l is not None and len(append_str_l) == imgs.shape[0]
    for idx,img in enumerate(imgs):
        img[:,:,:] *= 255
        img[:cfg.CROPPED_IMAGE_SIZE,:cfg.CROPPED_IMAGE_SIZE,:] += cfg.PIXEL_MEANS
        img = img.astype(np.uint8)
        if useAppendStr:
            fn = "save_blob_list_image_{}_{}.png".format(idx,append_str_l[idx])
        else:
            fn = "save_blob_list_image_{}.png".format(idx)
        if vis is False:
            cv2.imwrite(fn,img)
        else:
            plt.imshow(img[:,:,::-1])
            plt.show()

def createInfoBlob(im_data,im_scales):
    # ensure normalization of image data
    if np.max(im_data) > 1: # assume this means we haven't normalized
        im_data /= 255 
    im_info = {}
    im_info['data'] = im_data
    im_info['scales'] = im_scales
    return im_info


