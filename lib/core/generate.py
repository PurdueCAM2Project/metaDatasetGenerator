# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network."""

import caffe
from core.config import cfg
import roi_data_layer.roidb as rdl_roidb
import vae_data_layer.roidb as vae_rdl_roidb
from utils.timer import Timer
import numpy as np
import os,sys,cv2
from utils.blob import blob_list_im

from caffe.proto import caffe_pb2
import google.protobuf as pb2
import google.protobuf.text_format


class GenerateWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, net, output_dir, output_size):
        """Initialize the SolverWrapper."""
        self.output_dir = output_dir
        self.net = net
        self.current_sample_count = 0
        self.display = 2
        self.output_size = output_size

    def save_sample_set(self,imgs):
        for i in range(imgs.shape[0]):
            sq_img = np.squeeze(imgs[i])
            print(sq_img.shape)
            self.save_sample(sq_img)

    def save_sample(self,img):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (infix +
                    '_net_{:s}_{:d}'.format(self.net.name,self.current_sample_count) + '.png')
        filename = os.path.join(self.output_dir, filename)
        # save output as image
        if cfg.GENERATE.THRESHOLD:
            img = np.where(img > cfg.GENERATE.THRESHOLD,255,0)
            
        cv2.imwrite(filename,img)
        print('Wrote sample to: {:s}'.format(filename))
        
        self.current_sample_count += 1
        return filename

    def generate(self, numberOfSamples):
        """Network training loop."""
        last_snapshot_iter = -1
        timer = Timer()
        model_paths = []
        print(dir(self.net.layer_dict["sample"]))
        print(dir(self.net.layer_dict["sample"].blobs))

        while self.current_sample_count < numberOfSamples:
            # Make one SGD update
            timer.tic()
            
            blobs_out = self.net.forward()
            BATCH_SIZE = blobs_out["generated_images"].shape[0]
            #BATCH_SIZE = blobs_out["decode1neuron"].shape[0]
            blobs = blobs_out["generated_images"] * 255
            imgs = blob_list_im(blobs)
            # imgs = img.reshape(BATCH_SIZE,self.output_size,
            #                                           self.output_size,3) * 255
            print("imgs.shape",imgs.shape)
            
            self.save_sample_set(imgs)
            timer.toc()
            if self.current_sample_count % (10 * self.display) == 0:
                print('speed: {:.3f}s / iter'.format(timer.average_time))

        return model_paths

def generate_from_net(net, output_dir, output_size = 100, numberOfSamples=10):
    """Train *any* object detection network."""

    gw = GenerateWrapper(net, output_dir, output_size)
    print('Generating...')
    model_paths = gw.generate(numberOfSamples)
    print('done generating')
    return model_paths
