import sys,os,pickle,uuid,cv2,glob,csv
import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
import numpy.random as npr
from core.config import cfg,iconicImagesFileFormat
from ntd.hog_svm import plot_confusion_matrix,appendHOGtoRoidb,split_data, scale_data,train_SVM,findMaxRegions, make_confusion_matrix
from datasets.ds_utils import computeTotalAnnosFromAnnoCount

class PreviousCounts():

    def __init__(self,size,initVal):
        self._prevCounts = [initVal for _ in range(size)]

    def __getitem__(self,idx):
        return self._prevCounts[idx]

    def __str__(self):
        return str(self._prevCounts)

    def update(self,roidbs):
        for idx,roidb in enumerate(roidbs):
            if roidb is None: continue
            self._prevCounts[idx] = len(roidb)

    def zero(self):
        self.setAllTo(0)

    def setAllTo(self,val):
        for idx in range(8):
            self._prevCounts[idx] = val


def printSaveBboxInfo(roidb,numAnnos,splitStr):
    print("computing bbox info [{:s}]...".format(splitStr))
    areas, widths, heights = get_bbox_info(roidb,numAnnos)

    print("[{:s}] ave area: {} | std. area: {}".format(splitStr,np.mean(areas),np.std(areas,dtype=np.float64)))
    print("[{:s}] ave width: {} | std. width: {}".format(splitStr,np.mean(widths),np.std(widths,dtype=np.float64)))
    print("[{:s}] ave height: {} | std. height: {}".format(splitStr,np.mean(heights),np.std(heights,dtype=np.float64)))
    prefix_path = cfg.IMDB_REPORT_OUTPUT_PATH
    if osp.exists(prefix_path) is False:
        os.makedirs(prefix_path)

    path = osp.join(prefix_path,"areas_{}.dat".format(splitStr))
    np.savetxt(path,areas,fmt='%.18e',delimiter=' ')
    path = osp.join(prefix_path,"widths_{}.dat".format(splitStr))
    np.savetxt(path,widths,fmt='%.18e',delimiter=' ')
    path = osp.join(prefix_path,"heights_{}.dat".format(splitStr))
    np.savetxt(path,heights,fmt='%.18e',delimiter=' ')
    

def print_report(roidbTr,annoCountTr,roidbTe,annoCountTe,setID,repeat,size):
    # legacy
    printRoidbReport(roidbTr,annoCountTr,roidbTe,annoCountTe,setID,repeat,size)

def printTrainTestRoidbReport(roidbTr,annoCountTr,roidbTe,annoCountTe,setID,repeat,size):

    numAnnosTr = computeTotalAnnosFromAnnoCount(annoCountTr)
    numAnnosTe = computeTotalAnnosFromAnnoCount(annoCountTe)

    print("\n\n-=-=-=-=-=-=-=-=-\n\n")
    print("Report:\n\n")
    print("Mixture Dataset: {} {} {}\n\n".format(setID,repeat,size))
    print_set_report(roidbTr,numAnnosTr,"train")
    print_set_report(roidbTe,numAnnosTe,"test")
    print("example [train] roidb:")
    for k,v in roidbTr[10].items():
        print("\t==> {},{}".format(k,type(v)))
        print("\t\t{}".format(v))
    printSaveBboxInfo(roidbTr,numAnnosTr,"train")
    printSaveBboxInfo(roidbTe,numAnnosTe,"test")

def printTrainTestRoidbDictReport(roidbTr,annoCountTr,roidbTe,annoCountTe,setID,repeat,size):

    numAnnosTr = computeTotalAnnosFromAnnoCount(annoCountTr)
    numAnnosTe = computeTotalAnnosFromAnnoCount(annoCountTe)

    print("\n\n-=-=-=-=-=-=-=-=-\n\n")
    print("Report:\n\n")
    print("Mixture Dataset: {} {} {}\n\n".format(setID,repeat,size))
    print_set_report(roidbTr,numAnnosTr,"train")
    print_set_report(roidbTe,numAnnosTe,"test")
    print("example [train] roidb:")
    for k,v in roidbTr[10].items():
        print("\t==> {},{}".format(k,type(v)))
        print("\t\t{}".format(v))
    printSaveBboxInfo(roidbTr,numAnnosTr,"train")
    printSaveBboxInfo(roidbTe,numAnnosTe,"test")

    
def printRoidbReport(roidb,numAnnos,splitStr):
    print("number of images [{}]: {}".format(splitStr,len(roidb)))
    print("number of annotations [{}]: {}".format(splitStr,numAnnos))
    print("size of roidb in memory [{}]: {}kB".format(splitStr,len(roidb) * sys.getsizeof(roidb[0])/1024.))

def print_set_report(roidb,numAnnos,splitStr):
    # legacy
    printRoidbReport(roidb,numAnnos,splitStr)

def get_bbox_info(roidb,size):
    areas = np.zeros((size))
    widths = np.zeros((size))
    heights = np.zeros((size))
    actualSize = 0
    idx = 0
    print(size)
    for image in roidb:
        # skipped *flipped* samples
        if image['flipped'] is True: continue
        bbox = image['boxes']
        for box in bbox:
            actualSize += 1
            widths[idx] = box[2] - box[0]
            heights[idx] = box[3] - box[1]
            assert widths[idx] >= 0,"widths[{}] = {}".format(idx,widths[idx])
            assert heights[idx] >= 0
            areas[idx] = widths[idx] * heights[idx]
            idx += 1
    print("actual: {} | theoretical: {}".format(idx,size))
    return areas,widths,heights

def saveNtdConfMats(cmRaw,cmCropped,ntdGameInfo,infix=None):
    fn = "confMats_{}_{}_{}_{}.pkl".format(ntdGameInfo['setID'],ntdGameInfo['repeat'],
                                           ntdGameInfo['size'],cfg.uuid)
    fid = open(iconicImagesFileFormat().format(fn),"wb")
    pickle.dump({"raw":cmRaw,"cropped":cmCropped},fid)
    fid.close()

def plotNtdConfMats(cmRaw,cmCropped,cmDiff,ntdGameInfo,infix=None):

    if infix in ntdGameInfo.keys():
        appendStr = '{}_{}_{}_{}_{}'.format(ntdGameInfo['setID'],ntdGameInfo['size'],
                                                 cfg.uuid,infix,ntdGameInfo[infix])
    else:
        appendStr = '{}_{}_{}_{}'.format(ntdGameInfo['setID'],ntdGameInfo['repeat'],
                                         ntdGameInfo['size'],cfg.uuid)
                                  
    pathToRaw = osp.join(cfg.PATH_TO_NTD_OUTPUT, 'ntd_raw_{}'.format(appendStr))
    pathToCropped = osp.join(cfg.PATH_TO_NTD_OUTPUT, 'ntd_cropped_{}'.format(appendStr))
    pathToDiff = osp.join(cfg.PATH_TO_NTD_OUTPUT, 'ntd_diff_raw_cropped_{}'.format(appendStr))
    plot_confusion_matrix(np.copy(cmRaw), cfg.clsToSet,
                          pathToRaw, title="Raw Images",
                          cmap = plt.cm.bwr_r,vmin=-100,vmax=100)
    plot_confusion_matrix(np.copy(cmCropped), cfg.clsToSet,
                          pathToCropped, title="Cropped Images",
                          cmap = plt.cm.bwr_r,vmin=-100,vmax=100)
    plot_confusion_matrix(np.copy(cmDiff), cfg.clsToSet, 
                          pathToDiff,title="Raw - Cropped",
                          cmap = plt.cm.bwr_r,vmin=-100,vmax=100)


def printRoidbDictImageNamesToTextFile(roidbDict,postfix):
    fid = open("output_{}.txt".format(postfix),"w+")
    for key in roidbDict.keys():
        for idx,sample in enumerate(roidbDict[key]):
            fid.write(sample['image']+"\n")
    fid.close()
    
def printRoidbImageNamesToTextFile(roidb,postfix):
    fid = open("output_{}.txt".format(postfix),"w+")
    for sample in roidb:
        print(sample['image'])
        fid.write(sample['image']+"\n")
    fid.close()
        
def computeRoidbDictLens(roidbTrDict,roidbTeDict):
    lenTr = 0
    for roidb in roidbTrDict.values():
        lenTr += len(roidb)
    lenTe = 0
    for roidb in roidbTeDict.values():
        lenTe += len(roidb)

    return lenTr,lenTe

def flattenRoidbDict(roidbDict,numSamples=None):
    roidbFlattened = []
    for key,roidb in roidbDict.items():
        print("{}: {} images".format(key,len(roidb)))
        toExtend = roidb
        if numSamples is not None:
            index = cfg.DATASET_NAMES_ORDERED.index(key)
            sizeToKeep = numSamples[index]
            if sizeToKeep is not None:
                print("[flattenRoidbdict] shortened roidb from {} to {}".format(len(roidb),
                                                                                sizeToKeep))
            else:
                sizeToKeep = len(roidb)
            toExtend = roidb[:sizeToKeep]
        roidbFlattened.extend(toExtend)
    return roidbFlattened

def vis_dets(im, class_names, dets, _idx_, fn=None, thresh=0.5):
    """Draw detected bounding boxes."""
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    if dets is not None:
       for i in range(len(dets)):
           bbox = dets[i, :4]


           ax.add_patch(
               plt.Rectangle((bbox[0], bbox[1]),
                             bbox[2] - bbox[0],
                             bbox[3] - bbox[1], fill=False,
                             edgecolor='red', linewidth=3.5)
           )
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    if fn is None:
        plt.savefig("img_{}_{}.png".format(_idx_,str(uuid.uuid4())))
    else:
        plt.savefig(fn.format(_idx_,str(uuid.uuid4())))


def toRadians(angle):
    return (np.pi/180 * angle)

def getRotationScale(M,rows,cols):
    a = np.array([cols,0,1])
    b = np.array([0,0,1])
    ta = np.matmul(M,a)
    tb = np.matmul(M,b)
    if cfg._DEBUG.utils.misc:
        print("[getRotationScale] ta",ta)
        print("[getRotationScale] tb",tb)
    if ta[1] < tb[0]:
        rotate_y = 2 * (-ta[1]) + rows
        scale = (rows) / ( rotate_y )
    else:
        rotate_y = 2 * (-tb[0]) + cols
        scale = (cols) / ( rotate_y )
    return scale

def getRotationInfo(angle,cols,rows):
    if cfg._DEBUG.utils.misc: print("cols,rows",cols,rows)
    rotationMat = cv2.getRotationMatrix2D((cols/2,rows/2),\
                                          angle,1.0)
    scale = getRotationScale(rotationMat,rows,cols)
    if cfg._DEBUG.utils.misc: print("scale: {}".format(scale))
    rotationMat = cv2.getRotationMatrix2D((cols/2,rows/2),\
                                          angle,scale)
    return rotationMat,scale

def addImgBorder(img,border=255):
    img[0,:,:] = border
    img[-1,:,:] = border
    img[:,0,:] = border
    img[:,-1,:] = border

def getImageWithBorder(_img,border=255,rotation=None):
    img = _img.copy()
    if cfg._DEBUG.utils.misc: print("[save_image_with_border] rotation",rotation)
    if rotation:
        angle,cols,rows = rotation[0],rotation[1],rotation[2]
        rotationMat,scale = getRotationInfo(angle,cols,rows)
        if cfg._DEBUG.utils.misc: print("[utils/misc.py] rotationMat",rotationMat)
        img_blank = np.zeros(img.shape,dtype=np.uint8)
        addImgBorder(img_blank,border=border)
        if cfg._DEBUG.utils.misc: print(img_blank.shape)
        img_blank = cv2.warpAffine(img_blank,rotationMat,(cols,rows),scale)
        img += img_blank
    addImgBorder(img,border=border)
    return img

def save_image_with_border(fn,_img,border=255,rotation=None):
    img = getImageWithBorder(_img,border=border,rotation=rotation)
    fp = osp.join(cfg.ROTATE_PATH,fn)
    cv2.imwrite(fp,img)

def save_image_of_overlap_bboxes(fn,img_bb1,img_bb2,rot1,rot2):
    img = np.zeros(img_bb1.shape)

    inter = np.bitwise_and(img_bb1.astype(np.bool),img_bb2.astype(np.bool))
    union = np.bitwise_or(img_bb1.astype(np.bool),img_bb2.astype(np.bool))
    overlap = np.sum(inter) / float(np.sum(union))

    img1 = getImageWithBorder(img_bb1,border=255,rotation=rot1)
    img2 = getImageWithBorder(img_bb2,border=255,rotation=rot2)
    img[:,:,0] = npBoolToUint8(inter)[:,:,0] # blue
    img[:,:,1] = img1[:,:,1] # green; the guess
    img[:,:,2] = img2[:,:,2] # red; the groundtruth
    cv2.putText(img,'{:0.2f}'.format(overlap),(0,60),cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255,255))
    fp = osp.join(cfg.ROTATE_PATH,fn)
    cv2.imwrite(fp,img)

def save_list_of_bbox_imgs(fn,pimgs):
    img = pimgs[0].astype(np.uint8)
    for pimg in pimgs:
        img = np.bitwise_or(pimg.astype(np.uint8),img)
    fp = osp.join(cfg.ROTATE_PATH,fn)
    cv2.imwrite(fn,img)    

    ADD_NOISE = True
    moves = []
    if ADD_NOISE:
        moves = zip([0,1,-5,3],[0,-1,3,-5])
    base_fn = fn.split(".")[0]
    cols = img.shape[0]
    rows = img.shape[1]
    for idx,m_xy in enumerate(moves):
        xm,ym = m_xy[0],m_xy[1]
        fn = "{}_{}.png".format(base_fn,idx)
        M = np.float32([[1,0,xm],[0,1,ym]])
        dst = cv2.warpAffine(img,M,(cols,rows))
        cv2.imwrite(fn,dst)
        
def npBoolToUint8(a):
    return a.astype(np.uint8)*255

def centerAndScaleBBox(bbox,rotMat,scale):
    if cfg._DEBUG.utils.misc: print("[centerAndScaleBBox] before bbox",bbox)

    cx = np.mean([bbox[0],bbox[2]])
    cy = np.mean([bbox[1],bbox[3]])
    center = np.array([cx,cy,1])
    new_center = np.matmul(rotMat,center)

    x_len = bbox[2] - bbox[0]
    y_len = bbox[3] - bbox[1]

    x1 = int(new_center[0] - 0.5 * scale * x_len)
    x2 = int(new_center[0] + 0.5 * scale * x_len)
    y1 = int(new_center[1] - 0.5 * scale * y_len)
    y2 = int(new_center[1] + 0.5 * scale * y_len)
    new_bbox = [x1,y1,x2,y2]
    if cfg._DEBUG.utils.misc: print("[centerAndScaleBBox] after bbox",new_bbox)

    return new_bbox


def print_net_activiation_data(net,layers_to_print):
    print("-"*50)
    for name,blob in net.blobs.items():
        if name in layers_to_print:
            print("{}: {}".format(name,blob.data.shape))
    print("-"*50)


def createNoisyBox(scale):
    r = npr.rand(4)*scale
    return r - np.mean(r)


def loadActivityVectors():
    av = {}
    avDir = cfg.GET_SAVE_ACTIVITY_VECTOR_BLOBS_DIR()
    for fn in glob.glob('{}/*.pkl'.format(avDir)):
        start = fn.rfind('/') + 1
        end = fn.rfind('.')
        blob_name = fn[start:end]
        with open(fn,'rb') as f:
            av[blob_name] = pickle.load(f)
    return av


def evaluate_image_detections(roidb,cls_dets):
    # we only decide if a bounding box is detected;
    # e.g. (TP and FN) only
    #      NOT (FP or FN)
    # print("[./utils/misc.py: evaluate_image_detections]")
    BBGT = []
    for elem in roidb:
        for idx,cls in enumerate(elem['gt_classes']):
            if cls != 0: continue #cfg.PERSON_INDEX
            BBGT.append(elem['boxes'][idx])
    BBGT = np.array(BBGT)
    # print(BBGT.shape)
    found = np.zeros((BBGT.shape[0])).astype(np.uint8)
    for bb in cls_dets:
        ixmin = np.maximum(BBGT[:, 0], bb[0])
        iymin = np.maximum(BBGT[:, 1], bb[1])
        ixmax = np.minimum(BBGT[:, 2], bb[2])
        iymax = np.minimum(BBGT[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih

        # union
        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
               (BBGT[:, 2] - BBGT[:, 0] + 1.) *
               (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)
        found[jmax] = 1
    return found

def openAlResultsCsv():
    alResults = {}
    fn = "resultsAL_{}_{}_{}_{}.csv".format(
        cfg.ACTIVE_LEARNING.N_ITERS,
        cfg.ACTIVE_LEARNING.VAL_SIZE,
        cfg.ACTIVE_LEARNING.SUBSET_SIZE,
        cfg.ACTIVE_LEARNING.N_COVERS)
    with open(fn,"r") as f:
        csv_reader = csv.reader(f,delimiter=',')
        for line_num,row in enumerate(csv_reader):
            if line_num == 0: continue
            image_index = row[0]
            if row[1] == "":
                pctErrorRed_ave = -1
                pctErrorRed_std = -1
            else:
                pctErrorRed_ave = float(row[1]) # average
                pctErrorRed_std = float(row[2]) # std (error or deviation?)
            alResults[image_index] = {}
            alResults[image_index]['ave'] = pctErrorRed_ave
            alResults[image_index]['std'] = pctErrorRed_std
    return alResults
    
def startAlReport(imdb,net):
    fn = "alReport_{}_{}.csv".format(imdb.name,net.name)
    fidAlReport = open(fn,"w+")
    createAlReportHeader(fidAlReport,imdb)
    return fidAlReport

def createAlReportHeader(fidAlReport,imdb):
    headerStr = "image_index,"
    cls_probs_prefix = "cls_prob"
    for idx in range(imdb.num_classes):
        headerStr += "{}_{},".format(cls_probs_prefix,str(idx))
    headerStr += "cls_prob_entropy"
    av_prefix = "av"
    for av_blobName in cfg.SAVE_ACTIVITY_VECTOR_BLOBS:
        headerStr += "{}_{},".format(av_prefix,av_blobName)
    headerStr += "%errorReduction\n"
    fidAlReport.write(headerStr)

def transformField(field,value):
    if field == 'conv1_1':
        return computeEntopy(value)
    elif field == 'cls_prob':
        return computeEntopy(value)
    return value
        
def computeEntopy(value):
    cvalue = value.copy()
    cvalue += np.abs(np.min(cvalue))
    cvalue /= cvalue.sum()
    return -(cvalue * np.ma.log(cvalue)/np.log(np.e)).sum()





