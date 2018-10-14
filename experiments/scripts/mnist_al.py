#!/usr/bin/env python3
import os,sys,re,yaml,subprocess,pickle
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

trainCommandTemplate = "./tools/train_net.py --iters {} --imdb mnist-{}-default --cfg ./experiments/cfgs/cls_mnist_al.yml --solver ./models/mnist/{}/solver.prototxt --solver_state ./mnist_al_net_{}_iter_{}.solverstate --new_path_to_imageset {} --snapshot_infix {}"
testCommandTemplate = "./tools/test_net.py --imdb mnist-test-default --cfg ./experiments/cfgs/cls_mnist.yml --def ./models/mnist/{}/test.prototxt --net ./output/classification/mnist/mnist_al_net_{}_{}_iter_{}.caffemodel"
trainOriginalCommandTemplate = "./tools/train_net.py --iters {} --imdb mnist-train-default --cfg ./experiments/cfgs/cls_mnist_al.yml --solver ./models/mnist/{}/solver.prototxt"
testOriginalModelCommandTemplate = "./tools/test_net.py --imdb mnist-test-default --cfg ./experiments/cfgs/cls_mnist.yml --def ./models/mnist/{}/test.prototxt --net {}"

def dbprint(item):
    print("DEBUG_PRINT")
    print(item)
    
def getAlInfo(fn):
    with open(fn, 'r') as f:
        yaml_cfg = edict(yaml.load(f))
    return yaml_cfg

def assignAccuracy(alIds,alSet,acc):
    alSubset = getAlSubsetInfo(alSet)
    for alId in alSubset:
        if alIds[alId] is None: alIds[alId] = []
        alIds[alId].append(acc)
    
def getIdList(imageSetPath):
    with open(imageSetPath,"r") as f:
        lines = f.readlines()
    ids = [int(line.strip()) for line in lines]
    return ids
    
def runCommandProcess(setCommand):
    modelProc = subprocess.Popen(setCommand.split(' '),stdout=subprocess.PIPE)
    output_b,isSuccess = modelProc.communicate()
    assert isSuccess is None, print("ERROR; command failed.")
    output = output_b.decode('utf-8')
    return output

def testOriginalModel(netPath,modelType):
    setTestCommand = testOriginalModelCommandTemplate.format(modelType,netPath)
    return testModel(setTestCommand)

def trainOriginalModel(nIters,modelType):
    setTrainCommand = trainOriginalCommandTemplate.format(nIters,modelType)
    output = trainModel(setTrainCommand)
    return findCaffemodelSnapshot(output)

def trainModel(setTrainCommand):
    print(setTrainCommand)
    return runCommandProcess(setTrainCommand)

def findCaffemodelSnapshot(output):
    findName = r"Wrote snapshot to: (?P<name>.*)"
    trainedModelName = re.findall(findName,output)[-1]
    return trainedModelName

def testModel(setTestCommand):
    print(setTestCommand)
    output = runCommandProcess(setTestCommand)
    findAcc = r"Accuracy: (?P<acc>[0-9]+\.[0-9]+)"
    acc = float(re.findall(findAcc,output)[0])
    return acc

def trainAlModel(alSet,nIters,originalIters,alSetDir,modelType):
    setTrainCommand = trainCommandTemplate.format(nIters,alSet,modelType,\
                                                  modelType,originalIters,alSetDir,alSet)
    output = trainModel(setTrainCommand)
    print("trainAlModel")
    print(output)
    return None

def testAlModel(alSet,nIters,modelType):
    setTestCommand = testCommandTemplate.format(modelType,modelType,alSet,nIters)
    return testModel(setTestCommand)

def getAlSubsetInfo(alSet):
    alSetFn = alSet + ".txt"
    alSetPath = osp.join(alSetDir,alSetFn)
    alSubset = getIdList(alSetPath)
    return alSubset

def saveStatePickle(saveALResultsFn,alIds,originalAcc,subsetIndex,cover):
    stateInfo = {"subsetIndex":subsetIndex,"cover":cover}
    saveALResultsFnPkl = saveALResultsFn+".pkl"
    print("saving AL state information to {}".format(saveALResultsFnPkl))
    with open(saveALResultsFnPkl, 'wb') as f:
        pickle.dump({'alIds':alIds,'originalAcc':originalAcc,'stateInfo':stateInfo}, f)

def restoreStatePickle(saveALResultsFn):
    saveALResultsFnPkl = saveALResultsFn+".pkl"
    with open(saveALResultsFnPkl, 'rb') as f:
        stateInfo = pickle.load(f)
    return stateInfo['alIds'],stateInfo['originalAcc'],stateInfo['stateInfo']

def savePickleFile(saveALResultsFn,alIds,originalAcc):
    saveALResultsFnPkl = saveALResultsFn+".pkl"
    print("saving raw AL results to {}".format(saveALResultsFnPkl))
    with open(saveALResultsFnPkl, 'wb') as f:
        pickle.dump({'alIds':alIds,'originalAcc':originalAcc}, f)
    
def saveCsvFile(saveALResultsFn,alIds,originalAcc):
    saveALResultsFnCsv = saveALResultsFn+".csv"
    print("saving summary stat AL results to {}".format(saveALResultsFnCsv))
    f = open(saveALResultsFnCsv,"w+")
    f.write("image_index,mean_acc,std_acc\n")
    for image_index,acc_list in alIds.items():
        mean = ""
        std = ""
        if acc_list is None:
            pass
            #print("warning: we actually have a none; very unlikely to occur")
        else:
            percent_diff = computePercentDifference(acc_list,originalAcc)
            print("% difference")
            print(percent_diff)
            print(acc_list)
            print(originalAcc)
            mean = np.mean(percent_diff)
            std = np.std(percent_diff)
        f.write("{},{},{}\n".format(image_index,mean,std))
    f.close()

def computePercentDifference(accuracy_list,originalAccuracy):
    acc = np.array(accuracy_list)
    oa = originalAccuracy
    return (acc - oa) / ((acc + oa) / 2.)

def handleTrainedModel(caffemodelPath):
    """
    the caffemodel needs to be copied to the solverstate dir &
    the solverstate needs to be copied to the local dir
    """
    cmDir = '/'.join(caffemodelPath.split("/")[:-1])
    trainModelNetNameBase = caffemodelPath.split("/")[-1].split(".")[0]
    ss = trainModelNetNameBase+".solverstate"
    cm = trainModelNetNameBase+".caffemodel"
    # copySolverstateDest = osp.join(cmDir,ss)
    copyCaffemodelDest = osp.join(cmDir,ss)
    copyCommand = "cp {} {}".format(caffemodelPath,copyCaffemodelDest)
    print(runCommandProcess(copyCommand))
    return ss

if __name__ == "__main__":

    # arguments
    oIters = 6000
    alInfoFn = "/home/gauenk/Documents/data/mnist/ImageSets/AL_single/alSet_info.txt"
    modelType = "lenet5"
    nPassesofAlSubsets = 1
    saveMod = 300
    restoreState = False

    #-----------------------
    # CREATE ORIGINAL MODEL
    #-----------------------

    # train original model
    # caffemodel = trainOriginalModel(oIters,modelType)
    # solverstate = handleTrainedModel(caffemodel)
    caffemodel = "./output/classification/mnist/mnist_al_net_lenet5_iter_6000.caffemodel"
    solverstate = "./mnist_al_net_lenet5_iter_6000.solverstate"

    # test original model
    originalAcc = testOriginalModel(caffemodel,modelType)
    print(originalAcc)

    #-------------------
    # ACTIVE LEARNING
    #-------------------

    # set al arguments from generating the al subsets
    alInfo = getAlInfo(alInfoFn)
    alSetOrigin = alInfo['alSetOrigin']
    alSetDir = alInfo['alSetDir']
    nIters = alInfo['subsetSize'] * nPassesofAlSubsets + oIters

    # get of ALL image indicies available for active learning
    alIds = dict.fromkeys(getIdList(alSetOrigin),None)

    # save active learning information
    saveALResultsFn = "resultsAL_{}_{}_{}_{}".format(nIters,
                                                     alInfo['valSize'],
                                                     alInfo['subsetSize'],
                                                     alInfo['numberOfCovers'])
    # load state information
    start_cover,start_subset = 0,0
    if restoreState:
        alIds,originalAcc,stateInfo = restoreStatePickle(saveALResultsFn)
        start_cover,start_subset = stateInfo['cover'],stateInfo['subsetIndex']
    
    # run the AL experiments
    for cover in range(start_cover,alInfo['numberOfCovers']):
        for subsetIndex in range(start_subset,alInfo['subsetsInCover']):
            alSet = "alSet_{}_{}_{}_{}".format(subsetIndex,cover,
                                               alInfo['subsetSize'],
                                               alInfo['valSize']) 
            trainAlModel(alSet,nIters,oIters,alSetDir,modelType) # train the new model
            acc = testAlModel(alSet,nIters,modelType) # test the new model
            assignAccuracy(alIds,alSet,acc) # aggregate the accuracy
            # save the current state information
            if subsetIndex % saveMod == 0:
                saveStatePickle(saveALResultsFn,alIds,originalAcc,subsetIndex,cover)
                
    # save information
    savePickleFile(saveALResultsFn,alIds,originalAcc)
    saveCsvFile(saveALResultsFn,alIds,originalAcc)


