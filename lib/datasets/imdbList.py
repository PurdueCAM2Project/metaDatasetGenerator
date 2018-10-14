import sys,os,re
import os.path as osp
from factory import get_repo_imdb


class imdbList():

    def __init__(self,datasetsToLoad):
        self.datasets = {}
        for sets in setsToLoad:
            self.datasets[sets] = get_repo_imdb(sets)
        self.trainSets = {}
        self.testSets = {}
        self.roidb = None
            
    def splitDatasets(self,trainSets,testSets):
        # TODO: CHECK FOR OVERLAP

        for trainSet in trainSets:
            self.trainSets[trainSet] = self.datasets[trainSet]
        for testSet in testSets:
            self.testSets[testSets] = self.datasets[testSets]




