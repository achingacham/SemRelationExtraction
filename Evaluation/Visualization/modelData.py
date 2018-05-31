import random
import torch
import re
import numpy as np
import ipdb


class modelData:
    
    def __init__(self, blessFile, tag, finalDf):
        
        self.pairs = []
        with open(blessFile) as inputFile:
            
            for lines in inputFile:
                word1,word2,relation = lines.split()
                key = word1+":::"+word2
                self.pairs.append(key)
                finalDf.loc[finalDf['0']==key, 'class'] = relation

        inputFile.close()
        
        print("Initialized model data")
        
    def create_labelsToIndex(self):
        
        self.labelsToIndex = dict()
        self.indexToLabels = dict()
        
        for dataset in [self.trainData, self.testData, self.devData]:
            for data in dataset:
                split_data = data.split('\t')
                tempRelation = split_data[2].strip("[']")

                if tempRelation not in self.labelsToIndex:
                    indexT = len(self.labelsToIndex)
                    self.labelsToIndex[tempRelation] = indexT
                    self.indexToLabels[indexT] = tempRelation

        return len(self.labelsToIndex)

    
    def create_diffVectors(self, finalDf):
        
        self.dictDiffVectors = dict()
        
        for pairs in finalDf['0']:
            word1,word2 = pairs.split(':::')
            vector1 = self.dictWordVectors[word1]
            vector2 = self.dictWordVectors[word2]
            
            diffVector = vector1 - vector2
            self.dictDiffVectors[pairs] = diffVector
            
        print("Completed creation of diff vectors")
    
    def create_dictWordVectors(self, preTrainedVectors, dim):
        
        self.dictWordVectors = dict()
        with open(preTrainedVectors) as inputFile:
        
            for Vectors in inputFile:
                vec = Vectors.split()
                try:
                    if len(vec) == dim+1:
                        self.dictWordVectors[vec[0]] = np.array([float(value) for value in vec[1:]])
                except:
                    print(vec[0],len(vec))
                    
        inputFile.close()
        
        
        
   
