import os
import sys
import numpy as np
import ipdb
import time


class modelData:
    
    def __init__(self, wordEmbeddingFile, relEmbeddingFile, validationFile):
        
        self.createWordDictionary(wordEmbeddingFile)
        
        self.createRelDictionary(relEmbeddingFile)
        
        self.createValidationSet(validationFile)
        
    
    def createValidationSet(self, validationFile):
        
        with open(validationFile) as inputFile:
               
            print("\nReading evalution dataset....\t")
            
            self.validationList = dict()

            for line in inputFile:
                line = line.strip('\n')
                tempList = line.split()
                self.validationList[(tempList[0],tempList[1])] = tempList[2]
                
        
    def createRelDictionary(self, iFile):
    
        self.dictRelVectors = dict()
        
        startTime = time.time()
        
        with open(iFile) as inputFile:
            
            inputFile.readline() #Read just the first line
            
            for vectors in inputFile:
                vec = vectors.split()
                
                words = vec[0].split(':::')

                # Check if corresponding word vectors exists for the pair
                if words[0] in self.dictWordVectors and words[1] in self.dictWordVectors:
                    
                    relVector = [float(value) for value in vec[1:]]
                    self.dictRelVectors[(words[0],words[1])] = relVector

                else:
                    continue

        
        stopTime = time.time()
        
        self.tapTime(startTime, stopTime)
        
        print("\n Pair vocabulary of size", len(self.dictRelVectors), "is created.")
   
    def createWordDictionary(self, iFile):
        
        self.dictWordVectors = dict()
        
        startTime = time.time()
        
        with open(iFile) as inputFile:
            
            for vectors in inputFile:
                vec = vectors.split()
                
                if len(vec) == 400+1:
                    self.dictWordVectors[vec[0]] = [float(value) for value in vec[1:]]
                    
                    
                
        stopTime = time.time()
        
        self.tapTime(startTime, stopTime)
        
        print("\n Word vocabulary of size", len(self.dictWordVectors), "is created.")
        
    def tapTime(self, start, stop):
        
        print("Time taken (seconds): ", stop-start)
        
        
    def makeBatchInput(self, batchPairs):
    
        batchInputVector = []
        
        for pair in batchPairs:
            
            word1 = pair[0]
            word2 = pair[1]
            
            vector1 = [float(value) for value in self.dictWordVectors[word1]]
            vector2 = [float(value) for value in self.dictWordVectors[word2]]
            
            #Concat two word vectors in order
            vector1.extend(vector2)    
            
            batchInputVector.append(vector1)
        
        return batchInputVector
    
    def makeBatchTarget(self, batchPairs):
    
        batchTargetVector = []
        
        for pair in batchPairs:
            
            relVector = [float(value) for value in self.dictRelVectors[pair]]
            batchTargetVector.append(relVector)
        
        return batchTargetVector
        
        