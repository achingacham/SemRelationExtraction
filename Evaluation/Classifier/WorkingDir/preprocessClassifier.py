import numpy as np
import re
import os
import sys
import torch

class relationEmbedding:
    
    def __init__(self, ifolder, evalFile):
        
        self.evalSet = dict()
        self.embeddingSet = dict()
        
        #self.readTransitionMatrix(ifolder)
        self.readEmbeddingFile(ifolder)
        self.readEvalFile(evalFile)
        
    def readTransitionMatrix(self, ifolder):
        
        TransitionMatrixW = []
        
        with open(ifolder+"TransitionWeight") as ifile:
            
            for line in ifile:
                items = line.split()
                temp = [float(i) for i in items[:]]
                TransitionMatrixW.append(temp)  
            
            self.npTransitionMatrixW = np.array(TransitionMatrixW)
            self.npTransitionMatrixW = torch.FloatTensor(self.npTransitionMatrixW)
            
        TransitionMatrixB = []
        
        with open(ifolder+"TransitionBias") as ifile:
            
            for line in ifile:
                TransitionMatrixB.append(float(line))  
       
            self.npTransitionMatrixB = np.array(TransitionMatrixB)
            self.npTransitionMatrixB = torch.FloatTensor(self.npTransitionMatrixB)
            
            
    def readEmbeddingFile(self, ifolder):
        
        with open(ifolder+"Epoch_0_EMB_400_All.txt") as ifile:
            
            for line in ifile:
                items = line.split()
                self.embeddingSet[items[0]] = 1 #np.array([float(i) for i in items[1:]])
                
        
        
    def readEvalFile(self, inputFile):

        with open(inputFile) as ifile:

            for line in ifile:
                items = line.split()
                self.evalSet[(items[0],items[1])] = items[2]
                
                
                
                try:
                    w1Emb = self.embeddingSet[items[0]]
                    
                except:
                    
                    continue
                    
                try:
                   
                    w2Emb = self.embeddingSet[items[1]]
                    
                except:
                    print(items[1])
                    continue
                
                pairEmb = torch.FloatTensor(np.append(w1Emb,w2Emb)).view(1,-1)
                
                
            #print(self.npTransitionMatrixW)
            
            #relEmb = torch.mul(self.npTransitionMatrixW, pairEmb)  #+ self.npTransitionMatrixB
            
            
            
            

if __name__ == '__main__':
    
    ifolder = sys.argv[1]
    ievalfile = sys.argv[2]
    
    rEmb = relationEmbedding(ifolder, ievalfile)

