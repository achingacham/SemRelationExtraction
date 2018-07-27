import os
import re
import sys
import math
import ipdb
import copy
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.autograd as autograd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as matpy

from readData import modelData
from modelMapper import relationMapper

def plot_results(epoch_train ,xlabel, ylabel, plotFile):
    
    for index, item in epoch_train.data:
        
        matpy.plot(epoch_train[item[0]], label="Train")
        #matpy.plot(epoch_test, label="Test")
        matpy.ylabel(ylabel)
        matpy.xlabel(xlabel)
        matpy.legend()
        matpy.savefig(plotFile)
        matpy.close()

##Testing
def createValidationVectors(epoch, validationFile):
    
    print("Testing model")
    
    #Read validationList
    #dictValidationVectors = dict()
    
    try:
        os.remove(validationFile)
    
    except:
        pass
    
    outputFile = open(validationFile,"a")
    
    count = 0
    
    for items in inputData.validationList[validationFile]:
        word1 = items[0].lower()
        word2 = items[1].lower()
        
        relationType = inputData.validationList[validationFile][items]
    
        if word1 in inputData.dictWordVectors and word2 in inputData.dictWordVectors:
            
            wordVec1 = inputData.dictWordVectors[word1]
            wordVec2 = inputData.dictWordVectors[word2]
        
        else:
            continue
        
        #Create model input (by concatenation) and generate relation representation for each pair
        pairVec = copy.deepcopy(wordVec1)
        pairVec.extend(wordVec2)
        
        inputVector = autograd.Variable(torch.cuda.FloatTensor(pairVec)) #two dimensional vector
        
        predictVector = model.forward(inputVector)
        
        relationVector = predictVector.data
        
        count += 1
        
        outputFile.write("\n"+word1+":::"+word2)
        
        for vec in relationVector:
            
            outputFile.write("\t"+str(vec))
    
    
    outputFile.close()
    #Then run supervised classifier for whole dataset
        
    print("Validation list count: ", count)      

if __name__ == '__main__':
    
    """
    Arguements required:
    
    Input folder : Folder containg relation embeddings
    Output folder : Folder to store results
    epochFile : Relation embedding file at a certain epoch
    preTrainedWordEmbedding : Pre trained word vectord for offset method
    Validation file : BLESS / EVAL/ EACL
    
    """
    
    ifolder = sys.argv[1]
    ofolder = sys.argv[2]
    epochFile = sys.argv[3]
    preTrainedWordEmbedding = sys.argv[4]
    validationFilesList = sys.argv[5]
    
    epochCount = 3
    batchSize = 32
    lr = 0.05
    l2_factor = 0.001
    
    outfolder = ofolder+ifolder.rsplit('/',2)[1]+'/'
    
    try:
        os.makedirs(outfolder)
        
    except:
        print("Output folder will be overwritten!")
    
    
    relationEmbedding = ifolder+epochFile
    inputData = modelData(preTrainedWordEmbedding, relationEmbedding)
    
    with open(validationFilesList) as listFile:
        
        for file in listFile:
            
            inputData.createValidationSet(file.strip('\n'))
    
    pairCount = len(inputData.dictRelVectors)
    allPairs = [pair for pair in inputData.dictRelVectors.keys()]
    
    wordVecDim = len(inputData.dictWordVectors['the'])  #Word vector size of an obvious word in vocabulary
    relVecDim = len(inputData.dictRelVectors[allPairs[0]]) #Pair vector size of an observed pair
    
    #Initialize modelMapper and SGD optimizer
    model = relationMapper(wordVecDim, relVecDim)
    optimizer = torch.optim.SGD(model.parameters(),lr)
    
    if torch.cuda.is_available():
        model.cuda()
    
    batchCount = math.ceil(pairCount/batchSize)
    
    epochCost = []
    
    ##Training
    for currentEpoch in range(epochCount):
        
        print("\n Epoch: ", currentEpoch)
        
        
        random.shuffle(allPairs)
        
        batchCost = 0
        
        process_bar = tqdm(range(int(batchCount)))
        
        for currentBatch in process_bar:
            
            batchPairs = allPairs[currentBatch*batchSize:(currentBatch+1)*batchSize]
            batchInputVector = autograd.Variable(torch.cuda.FloatTensor(inputData.makeBatchInput(batchPairs)))
            batchTargetVector = autograd.Variable(torch.cuda.FloatTensor(inputData.makeBatchTarget(batchPairs)))
            
            model.zero_grad()
            batchPredictVector = model.forward(batchInputVector)
            
            #Mean square error between predicted and relation vector
            cost = F.mse_loss(batchPredictVector , batchTargetVector)
            
            
            ### Regularize before back propogation
                
            l2_reg = None
            for params in model.parameters():
                if l2_reg is None:
                    l2_reg = params.norm(2)
                else:
                    l2_reg = l2_reg + params.norm(2)

            #if epoch > 1:
            cost += l2_factor * l2_reg
            ###
            
            cost.backward()
            optimizer.step()
            batchCost += cost
            
            process_bar.set_description("Loss: %0.8f, lr: %0.6f" %
                                            (cost.data[0],optimizer.param_groups[0]['lr']))
            
        epochCost.append(batchCost/batchCount)
        
        with open(validationFilesList) as listFile:
        
            for validationFile in listFile:
        
                createValidationVectors(currentEpoch, validationFile.strip('\n'))
        
    print(epochCost)
    
    plot_results(epochCost, "Epochs", "Average Cost", outfolder+"Plots.jpg")            
    
    
    
  