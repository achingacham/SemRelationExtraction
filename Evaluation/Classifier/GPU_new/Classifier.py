import re
import os
import sys
import torch
import torch.nn as nn
from modelClassifier import RelationClassifier 
from dataClassifier import modelData
from trainClassifier import modelTrain
import ipdb

torch.cuda.manual_seed(1)

    
if __name__ == '__main__':
    
    ifolder = sys.argv[1]
    ofolder = sys.argv[2]
    print("\n Enter validation files like: BlessSet, EvalSet and Tags like: JustRelVectors, JustWordVectors, RelWordVector \n") 
    validationfile = sys.argv[3]
    tag = sys.argv[4]
    epochEMB = sys.argv[5]
    batchSize = int(sys.argv[6])
    epochTrain = int(sys.argv[7])
    ###and preTrainedWordEmbedding from sys.argv[8]
    
    outfolder = ofolder+ifolder.rsplit('/',2)[1]+'/'+tag+'/'+validationfile+'/'
    
    try:
        os.makedirs(outfolder)
        
    except:
        print("Output folder will be overwritten!")
    
    
    if re.search("JustRel",tag):
        validatedBlessSet = ifolder+validationfile+'.txt'
        validatedRelationEmbedding = ifolder+"Epoch_"+epochEMB+"_EMB_All.txt"
        
        #DATA
        inputData = modelData(validatedBlessSet,tag)
        inputDim = inputData.create_dictRelVectors(validatedRelationEmbedding)

    
    elif re.search("JustWord",tag):
        validatedBlessSet = ifolder+validationfile+'.txt'
        preTrainedWordEmbedding = sys.argv[8]
        
        #DATA
        inputData = modelData(validatedBlessSet, tag)
        inputDim = 400
        inputData.create_dictWordVectors(preTrainedWordEmbedding, inputDim)

    elif re.search("RelWord",tag):
        validatedBlessSet = ifolder+validationfile+'.txt'
        validatedRelationEmbedding = ifolder+"Epoch_"+epochEMB+"_EMB_All.txt"
        preTrainedWordEmbedding = sys.argv[8]
        
        #DATA
        inputData = modelData(validatedBlessSet, tag)
        inputDim = inputData.create_dictRelVectors(validatedRelationEmbedding)
        inputData.create_dictWordVectors(preTrainedWordEmbedding, 400)
        
    else:
        print("\n Tag not acceptable")
    
   
    
    #MODEL
    INPUT_DIM = inputDim
    HIDDEN_UNIT = inputDim 
    OUTPUT_LABEL = inputData.create_labelsToIndex()


    model = RelationClassifier(INPUT_DIM, HIDDEN_UNIT, OUTPUT_LABEL)
    loss = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=0.0050)
    
    if torch.cuda.is_available():
        model.cuda()
    
    
    mTrain = modelTrain(inputData, model, loss, optimizer,outfolder,'logfile.log')
    mTrain.train(batchSize, epochTrain)
    
