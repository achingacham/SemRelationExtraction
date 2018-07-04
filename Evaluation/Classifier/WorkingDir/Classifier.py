import re
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from modelClassifier import RelationClassifier 
#from dataClassifier import modelData
from dataClassifier import modelData
from trainClassifier import modelTrain
import ipdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as matpy


torch.cuda.manual_seed(1)


def plot_results(results):
    
    rows = 3
    cols = 2
    
    fig = matpy.figure(figsize=(15,15))
    matpy.suptitle(validationfile)
    
    for index_r,result in enumerate(results):
        
        for index_i,item in enumerate(result):
            
            axplot = fig.add_subplot(rows, cols, (cols*index_r)+index_i+1)
            axplot.plot(item[0], label="Train")
            axplot.plot(item[1] , label="Dev")
            axplot.plot(item[2], label="Test")
            axplot.set_xlabel(item[3])
            axplot.set_ylabel(item[4])
            axplot.set_title(item[5])
            axplot.legend()
            
            
    fig.savefig(outfolder+"Plots")
            

if __name__ == '__main__':
    
    """
    Arguements required:
    
    Input folder : Folder containg relation embeddings
    Output folder : Folder to store results
    Validation file : BLESS / EVAL/ EACL
    Tag : for tracking the files considered(enter 'All')
    epochFile : Relation embedding file at a certain epoch
    batchSize: batch size
    epochTrain : number of epochs to be trained for classifier
    preTrainedWordEmbedding : Pre trained word vectord for offset method
    
    
    """
    
    ifolder = sys.argv[1]
    ofolder = sys.argv[2]
    print("\n Enter validationFile Tag EpochFile BatchSize #Epochs \n") 
    validationfile = sys.argv[3]
    tag = sys.argv[4]
    epochFile = sys.argv[5]
    batchSize = int(sys.argv[6])
    epochTrain = int(sys.argv[7])
    preTrainedWordEmbedding = sys.argv[8]
    
    results = [[None]] * 3
    
    outfolder = ofolder+ifolder.rsplit('/',2)[1]+'/'+tag+'/'+validationfile+'/'
    
    try:
        os.makedirs(outfolder)
        
    except:
        print("Output folder will be overwritten!")
    
    if re.search("All", tag):
   
        validatedBlessSet = ifolder+validationfile+'.txt'
        validatedRelationEmbedding = ifolder+epochFile
        
        #DATA
        inputData = modelData(validatedBlessSet, validatedRelationEmbedding)
        inputData.create_dictWordVectors(preTrainedWordEmbedding)

        
        
    else:
        print("\n Tag not acceptable")
    
   
    
    
    #MODEL
    INPUT_DIM = inputData.input_dim
    HIDDEN_UNIT = inputData.input_dim
    OUTPUT_LABEL = inputData.create_labelsToIndex()

    
    
    lr = [0.05, 0.05, 0.05]
    l2_factor = [0.1, 0.07, 0.1]
    
    with open(outfolder+"Modelparams_Log","a") as paramFile:
        
        paramFile.write("\n L2 factors and Learning rate : "+str(l2_factor)+" & "+str(lr)+", for files "+ str(validationfile)+ " and "+str( epochFile) +"With batchSize "+ str(batchSize))
        
   
    model = RelationClassifier(INPUT_DIM, HIDDEN_UNIT, OUTPUT_LABEL)
    loss = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=lr[0])
    
    if torch.cuda.is_available():
        model.cuda()
    
    if re.search("SVD", epochFile):
        mTrain = modelTrain(inputData, model, loss, optimizer,outfolder,'logfile.log', "SVDRel")
    else:
        mTrain = modelTrain(inputData, model, loss, optimizer,outfolder,'logfile.log', "SkipRel")
        
    results[0] = mTrain.train(batchSize, epochTrain, l2_factor[0], lr[0])
    
    
    
    model = RelationClassifier(400, 400, OUTPUT_LABEL)
    loss = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=lr[1])
    
    if torch.cuda.is_available():
        model.cuda()
    
    mTrain = modelTrain(inputData, model, loss, optimizer, outfolder, "logfile.log", "JustWord")
    results[1] = mTrain.train(batchSize, epochTrain, l2_factor[1], lr[1])
    
    model = RelationClassifier(INPUT_DIM + 400, HIDDEN_UNIT + 400, OUTPUT_LABEL)
    loss = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=lr[2])
    
    
    if torch.cuda.is_available():
        model.cuda()
    
    mTrain = modelTrain(inputData, model, loss, optimizer, outfolder, "logfile.log", "RelWord")
    results[2] = mTrain.train(batchSize, epochTrain, l2_factor[2], lr[2])
    
    
    plot_results(results)
   
    