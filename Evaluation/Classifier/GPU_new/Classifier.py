import re
import os
import sys
import torch
import torch.nn as nn
from modelClassifier import RelationClassifier 
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
        inputData = modelData(validatedBlessSet)
        inputDim = inputData.create_dictRelVectors(validatedRelationEmbedding)
        inputData.create_dictWordVectors(preTrainedWordEmbedding, inputDim)

        
    else:
        print("\n Tag not acceptable")
    
   
    
    #MODEL
    INPUT_DIM = inputDim
    HIDDEN_UNIT = inputDim 
    OUTPUT_LABEL = inputData.create_labelsToIndex()

    
    model = RelationClassifier(INPUT_DIM, HIDDEN_UNIT, OUTPUT_LABEL)
    loss = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=0.050)
    
    if torch.cuda.is_available():
        model.cuda()
    
    
    mTrain = modelTrain(inputData, model, loss, optimizer,outfolder,'logfile.log', "JustRel")
    results[0] = mTrain.train(batchSize, epochTrain)

    
    model = RelationClassifier(INPUT_DIM, HIDDEN_UNIT, OUTPUT_LABEL)
    loss = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=0.050)
    
    if torch.cuda.is_available():
        model.cuda()
    
    mTrain = modelTrain(inputData, model, loss, optimizer, outfolder, "logfile.log", "JustWord")
    results[1] = mTrain.train(batchSize, epochTrain)
    
    
    model = RelationClassifier(INPUT_DIM*2, HIDDEN_UNIT*2, OUTPUT_LABEL)
    loss = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=0.050)
     
    if torch.cuda.is_available():
        model.cuda()
    
    mTrain = modelTrain(inputData, model, loss, optimizer, outfolder, "logfile.log", "RelWord")
    results[2] = mTrain.train(batchSize, epochTrain)
    
    
    plot_results(results)