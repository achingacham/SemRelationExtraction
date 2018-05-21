
# coding: utf-8

# In[2]:


import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext.data
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as matpy
import math
import sys
import os
import time

torch.cuda.manual_seed(1)


# In[3]:


class RelationClassifier(nn.Module):
    # Module which implements the model
    def __init__ (self,input_dim, hidden_dim, output_dim):
        nn.Module.__init__(self)
        
        self.linear_input = nn.Linear(input_dim,hidden_dim)
        self.linear_hidden = nn.Linear(hidden_dim,output_dim)
        
    def forward(self, batch_input_vector, label_size):
        
        
        hidden_layer = self.linear_input(batch_input_vector)
        hidden_units = F.tanh(hidden_layer)
        batch_output = self.linear_hidden(hidden_units)
       
        return F.log_softmax(self.linear_hidden(hidden_units))
       

class modelData:
    
    def __init__(self, blessFile):

        with open(blessFile) as inputFile:

                content = [line.strip('\n') for line in inputFile.readlines()]

                totalData = len(content)
                random.shuffle(content)
                #60% train, 10% dev, 30% test
                self.trainData  = content[1:int(totalData*.6)]
                self.devData    = content[int(totalData*.6):int(totalData*.7)]
                self.testData   = content[int(totalData*.7):]

        inputFile.close()
        
        
        self.devCount = len(self.devData)
        self.trainCount = len(self.trainData)
        self.testCount = len(self.testData)
        
        print(" Dataset size: \n Train: ",self.trainCount,"\n Test: ", self.testCount, " \n Validation :", self.devCount)
        
    def create_labelsToIndex(self):
        
        self.labelsToIndex = dict()
        self.indexToLabels = dict()
        
        for dataset in [self.trainData, self.testData, self.devData]:
            for data in dataset:
                split_data = data.split(', ')
                tempRelation = split_data[2].strip("[']")

                if tempRelation not in self.labelsToIndex:
                    indexT = len(self.labelsToIndex)
                    self.labelsToIndex[tempRelation] = indexT
                    self.indexToLabels[indexT] = tempRelation

        return len(self.labelsToIndex)

    def shuffle_data(self):
        indices = list(range(self.trainCount))
        random.shuffle(indices)
        newTrainData = [self.trainData[i] for i in indices]
        self.trainData = newTrainData
    
    def create_dictRelVectors(self,embeddingFile):
        
        self.dictRelVectors = dict()
        with open(embeddingFile) as inputFile:
    
            for Vectors in inputFile:
                vec = Vectors.split()
                try:
                    vec[2]
                    self.dictRelVectors[vec[0]] = vec[1:]
                except:
                    
                    input_dim = int(vec[1])
        
        inputFile.close()
        return input_dim
    
    def create_dictWordVectors(self, preTrainedVectors, dim):
        
        self.dictWordVectors = dict()
        
        with open(preTrainedVectors) as inputFile:
            
            for Vectors in inputFile:
               
                vec = Vectors.split()
                
                try:
                    if len(vec) == dim+1:
                        self.dictWordVectors[vec[0]] = vec[1:]
                except:
                    print(vec[0],len(vec))
                    
        inputFile.close()
        
           

    def make_batch_input_vector(self,batch_target,batch_relata):

        batch_relation_vector = []
        for (target,relata) in zip(batch_target,batch_relata):

            word1 = target.lower()
            word2 = relata.lower()
            #collect the respective vectors for word 1 & 2
            
            vector_1 = np.array(self.dictWordVectors[word1])
            vector_1 = np.ndarray.astype(vector_1,float)
            vector_2 = np.array(self.dictWordVectors[word2])
            vector_2 = np.ndarray.astype(vector_2,float)
            
            
            #Combine these two vectors to form a single vector
            #vector_1 - vector_2

            key = target+':::'+relata
            #relation_vector = []
            #relation_vector = [float(value) for value in self.dictRelVectors[key]]
            #relation_vector.extend(vector_1)
            #relation_vector.extend(vector_2)
            
            relation_vector = vector_1 - vector_2
            
            batch_relation_vector.append(relation_vector)


        return(torch.cuda.FloatTensor(batch_relation_vector))

    def make_batch_target_vector(self, batch_relation):
    
        batch_relation_indices = []

        for relation in batch_relation:
            batch_relation_indices.append(self.labelsToIndex[relation])
            
        return(torch.cuda.LongTensor(batch_relation_indices))

    
    def make_input_vector(self,target,relata):

        word1 = target.lower()
        word2 = relata.lower()

        #collect the respective vectors for word 1 & 2
        vector_1 = np.array(self.dictWordVectors[word1])
        vector_1 = np.ndarray.astype(vector_1,float)
        vector_2 = np.array(self.dictWordVectors[word2])
        vector_2 = np.ndarray.astype(vector_2,float)
        
        

        #Combine these two vectors to form a single vector

        key = target+":::"+relata
        #relation_vector = []
        #relation_vector = [float(value) for value in self.dictRelVectors[key]]
        #relation_vector.extend(vector_1)
        #relation_vector.extend(vector_2)
        relation_vector = vector_1 - vector_2
        
        tensor = torch.cuda.FloatTensor(relation_vector)


        return(tensor)


# In[5]:


#TRAINING

def train(labelCount, logfile):

    batchSize = int(sys.argv[4])                   #batch size
    epochCost = []
    epochCostDev = []
    epochs = int(sys.argv[5])
    
    try:
        os.remove(logfile)
    except:
        print(" Log file does n't exist. Creating now..")

    logs = open(logfile,"a")
    
    logs.write("\n\nTraining : at \t" + str(time.time()))
    
    
    for epoch in range(epochs):

        #SPlit dataset to avoid lexical memorization
        
        inputData.shuffle_data()

        batchCount = math.ceil(len(inputData.trainData)/batchSize)

        Train_Error_cost = []
        Average_cost = 0

        for i in range(batchCount):

            batch_entry = inputData.trainData[i*batchSize:i*batchSize+batchSize]
            batch_concept = []
            batch_relata = []
            batch_relation = []

            for index,data in enumerate(batch_entry):

                split_data = data.split(', ')

                tempConcept = split_data[0].strip("[']")
                tempRelata = split_data[1].strip(" '")
                tempRelation = split_data[2].strip("[']")

                batch_concept.append(tempConcept)
                batch_relata.append(tempRelata)
                batch_relation.append(tempRelation)

            #print(batch_concept, batch_relata, batch_relation)
            
            batch_input_vector = autograd.Variable(inputData.make_batch_input_vector(batch_concept,batch_relata), requires_grad = True)
            batch_target_label = autograd.Variable(inputData.make_batch_target_vector(batch_relation))

            
            model.zero_grad() 
            batch_log_prob = model(batch_input_vector, labelCount)
            batch_cost = loss(batch_log_prob,batch_target_label)
            
            batch_cost.backward()
            optimizer.step()
            
            
            Train_Error_cost.append(batch_cost.data.tolist())
            Average_cost += batch_cost.data.tolist().pop()
            
        temp = Average_cost/batchCount
        
        print("Epoch :", epoch)
        #print("\t Average cost :",temp)
        
        logs.write("\n Epoch :"+str(epoch)+", \tAverage cost:"+str(temp))

        epochCost.append(temp)
        
        dev_cost = validate(OUTPUT_LABEL, outfolder+'logfile.log')
        
        epochCostDev.append(dev_cost)
        
    plot_results(epochCostDev, "Dev data",outfolder+"DevPlot_justWordVectors.png") 
        
    logs.close() 
        
    return epochCost
            
            


# In[26]:


def plot_results(epoch_cost,xlabel, plotFile):
    
    matpy.plot(epoch_cost)
    matpy.ylabel("Cost")
    matpy.xlabel(xlabel)
    matpy.savefig(plotFile)
    matpy.close()


# In[27]:


def validate(labelCount, logfile):
    
    count = 0
    Dev_Error_cost = []
    epochCost = 0
    
    logs = open(logfile,"a")
    
    logs.write("\n\nValidating : \n Input \t\t\t Prediction \t Actual")

    for data in inputData.devData:

        split_data = data.split(', ')
        
        
        tempConcept = split_data[0].strip("[']")
        tempRelata = split_data[1].strip(" '")
        tempRelation = split_data[2].strip("[']")

        input_vector = autograd.Variable(inputData.make_input_vector(tempConcept,tempRelata))
        target_label = autograd.Variable(torch.cuda.LongTensor([inputData.labelsToIndex[tempRelation]]))

        log_prob = model(input_vector, labelCount)
        predict_label = log_prob.max(0)[1]
        
        
        log_prob = log_prob.view(1,-1) # resize the tensor for loss calculation only
        dev_cost = loss(log_prob,target_label)
        Dev_Error_cost.append(dev_cost.data.tolist())
        epochCost += dev_cost.data
        
        
        if(str(predict_label.data) == str(target_label.data)):     
            count += 1
            
        prediction = inputData.indexToLabels[int(predict_label.data)]
        logs.write("\n"+tempConcept+":"+tempRelata+"\t\t\t"+prediction+"\t"+tempRelation)
        
    #print("Error for Dev set", Dev_Error_cost)
    accuracyT = (count/len(inputData.devData))*100
    print("Dev Accuracy :",accuracyT, "Average Dev Cost :",epochCost/len(inputData.devData))
    logs.write("\n Dev Accuracy : "+str(accuracyT))
    logs.close()
    
    return epochCost/len(inputData.devData)
    #return Dev_Error_cost

# In[28]:


def test(labelCount, logfile):
    
    #count = 0
    Test_Error_cost = []
    
    ###
    indexToLabels = inputData.indexToLabels
    perClassCount = dict.fromkeys(indexToLabels, 0)
    perClassPrediction = dict.fromkeys(indexToLabels, 0)
    ###
    
    
    logs = open(logfile,"a")
    
    logs.write("\n\n Testing : \n Input \t\t\t Prediction \t Actual")

    for data in inputData.testData:

        split_data = data.split(', ')

        tempConcept = split_data[0].strip("[']")
        tempRelata = split_data[1].strip(" '")
        tempRelation = split_data[2].strip("[']")


        ###
        tempIndex = inputData.labelsToIndex[tempRelation]
        
        input_vector = autograd.Variable(inputData.make_input_vector(tempConcept,tempRelata))
        target_label = autograd.Variable(torch.cuda.LongTensor([tempIndex]))
        ###
        
        log_prob = model(input_vector, labelCount)
        predict_label = log_prob.max(0)[1]

        log_prob = log_prob.view(1,-1)
        test_cost = loss(log_prob,target_label)
        Test_Error_cost.append(test_cost.data.tolist())

        ###
        perClassCount[tempIndex] += 1
        
        if(str(predict_label.data) == str(target_label.data)):  
            perClassPrediction[tempIndex] += 1
            
        ###
        
        prediction = inputData.indexToLabels[int(predict_label.data)]
        logs.write("\n"+tempConcept+":"+tempRelata+"\t\t\t"+prediction+"\t"+tempRelation)
        
        
    ###
    classCount = np.fromiter(perClassCount.values(), dtype=int)
    classPrediction = np.fromiter(perClassPrediction.values(), dtype=int)
    
    # create plot
    fig, ax = matpy.subplots()
    bar_width = 0.75
    opacity = 0.8    
    index = np.arange(labelCount)
    rect1 = matpy.bar(index, classCount, bar_width, alpha=opacity, color='b', label='per Class Targets')
    rect2 = matpy.bar(index, classPrediction, bar_width, alpha=opacity, color='g', label='per Class Correct Predictions')
    matpy.xlabel('Relation types')
    matpy.ylabel('Counts')
    matpy.title('Per class predictions and total counts')
    matpy.xticks(index, inputData.indexToLabels.values())
    matpy.legend()
    matpy.savefig(outfolder+"PerClass_testSet.png")
    fig.clear()
    
    ###
    #print("Error for Test set", Test_Error_cost)
    accuracyT = np.sum(classPrediction)/np.sum(classCount)*100
    print("Test Accuracy", accuracyT)
    
    logs.write("\n Test Accuracy :"+str(accuracyT))
    logs.close()
    
    return Test_Error_cost

# In[29]:



if __name__ == '__main__':
    
    ifolder = sys.argv[1]
    ofolder = sys.argv[2]
    
    outfolder = ofolder+ifolder.rsplit('/',2)[1]+'/Bless/JustWordVectors/'
    
    try:
        os.makedirs(outfolder)
        
    except:
        print("Output folder will be overwritten!")
    
    
    validatedBlessSet = ifolder+"BlessSet.txt"
    validatedRelationEmbedding = ifolder+"EMB_Bless.txt"
    preTrainedWordEmbedding = sys.argv[3]
    
    #DATA

    inputData = modelData(validatedBlessSet)
    #inputDim = inputData.create_dictRelVectors(validatedRelationEmbedding)
    inputData.create_dictWordVectors(preTrainedWordEmbedding, 400)
   

    #MODEL
    INPUT_DIM = 400
    HIDDEN_UNIT = INPUT_DIM
    OUTPUT_LABEL = inputData.create_labelsToIndex()


    model = RelationClassifier(INPUT_DIM, HIDDEN_UNIT, OUTPUT_LABEL)
    loss = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=0.0025)

    if torch.cuda.is_available():
        model.cuda()
        pass
    
    train_cost = train(OUTPUT_LABEL, outfolder+'logfile.log')
    plot_results(train_cost, "Epochs",outfolder+"TrainPlot_justWordVectors.png")
    
    
    test_cost = test(OUTPUT_LABEL, outfolder+'logfile.log')
    #plot_results(test_cost, "Test data",outfolder+"TestPlot_justWordVectors.png")
    
    