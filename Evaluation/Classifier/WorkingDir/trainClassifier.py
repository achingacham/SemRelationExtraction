import os
import time
import math
import numpy as np
import torch
import torch.autograd as autograd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as matpy
import ipdb
import re
import csv
from lr import LearningRate
from matplotlib import pyplot as plt 
from sklearn.metrics import confusion_matrix


class modelTrain:
    
    """
    Module for model training, validation and testing
    
    Inputs:
    
    inputData : object consisitng all model data
    model: object containing model itslef
    loss: loss function utilized
    optimizer : optimizer utilized
    outfolder: Final results folder
    logfile: LOGs stored
    tag : tracking the eval file
    
    
    """
    
    def __init__(self, inputData, model, loss, optimizer, outfolder,logfile, tag):
       
        self.epochCostTrain = []
        self.epochCostDev = []
        self.epochCostTest = []
        
        self.accuracyTrain = []
        self.accuracyDev = []
        self.accuracyTest = []
        
        self.inputData = inputData
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.outfolder = outfolder
        self.tag = tag

        try:
            os.remove(outfolder+self.tag+logfile)
        except:
            print(" Log file does n't exist. Creating now..")

        self.logs = open(outfolder+self.tag+logfile,"a")
    
        self.logs.write("\n\nTraining : at \t" + str(time.time()))
        
        self.logs.write("\nEpoch No:\tAverage train Cost\t DevAccuracy\tDevCost\tTestAccuracy\tTestCost")

        try:
            os.remove(outfolder+self.tag+"_ErrorAnalysis.csv")
        except:
            print(" Error analysis file does n't exist. Creating now..")
            
        self.errorAnalysisFile = open(outfolder+self.tag+"_ErrorAnalysis.csv","a")
        
        self.CSVwriter = csv.writer(self.errorAnalysisFile, delimiter="\t", lineterminator="\n")
        
        self.epochPredictions = dict()
    
    def plot_results(self, epoch_train, epoch_dev, epoch_test ,xlabel, ylabel, plotFile):
    
        matpy.plot(epoch_train, label="Train")
        matpy.plot(epoch_dev , label="Dev")
        matpy.plot(epoch_test, label="Test")
        matpy.ylabel(ylabel)
        matpy.xlabel(xlabel)
        matpy.legend()
        matpy.savefig(plotFile)
        matpy.close()
    
    def train(self, batchSize, epochs, l2_factor, initial_lr):
        """
        Model training module
        
        l2_factor : L2 regularization factor
        initial_lr : initial learning rate
        
        """
        
        print("\n\n ", self.tag)
        
        for epoch in range(epochs):
            #SPlit dataset to avoid lexical memorization
            self.inputData.shuffle_data()
            testClassDist = dict()
            
            batchCount = math.ceil(len(self.inputData.trainData)/batchSize)
            
            Train_Error_cost = []
            Average_cost = 0
            
            if epoch != 0:
                lr_object = LearningRate(self.epochCostDev[epoch-1], initial_lr)

            #For every batch, the tensors are created with repect to model type
            
            for i in range(batchCount):

                batch_entry = self.inputData.trainData[i*batchSize:i*batchSize+batchSize]
                batch_concept = []
                batch_relata = []
                batch_relation = []

                for index,data in enumerate(batch_entry):
                    
                    
                    split_data = data.split()
                   
                    tempConcept = split_data[0]
                    tempRelata = split_data[1]
                    tempRelation = split_data[2]

                    batch_concept.append(tempConcept)
                    batch_relata.append(tempRelata)
                    batch_relation.append(tempRelation)
                    
                    ###
                    if tempRelation in testClassDist:
                        testClassDist[tempRelation] += 1
                    else:
                        testClassDist[tempRelation] = 1
                        
                    ###
                    
                
                    
                batch_input_vector = autograd.Variable(self.inputData.make_batch_input_vector(batch_concept,batch_relata, self.tag), requires_grad = True)
                batch_target_label = autograd.Variable(self.inputData.make_batch_target_vector(batch_relation))

                self.model.zero_grad()
                
                batch_log_prob = self.model(batch_input_vector)
                batch_cost = self.loss(batch_log_prob,batch_target_label)
                
               
                
                ### Regularize before back propogation
                
                
                l2_reg = None
                for params in self.model.parameters():
                    if l2_reg is None:
                        l2_reg = params.norm(2)
                    else:
                        l2_reg = l2_reg + params.norm(2)
                
                #if epoch > 1:
                batch_cost += l2_factor * l2_reg
                ###
                
                batch_cost.backward()
                self.optimizer.step()
                
                Train_Error_cost.append(batch_cost.data.tolist())
                Average_cost += batch_cost.data.tolist().pop()

                
            ### Finished all batches in an epoch
            
            temp = Average_cost/batchCount
            
            self.epochCostTrain.append(temp)

            self.validate()
            
            Count,Prediction  = self.test(epoch+1)
            
            
            #### Update learning rate after every Epoch based on validation cost
            if epoch != 0:
                lr_object.update_learning_rate(self.epochCostDev[epoch])
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr_object.learning_rate
                    
            ####
            
            if epoch == 0:
                
                bestTestAccuracy = self.accuracyTest[epoch]
                bestTestEpoch = epoch
                
                bestDevEpoch = epoch
                bestDevCost = self.epochCostDev[epoch]
                bestDevCost = self.epochCostDev[epoch]
                
                classCount = Count
                classPrediction = Prediction
            else:
                
                ### Track the per class counts for the epoch in which lowest Dev cost occured
                
                if bestDevCost > self.epochCostDev[epoch]:


                    bestDevEpoch = epoch
                    bestDevCost = self.epochCostDev[epoch]
                    
                    classCount = Count
                    classPrediction = Prediction
                    bestTestEpoch = epoch
                    bestTestAccuracy = self.accuracyTest[epoch]     
                
                ### Track the per class counts for the epoch in which highest Test accuracy occured
                
                #if bestTestAccuracy < self.accuracyTest[epoch]:

                #    classCount = Count
                #    classPrediction = Prediction
                #    bestTestEpoch = epoch
                #    bestTestAccuracy = self.accuracyTest[epoch]
                            
            self.logs.write("\n"+str(epoch)+":\t"+str(temp)+"\t")
            self.logs.write(str(self.accuracyDev[epoch])+"\t"+str(self.epochCostDev[epoch])+"\t")
            self.logs.write(str(self.accuracyTest[epoch])+"\t"+str(self.epochCostTest[epoch]))
            
            print("\n",epoch,":\t",temp,"\t")
            #print("Dev : ",self.accuracyDev[epoch],"\t",self.epochCostDev[epoch],"\t")
            #print("Test: ",self.accuracyTest[epoch],"\t",self.epochCostTest[epoch])      

        ### Finished all Epochs
        # create plot for representing Per class predictions 
        
        fig, ax = matpy.subplots()
        bar_width = 0.75
        opacity = 0.8    
        index = np.arange(self.model.label_size)
        rect1 = matpy.bar(index, classCount, bar_width, alpha=opacity, color='b', label='per Class Targets')
        rect2 = matpy.bar(index, classPrediction, bar_width, alpha=opacity, color='g', label='per Class Correct Predictions')
        matpy.xlabel('Relation types')
        matpy.ylabel('Counts')
        matpy.title('Best Per class predictions and total counts : '+ self.tag +" at epoch "+ str(bestTestEpoch))
        matpy.xticks(index, self.inputData.indexToLabels.values(), rotation=20, visible=True)
        matpy.legend()
        matpy.savefig(self.outfolder+self.tag+"PerClass_testSet.png")
        fig.clear()
            
        print("\n\nPer-class Train samples :", testClassDist)
        
        print("Best Train accuracy at Epoch ",bestDevEpoch," of minimum dev cost :", self.accuracyTest[bestDevEpoch])
        
        self.logs.write("\n\nPer-class Train samples"+str(testClassDist))
        self.logs.write("\nBest Train accuracy at Epoch "+str(bestDevEpoch)+"of minimum dev cost :"+ str(self.accuracyTest[bestDevEpoch]))
        self.logs.close()    
        
        rel_true = []
        rel_pred = []
        
        ###Error Analysis:
        for item in self.epochPredictions:
            self.CSVwriter.writerow([item, self.epochPredictions[item][bestDevEpoch], self.epochPredictions[item]])
            #print("y_true :",item[2], "y_pred :", self.epochPredictions[item][bestDevEpoch])
            rel_true.append(item[2])
            rel_pred.append(self.epochPredictions[item][bestDevEpoch])
        
        
        self.plot_confusion_matrix(rel_true, rel_pred)
        self.errorAnalysisFile.close()    
        
        return ([[self.epochCostTrain, self.epochCostDev, self.epochCostTest,  "Epochs"," Average Cost",self.tag], [self.accuracyTrain, self.accuracyDev, self.accuracyTest,  "Epochs", "Accuracy ",self.tag]])
        
       
    def validate(self):
    
        count = 0
        epochCost = 0
        
        #logs.write("\n\nValidating : \n Input \t\t\t Prediction \t Actual")

        for data in self.inputData.devData:

            split_data = data.split()

            tempConcept = split_data[0]
            tempRelata = split_data[1]
            tempRelation = split_data[2]


            input_vector = autograd.Variable(self.inputData.make_input_vector(tempConcept,tempRelata,self.tag))
            target_label = autograd.Variable(torch.cuda.LongTensor([self.inputData.labelsToIndex[tempRelation]]))
            
            log_prob = self.model(input_vector)

            predict_label = log_prob.max(0)[1]

            log_prob = log_prob.view(1,-1)
            dev_cost = self.loss(log_prob,target_label)
            epochCost += dev_cost.data

            
            if(predict_label.data[0] == target_label.data[0]):     
                count += 1

            
            prediction = self.inputData.indexToLabels[int(predict_label.data)]
            #self.logs.write("\n"+tempConcept+":"+tempRelata+"\t\t\t"+prediction+"\t"+tempRelation)

        accuracyT = (count/self.inputData.devCount)*100
        devAvgCost = epochCost/self.inputData.devCount
        
        
        self.epochCostDev.append(devAvgCost[0])
        self.accuracyDev.append(accuracyT)

    
    def plot_confusion_matrix(self, rel_true, rel_pred, filename="_ConfusionMatrix.png", cmap=plt.cm.Blues):
        confusion =  confusion_matrix(rel_true, rel_pred)
        
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        
        image = ax.matshow(confusion, cmap=cmap) # imshow
        fig.colorbar(image)
        tick_marks = np.arange(confusion.shape[0])
        
        ax.set_xlabel("Predicted")
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(np.unique(rel_true))
    
        ax.set_ylabel("Actual")
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(np.unique(rel_true))

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        plt.savefig(self.outfolder+self.tag+filename, bbox_inches='tight')

     
    def test(self, epoch):

        epochCost = 0
        
        indexToLabels = self.inputData.indexToLabels
        perClassCount = dict.fromkeys(indexToLabels, 0)
        perClassPrediction = dict.fromkeys(indexToLabels, 0)
        
        ###
        length = len(indexToLabels)
        ###
        
        
        #self.errorAnalysisFile.write("\n\n Epoch: "+str(epoch)+"\n"+"Word1 \t\t Word2 \t\t Actual Label \t\t Prediction"+"\n")
        
        
        for data in self.inputData.testData:

            split_data = data.split()

            tempConcept = split_data[0]
            tempRelata = split_data[1]
            tempRelation = split_data[2]

            ###
            tempIndex = self.inputData.labelsToIndex[tempRelation]

            input_vector = autograd.Variable(self.inputData.make_input_vector(tempConcept,tempRelata,self.tag))
            target_label = autograd.Variable(torch.cuda.LongTensor([tempIndex]))
            ###

            log_prob = self.model(input_vector)
            predict_label = log_prob.max(0)[1]

            log_prob = log_prob.view(1,-1)
            test_cost = self.loss(log_prob,target_label)
            epochCost += test_cost.data
            
            
            ###
            perClassCount[tempIndex] += 1

             
            if(predict_label.data[0] == target_label.data[0]):  
                perClassPrediction[tempIndex] += 1

            ###

            prediction = self.inputData.indexToLabels[int(predict_label.data[0])]
            
            ###Error - analysis
            if epoch == 1:
                self.epochPredictions[tempConcept, tempRelata, tempRelation] = [prediction]
            else:
                self.epochPredictions[tempConcept, tempRelata, tempRelation].append(prediction)
            

            
        ###
        
        
        classCount = np.fromiter(perClassCount.values(), dtype=int)
        classPrediction = np.fromiter(perClassPrediction.values(), dtype=int)

        
        accuracyT = np.sum(classPrediction)/np.sum(classCount)*100
        testAvgCost = epochCost/self.inputData.testCount
        
        self.accuracyTest.append(accuracyT)
        self.epochCostTest.append(testAvgCost[0])
        
        
        return (classCount, classPrediction)

            

        
