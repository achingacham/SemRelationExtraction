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

class modelTrain:
    
    def __init__(self, inputData, model, loss, optimizer, outfolder,logfile):
       
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
        
        try:
            os.remove(outfolder+logfile)
        except:
            print(" Log file does n't exist. Creating now..")

        self.logs = open(outfolder+logfile,"a")
    
        self.logs.write("\n\nTraining : at \t" + str(time.time()))
    
    def plot_results(self, epoch_train, epoch_dev, epoch_test ,xlabel, ylabel, plotFile):
    
        matpy.plot(epoch_train, label="Train")
        matpy.plot(epoch_dev , label="Dev")
        matpy.plot(epoch_test, label="Test")
        matpy.ylabel(ylabel)
        matpy.xlabel(xlabel)
        matpy.legend()
        matpy.savefig(plotFile)
        matpy.close()
    
    def train(self, batchSize, epochs):
        
        for epoch in range(epochs):
            #SPlit dataset to avoid lexical memorization
            self.inputData.shuffle_data()
            testClassDist = dict()
            
            batchCount = math.ceil(len(self.inputData.trainData)/batchSize)
            
            Train_Error_cost = []
            Average_cost = 0

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
                    
                batch_input_vector = autograd.Variable(self.inputData.make_batch_input_vector(batch_concept,batch_relata), requires_grad = True)
                batch_target_label = autograd.Variable(self.inputData.make_batch_target_vector(batch_relation))

                self.model.zero_grad()
                
                batch_log_prob = self.model(batch_input_vector)
                batch_cost = self.loss(batch_log_prob,batch_target_label)
                print(batch_target_label, batch_cost)
                
                '''
                tempW = [param.data for param in self.model.parameters()]
                temp1 = torch.tanh(batch_input_vector.data @ tempW[0]) + tempW[1]
                temp2 = temp1 @ torch.transpose(tempW[2], 0,1) + tempW[3]
                
                print(batch_concept, batch_relata, batch_relation)
                print(batch_log_prob)
                print(batch_cost)
                
                ipdb.set_trace()
                '''
                
                
                ### Regularize before back propogation
                '''
                tempParams = autograd.Variable(torch.cuda.FloatTensor([0]))
                
                for params in self.model.parameters():
                    tempParams += torch.sum(params**2)
                Lambda = 0.3
                batch_cost -= Lambda *  tempParams
                print(batch_cost)
                '''
                ###
                print(self.model.linear_hidden.bias)
                batch_cost.backward()
                self.optimizer.step()
                print(self.model.linear_hidden.bias)
                
                Train_Error_cost.append(batch_cost.data.tolist())
                Average_cost += batch_cost.data.tolist().pop()

                #print(batch_cost.data, Train_Error_cost, Average_cost)
            
            temp = Average_cost/batchCount

            print("Epoch :", epoch)

            self.logs.write("\n Epoch :"+str(epoch)+", \tAverage cost:"+str(temp)+"\n")

            self.epochCostTrain.append(temp)

            #self.validate()
            
            self.test()

            
        print("Per-class Train samples", testClassDist)
        self.logs.close()    
        
        self.plot_results(self.epochCostTrain, self.epochCostDev, self.epochCostTest,  "Epochs"," Average Cost",self.outfolder+"Cost_justRelVectors.png")
        self.plot_results(self.accuracyTrain, self.accuracyDev, self.accuracyTest,  "Epochs", "Accuracy ",self.outfolder+"Accuracy_justRelVectors.png")
        
        #self.test() Final test
    
    def validate(self):
    
        count = 0
        epochCost = 0
        
        #logs.write("\n\nValidating : \n Input \t\t\t Prediction \t Actual")

        for data in self.inputData.devData:

            split_data = data.split()

            tempConcept = split_data[0]
            tempRelata = split_data[1]
            tempRelation = split_data[2]


            input_vector = autograd.Variable(self.inputData.make_input_vector(tempConcept,tempRelata))
            target_label = autograd.Variable(torch.cuda.LongTensor([self.inputData.labelsToIndex[tempRelation]]))

            log_prob = self.model(input_vector)

            predict_label = log_prob.max(0)[1]

            log_prob = log_prob.view(1,-1)
            dev_cost = self.loss(log_prob,target_label)
            epochCost += dev_cost.data

            if(str(predict_label.data) == str(target_label.data)):     
                count += 1

            prediction = self.inputData.indexToLabels[int(predict_label.data)]
            #self.logs.write("\n"+tempConcept+":"+tempRelata+"\t\t\t"+prediction+"\t"+tempRelation)

        accuracyT = (count/self.inputData.devCount)*100
        devAvgCost = epochCost/self.inputData.devCount
        
        print("Dev Accuracy :",accuracyT ,"\t")
        
        self.logs.write("\n Dev Accuracy : "+str(accuracyT))
        self.logs.write("\n Average Dev Cost : "+str(devAvgCost))
        
        self.epochCostDev.append(devAvgCost)
        self.accuracyDev.append(accuracyT)

     
    def test(self):

        epochCost = 0
        
        indexToLabels = self.inputData.indexToLabels
        perClassCount = dict.fromkeys(indexToLabels, 0)
        perClassPrediction = dict.fromkeys(indexToLabels, 0)
        
        ###
        length = len(indexToLabels)
        confusionMatrix = np.zeros((length,length))
        #print(confusionMatrix)
        ###
        self.logs.write("\n\n Testing : \n Input \t\t\t Prediction \t Actual")


        for data in self.inputData.testData:

            split_data = data.split()

            tempConcept = split_data[0]
            tempRelata = split_data[1]
            tempRelation = split_data[2]

            ###
            tempIndex = self.inputData.labelsToIndex[tempRelation]

            input_vector = autograd.Variable(self.inputData.make_input_vector(tempConcept,tempRelata))
            target_label = autograd.Variable(torch.cuda.LongTensor([tempIndex]))
            ###

            log_prob = self.model(input_vector)
            predict_label = log_prob.max(0)[1]

            log_prob = log_prob.view(1,-1)
            test_cost = self.loss(log_prob,target_label)
            epochCost += test_cost.data
            

            ###
            confusionMatrix[predict_label.data[0]][target_label.data[0]] += 1
            
            perClassCount[tempIndex] += 1

            if(str(predict_label.data) == str(target_label.data)):  
                perClassPrediction[tempIndex] += 1

            ###

            prediction = self.inputData.indexToLabels[int(predict_label.data)]
            self.logs.write("\n"+tempConcept+":"+tempRelata+"\t\t\t"+prediction+"\t"+tempRelation)

        ###
        
        
        classCount = np.fromiter(perClassCount.values(), dtype=int)
        classPrediction = np.fromiter(perClassPrediction.values(), dtype=int)

        # create plot
        fig, ax = matpy.subplots()
        bar_width = 0.75
        opacity = 0.8    
        index = np.arange(self.model.label_size)
        rect1 = matpy.bar(index, classCount, bar_width, alpha=opacity, color='b', label='per Class Targets')
        rect2 = matpy.bar(index, classPrediction, bar_width, alpha=opacity, color='g', label='per Class Correct Predictions')
        matpy.xlabel('Relation types')
        matpy.ylabel('Counts')
        matpy.title('Per class predictions and total counts')
        matpy.xticks(index, self.inputData.indexToLabels.values())
        matpy.legend()
        matpy.savefig(self.outfolder+"PerClass_testSet.png")
        fig.clear()
        
        accuracyT = np.sum(classPrediction)/np.sum(classCount)*100
        print("\t Test Accuracy", accuracyT)
        self.logs.write("\n Test Accuracy :"+str(accuracyT))
        #self.logs.close()

        testAvgCost = epochCost/self.inputData.testCount
        
        self.accuracyTest.append(accuracyT)
        self.epochCostTest.append(testAvgCost)
        print("\n Prediction on Rows and Actual on Columns")
        print(self.inputData.indexToLabels)
        print(confusionMatrix)


            

        