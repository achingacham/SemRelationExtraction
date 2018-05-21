import random
import torch
import re
import numpy as np
import ipdb


class modelData:
    
    def __init__(self, blessFile, tag):

        with open(blessFile) as inputFile:
                
                content = []
                self.validationList = []

                for line in inputFile:
                    line = line.strip('\n')
                    tempList = line.split()
                    self.validationList.append(tempList[0]+':::'+tempList[1])
                    content.append(line)
                     
                    
                totalData = len(content)
                random.shuffle(content)
                #60% train, 10% dev, 30% test
                self.trainData  = content[:int(totalData*.6)]
                self.devData    = content[int(totalData*.6):int(totalData*.7)]
                self.testData   = content[int(totalData*.7):]
                
        
        inputFile.close()
        self.tag = tag
        self.devCount = len(self.devData)
        self.trainCount = len(self.trainData)
        self.testCount = len(self.testData)
        
        print(" Dataset size: \n Train: ",self.trainCount,"\n Test: ", self.testCount, " \n Validation :", self.devCount)
       
        
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
                    if vec[0] in self.validationList:
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
            
            if re.search("JustRel",self.tag):
                key = target+':::'+relata
                relation_vector = [float(value) for value in self.dictRelVectors[key]]
                
            elif re.search("JustWord",self.tag):
                word1 = target.lower()
                word2 = relata.lower()
                #collect the respective vectors for word 1 & 2
            
                vector_1 = np.array(self.dictWordVectors[word1])
                vector_1 = np.ndarray.astype(vector_1,float)
                vector_2 = np.array(self.dictWordVectors[word2])
                vector_2 = np.ndarray.astype(vector_2,float)
                
                relation_vector = vector_1 - vector_2

            elif re.search("RelWord",self.tag):
                
                word1 = target.lower()
                word2 = relata.lower()
                #collect the respective vectors for word 1 & 2
            
                vector_1 = np.array(self.dictWordVectors[word1])
                vector_1 = np.ndarray.astype(vector_1,float)
                vector_2 = np.array(self.dictWordVectors[word2])
                vector_2 = np.ndarray.astype(vector_2,float)
                
                key = target+':::'+relata
                relation_vector = [float(value) for value in self.dictRelVectors[key]]
                relation_vector += vector_1 - vector_2
                
            else:
                pass

            batch_relation_vector.append(relation_vector)

        return(torch.cuda.FloatTensor(batch_relation_vector))

    def make_batch_target_vector(self, batch_relation):
    
        batch_relation_indices = []

        for relation in batch_relation:
            batch_relation_indices.append(self.labelsToIndex[relation])
            
        return(torch.cuda.LongTensor(batch_relation_indices))

    
    def make_input_vector(self,target,relata):
        
        if re.search("JustRel",self.tag):
            key = target+':::'+relata
            relation_vector = [float(value) for value in self.dictRelVectors[key]]

        elif re.search("JustWord",self.tag):
            word1 = target.lower()
            word2 = relata.lower()
            #collect the respective vectors for word 1 & 2

            vector_1 = np.array(self.dictWordVectors[word1])
            vector_1 = np.ndarray.astype(vector_1,float)
            vector_2 = np.array(self.dictWordVectors[word2])
            vector_2 = np.ndarray.astype(vector_2,float)

            relation_vector = vector_1 - vector_2

        elif re.search("RelWord",self.tag):

            word1 = target.lower()
            word2 = relata.lower()
            #collect the respective vectors for word 1 & 2

            vector_1 = np.array(self.dictWordVectors[word1])
            vector_1 = np.ndarray.astype(vector_1,float)
            vector_2 = np.array(self.dictWordVectors[word2])
            vector_2 = np.ndarray.astype(vector_2,float)

            key = target+':::'+relata
            relation_vector = [float(value) for value in self.dictRelVectors[key]]
            relation_vector += vector_1 - vector_2

        else:
            pass

        tensor = torch.cuda.FloatTensor(relation_vector)

        return(tensor)
