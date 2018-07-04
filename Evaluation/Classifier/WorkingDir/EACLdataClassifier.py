import random
import torch
import re
import numpy as np
import ipdb


class modelData:
    
    def __init__(self, blessFile, embeddingFile):
        
        
        with open(blessFile) as inputFile:
               
            print("\nReading evalution dataset....\t")
            self.validationList = dict()

            for line in inputFile:
                line = line.strip('\n')
                tempList = line.split()
                key = tempList[0]+':::'+tempList[1]
                self.validationList[key] = line
                
        
        inputFile.close()
        
        self.create_dictRelVectors(embeddingFile)
        
        totalData = len(self.content)
        random.shuffle(self.content)
        #60% train, 10% dev, 30% test
        self.trainData  = self.content[:int(totalData*.6)]
        self.devData    = self.content[int(totalData*.6):int(totalData*.7)]
        self.testData   = self.content[int(totalData*.7):]

        
        #self.tag = tag
        self.devCount = len(self.devData)
        self.trainCount = len(self.trainData)
        self.testCount = len(self.testData)
        
        print("Total Noun pairs", len(self.validationList))
        
        print("Mapped dataset size: \n Train: ",self.trainCount,"\n Test: ", self.testCount, " \n Validation :", self.devCount)
       
        
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
        self.content = []
        
        with open(embeddingFile) as inputFile:
    
            print("Reading input embedding file....\t")
            for Vectors in inputFile:
                vec = Vectors.split()
                
                try:
                    vec[2]
                    if vec[0] in self.validationList:
                        
                        relVector = [float(value) for value in vec[1:]]
                        ###
                        #norm1 = np.linalg.norm(relVector,1)
                        #relVector = relVector/norm1
                        ###
                        #print(relVector[10])
                        self.dictRelVectors[vec[0]] = relVector
                        self.content.append(self.validationList[vec[0]])
                        
                except:
                    pass
                
                self.input_dim = len(vec[1:])

        
        inputFile.close()
        
    
    def create_dictWordVectors(self, preTrainedVectors):
        
        self.dictWordVectors = dict()
        
        with open(preTrainedVectors) as inputFile:
            
            print("Reading preTrained vectors file....\t")
              
            for Vectors in inputFile:
               
                vec = Vectors.split()
                
                try:
                    if len(vec) == 400+1:
                        self.dictWordVectors[vec[0]] = vec[1:]
                except:
                    print(vec[0],len(vec))
                    
        inputFile.close()
        

    def make_batch_input_vector(self,batch_target,batch_relata, tag):

        batch_relation_vector = []
        for (target,relata) in zip(batch_target,batch_relata):
            
            if re.search("SVDRel",tag):
                key = target+':::'+relata
                relation_vector = [float(value) for value in self.dictRelVectors[key]]
                
            elif re.search("SkipRel",tag):
                key = target+':::'+relata
                relation_vector = [float(value) for value in self.dictRelVectors[key]]
                
            elif re.search("JustWord",tag):
                word1 = target.lower()
                word2 = relata.lower()
                #collect the respective vectors for word 1 & 2
            
                vector_1 = np.array(self.dictWordVectors[word1])
                vector_1 = np.ndarray.astype(vector_1,float)
                vector_2 = np.array(self.dictWordVectors[word2])
                vector_2 = np.ndarray.astype(vector_2,float)
                
                relation_vector = vector_1 - vector_2

            elif re.search("RelWord",tag):
                
                word1 = target.lower()
                word2 = relata.lower()
                #collect the respective vectors for word 1 & 2
            
                vector_1 = np.array(self.dictWordVectors[word1])
                vector_1 = np.ndarray.astype(vector_1,float)
                vector_2 = np.array(self.dictWordVectors[word2])
                vector_2 = np.ndarray.astype(vector_2,float)
                

                key = target+':::'+relata
                
                relation_vector = [float(value) for value in self.dictRelVectors[key]]
                relation_vector.extend((vector_1-vector_2).tolist())
                
                #relation_vector += vector_1 - vector_2
                
            else:
                pass

            batch_relation_vector.append(relation_vector)

        return(torch.cuda.FloatTensor(batch_relation_vector))

    def make_batch_target_vector(self, batch_relation):
    
        batch_relation_indices = []

        for relation in batch_relation:
            batch_relation_indices.append(self.labelsToIndex[relation])
            
        return(torch.cuda.LongTensor(batch_relation_indices))

    
    def make_input_vector(self,target,relata,tag):
        
        if re.search("SkipRel",tag):
            key = target+':::'+relata
            #relation_vector = [float(value) for value in self.dictRelVectors[key]]
            relation_vector = [float(value) for value in self.dictRelVectors[key]]
                

        if re.search("SVDRel",tag):
            key = target+':::'+relata
            #relation_vector = [float(value) for value in self.dictRelVectors[key]]
            relation_vector = [float(value) for value in self.dictRelVectors[key]]
            
        elif re.search("JustWord",tag):
            word1 = target.lower()
            word2 = relata.lower()
            #collect the respective vectors for word 1 & 2

            vector_1 = np.array(self.dictWordVectors[word1])
            vector_1 = np.ndarray.astype(vector_1,float)
            vector_2 = np.array(self.dictWordVectors[word2])
            vector_2 = np.ndarray.astype(vector_2,float)

            relation_vector = vector_1 - vector_2

        elif re.search("RelWord",tag):

            word1 = target.lower()
            word2 = relata.lower()
            #collect the respective vectors for word 1 & 2

            vector_1 = np.array(self.dictWordVectors[word1])
            vector_1 = np.ndarray.astype(vector_1,float)
            vector_2 = np.array(self.dictWordVectors[word2])
            vector_2 = np.ndarray.astype(vector_2,float)

            key = target+':::'+relata
            
            relation_vector = [float(value) for value in self.dictRelVectors[key]]
            relation_vector.extend((vector_1-vector_2).tolist())
            
        else:
            pass

        tensor = torch.cuda.FloatTensor(relation_vector)

        return(tensor)
