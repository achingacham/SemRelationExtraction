
# coding: utf-8

# In[80]:


import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext.data
import random

torch.manual_seed(1)


# In[4]:


class RelationClassifier(nn.Module):
    # Module which implements the model
    def __init__ (self, input_dim, hidden_dim, output_dim):
        nn.Module.__init__(self)
        
        self.linear_input = nn.Linear(input_dim,hidden_dim)
        self.linear_hidden = nn.Linear(hidden_dim,output_dim)
        
    def forward(self, input_vector):
        
        hidden_layer = self.linear_input(input_vector)
        hidden_units = F.tanh(hidden_layer)
        output = self.linear_hidden(hidden_units)
        
        return(F.log_softmax(output).view(1,-1))
    


# In[37]:


Vectors = np.loadtxt("preTrainedVectors_mini.txt",str,comments=None)

dict_Vectors = {}

for vector in Vectors:
    dict_Vectors[vector[0]] = vector[1:401]
    
print(type(dict_Vectors['radio'][0]))


# In[60]:


#Make input vectors

def make_input_vector(target,relata):
    
    word1 = target.lower()
    word2 = relata.lower()
    
    #collect the respective vectors for word 1 & 2
    vector_1 = np.ndarray.astype(dict_Vectors[word1],float)
    vector_2 = np.ndarray.astype(dict_Vectors[word2],float)
    
    #Combine these two vectors to form a single vector
    
    relation_vector = vector_1 - vector_2    
    tensor = torch.Tensor(relation_vector)
    
    return(tensor)


# In[73]:


#SPlit dataset to avoid lexical memorization

with open("UniqueTuples_mini") as inputFile:
    content = inputFile.readlines()
    total_data = len(content) 
    #60% train, 10% dev, 30% test
    random.shuffle(content)
    train_data  = content[0:int(total_data*.6)]
    dev_data    = content[int(total_data*.6):int(total_data*.7)]
    test_data   = content[int(total_data*.7):]
    


# In[74]:


#MODEL

INPUT_DIM = 400
HIDDEN_UNIT = 400
OUTPUT_LABEL = 8

model = RelationClassifier(INPUT_DIM, HIDDEN_UNIT, OUTPUT_LABEL)
loss = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.09)
    
    


# In[75]:


#TRAINING
labels_to_ix = {}

for entry in train_data:
    
    input_words = entry.split()
    
    concept = input_words[0]
    relata = input_words[1]   
    relation = input_words[2]
    
    model.zero_grad()
    
    if relation not in labels_to_ix:
        labels_to_ix[relation] = len(labels_to_ix)
    
    
    input_vector = autograd.Variable(make_input_vector(concept,relata))
    target_label = autograd.Variable(torch.LongTensor([labels_to_ix[relation]]))
    
    log_prob = model(input_vector)
    cost = loss(log_prob,target_label)
    #Make it mini batches
    #print(cost.data.tolist())
    
    cost.backward()
    optimizer.step()



# In[77]:


#VALIDATION

count = 0

for entry in dev_data:
    input_words = entry.split()
    
    concept = input_words[0]
    relata = input_words[1]   
    relation = input_words[2]
    
    input_vector = autograd.Variable(make_input_vector(concept,relata))
    target_label = autograd.Variable(torch.LongTensor([labels_to_ix[relation]]))
    
    log_prob = model(input_vector)
    
    predict_label = log_prob.max(1)[1]
    
    if(str(predict_label.data) == str(target_label.data)):     
        count += 1
        


# In[79]:


print("Accuracy",(count/(total_data*0.1))*100)

