import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


class relationMapper(nn.Module):
    '''
        relationMapper is a simple mapping function to map word vectors of a pair(w1,w2) to a single relation(Rel_w1,w2) vector
        
        inputs:
            
            Concatenation of w1_Vec and w2_Vec
            
    '''
    
    def __init__(self, wordVecDim, relVecDim):
        
        nn.Module.__init__(self)
        
        self.mappingModel = nn.Linear(2*wordVecDim, relVecDim)
        
    
    def forward(self, batchInputVectors):
        
        Output = self.mappingModel(batchInputVectors)
        batchOutput = F.relu(Output)
        
        return batchOutput
        