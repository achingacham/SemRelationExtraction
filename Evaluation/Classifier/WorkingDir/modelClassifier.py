import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RelationClassifier(nn.Module):
    
    """
    Single layer softmax classifier for relation classification
    
    """
    # Module which implements the model
    def __init__ (self,input_dim, hidden_dim, output_dim):
        nn.Module.__init__(self)
        self.label_size = output_dim
        self.linear_input = nn.Linear(input_dim,hidden_dim)
        self.linear_hidden = nn.Linear(hidden_dim,output_dim)
       
    def forward(self, batch_input_vector):
        
        hidden_layer = self.linear_input(batch_input_vector)
        hidden_units = F.relu(hidden_layer)
        batch_output = self.linear_hidden(hidden_units)
        
        return(F.log_softmax(batch_output))

