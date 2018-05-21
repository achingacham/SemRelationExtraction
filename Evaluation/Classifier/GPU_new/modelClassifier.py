import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RelationClassifier(nn.Module):
    # Module which implements the model
    def __init__ (self,input_dim, hidden_dim, output_dim):
        nn.Module.__init__(self)
        self.label_size = output_dim
        self.linear_input = nn.Linear(input_dim,hidden_dim)
        self.linear_hidden = nn.Linear(hidden_dim,output_dim)
       
    def forward(self, batch_input_vector):
        
        #batch_size = len(batch_input_vector) 
        #batch_op = np.zeros((batch_size,self.label_size)) #how is it 8?
        #batch_output = autograd.Variable(torch.cuda.FloatTensor(batch_op), requires_grad=True)
       
       
        hidden_layer = self.linear_input(batch_input_vector)
        hidden_units = F.tanh(hidden_layer)
        batch_output = self.linear_hidden(hidden_units)
        #batch_output = F.tanh(batch_output)  ### is the activation function correct? is it required here?
        #print(" Size")
        #print(" hidden_layer ", hidden_layer.requires_grad)
        #print(" hidden_units ", hidden_units.requires_grad)
        #print(" batch_output ", batch_output.requires_grad)
        
        return(F.log_softmax(batch_output))

