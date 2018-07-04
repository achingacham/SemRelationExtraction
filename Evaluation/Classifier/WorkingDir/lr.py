'''
Module to track the learning rate
'''
import numpy as np

class LearningRate(object):
    

    def __init__(self, current_valid, initial_lr):
        self.valid_current = None
        self.valid_history = current_valid
        self.delta = [0., 0.]
        self.delta_bar = [0., 0.]
        self.learning_rate = initial_lr
        
        self.THETA_LR = 1e-3
        self.DECREMENT_LR = 0.7
        self.INCREMENT_LR = 1.1
        self.ALPHA_LR = 1e-1

    def update_learning_rate(self, new_validation_score):
        '''
            Algorithm to update the learning rate
            Parameters
            ----------
            new_validation_score :  new cross entropy score
        '''
        assert  self.valid_history is not None, "valid_history has not been set"
        self.valid_current = new_validation_score
        self.delta[1] = (self.valid_current - self.valid_history) / self.valid_current
        if self.delta[1]*self.delta_bar[0] < 0 and \
                np.abs(self.delta_bar[0]) > self.THETA_LR:
            self.learning_rate *= self.DECREMENT_LR
        else:
            self.learning_rate *= self.INCREMENT_LR
        self.delta_bar[1] = self.ALPHA_LR*self.delta[1] + \
                (1 - self.ALPHA_LR)*self.delta_bar[0]
        self.delta[0] = self.delta[1]
        self.delta_bar[0] = self.delta_bar[1]
        self.valid_history = self.valid_current
        self.learning_rate = np.float(self.learning_rate)

    def copy(self, lr_obj):
        '''
            Copy the values of lr_obj to self
        '''
        assert isinstance(lr_obj, LearningRate),\
                "The given input is not an LearningRate object"
        self.valid_current = lr_obj.valid_current
        self.valid_history = lr_obj.valid_history
        self.delta = lr_obj.delta
        self.delta_bar = lr_obj.delta_bar
        self.learning_rate = lr_obj.learning_rate
