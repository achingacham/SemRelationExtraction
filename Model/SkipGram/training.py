from model import SkipGramModel
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys
import os
import re
import random
import cProfile
import ipdb
import numpy as np

class Word2Vec:
    
    def __init__(self, ifolder, ofolder, 
                 emb_dimension=400,
                 batch_size=32,
                 iteration=int(sys.argv[3]),
                 initial_lr=0.025):
        
        self.ifolder = ifolder
        
        self.outfolder = ofolder+ifolder.rsplit('/',2)[1]+'/'
        try:
            os.mkdir(self.outfolder)
        except:
            print(self.outfolder+ " folder exists. Will be overwritten")
        
        self.emb_dimension = emb_dimension
        self.initial_lr = initial_lr
        self.iteration = iteration
        self.batch_size = batch_size
        self.fpos = 0
        self.fneg = 0
        
        self.id2word = dict()
        self.id2pair = dict()
        self.pair2id = dict()
        
        self.read_word_dict(ifolder+"Word2Id")
        self.read_pair_dict(ifolder+"Pair2Id")
        
        self.pair_count = self.evaluate_pair_count()
        
        #self.positive_pairs = [None] * self.pair_count        
        self.positive_pairs = [[3242323, 13213121]] * self.pair_count
       #[i for i in range(self.pair_count)]
        # Dummy values to ensure size does not change
        self.negative_pairs = [[3242323,3242323,3242323,3242323,3242323]] * self.pair_count
        #self.negative_pairs = [None] * self.pair_count
        #[i for i in range(self.pair_count)]
        
        #ipdb.set_trace()
        self.emb_size     = len(self.id2word)
        self.pair_emb_size = len(self.id2pair)
        
        
        self.skip_gram_model = SkipGramModel(self.pair_emb_size,self.emb_size, self.emb_dimension)
        self.use_cuda = torch.cuda.is_available()
        
        if self.use_cuda:
            self.skip_gram_model.cuda()
        self.optimizer = optim.SGD(self.skip_gram_model.parameters(), lr=self.initial_lr)
        
        print("Start reading pairs")
        
    def read_word_dict(self, wdictfile ):
        
        with open(wdictfile) as inputFile:
            
            for item in inputFile:
                word,wid = item.split('\t')
                self.id2word[int(wid)] = word
                          
        print("\n Completed reading word dictionary.")
    def read_pair_dict(self, pdictfile ):
        
        with open(pdictfile) as inputFile:
            
            for item in inputFile:
                word1,word2,pid = item.split('\t')
               
                self.id2pair[int(pid)] = (word1,word2)
                self.pair2id[(word1,word2)] = int(pid)
                print(self.id2pair[int(pid)])
        print("\n Completed reading pair dictionary.")
        
        self.cross_verification()
        
    def evaluate_pair_count(self):
        
        self.datasets = dict()
        
        dsfile = self.ifolder+"Statistics"
        with open(dsfile) as inputFile:
             
            for item in inputFile:
                
                if re.match("Dataset",item):
                    i = item.split(':')[1]
                    
        print("Total positive pair samples :",i)        
        return int(i)
    
    def read_pairs(self, posFile, negFile):
        
        posDsfile = self.ifolder+posFile
        
        index = 0
        #ipdb.set_trace()
        with open(posDsfile) as inputFile:
            
            for line in inputFile:
                #pid = line.split(',')[0].strip('( )\n')
                #wid = line.split(',')[1].strip('( )\n')
                pid, wid = line.strip('( )\n').split(',')
                #self.positive_pairs.append([int(pid),int(wid)])
                #self.positive_pairs[index] = (int(pid),int(wid))
                self.positive_pairs[index][0] = int(pid)
                self.positive_pairs[index][1] = int(wid)
                index += 1
        print("Size of :", sys.getsizeof(self.positive_pairs))
        
        negDsfile = self.ifolder+negFile
        
        index = 0
        with open(negDsfile) as inputFile:
            for line in inputFile:
                #self.negative_pairs.append([int(i.strip('[ ] \n')) for i in line.split(',')])
                #self.negative_pairs[index] = tuple([int(i) for i in line.strip('[ ] \n').split(',')])
                temp = [int(i) for i in line.strip('[ ] \n').split(',')]
                for temp_index,temp_value in enumerate(temp):
                    self.negative_pairs[index][temp_index]=temp_value
                index += 1
        print(" Size of :", sys.getsizeof(self.negative_pairs))
        
    def get_batch_pairs(self, batch_count):
        
        return self.positive_pairs[(batch_count)*self.batch_size:(batch_count+1)*self.batch_size]
        
        
    def get_neg_v(self, batch_count):
        
        return self.negative_pairs[(batch_count)*self.batch_size:(batch_count+1)*self.batch_size]
        
    
    def cross_verification(self):

        #Remove the file if it already exists
        try:
            os.remove(self.outfolder+"BlessSet.txt")
        except:
            pass
        
        #Remove the file if it already exists
        try:
            os.remove(self.outfolder+"BlessSet_Except.txt")
        except:
            pass
        
        blessExceptFile = open(self.outfolder+"BlessSet_Except.txt","w")
        blessFile = open(self.outfolder+"BlessSet.txt","w")
        
        self.Bless_id2pair = dict()
        
        with open("/home/achingacham/Model/GRID_data/Evaluation_Datasets/BLESS/UniqueTuples") as evalFile:
            testDataset = evalFile.readlines()
            
            for items in testDataset:
                nouns = items.split()
                search_key = (nouns[0],nouns[1])
                rev_search_key = (nouns[1],nouns[0])
                
                if (search_key in self.pair2id):
                    temp_id = self.pair2id[search_key]
                    self.Bless_id2pair[temp_id] = str(nouns[0])+':::'+str(nouns[1])
                    blessFile.write(str(nouns)+"\n")
                
                else:
                    blessExceptFile.write(str(nouns)+"\n")
                    
                '''
                elif (rev_search_key in self.pair2id):
                    temp_id = self.pair2id[rev_search_key]
                    self.Bless_id2pair[temp_id] = nouns[1]+':::'+nouns[0]
                    
                    blessFile.write("\n"+str(nouns))
                else:
                    pass
                    
                '''
        
        print("Completed cross validation with Blessset")
        evalFile.close()
        blessFile.close()

    def train(self):
        """Multiple training.
        Returns:
            None.
        """
        
        
        batch_count = self.pair_count / self.batch_size
            
        for epoch in range(self.iteration):
            
            print("\n Epoch :", epoch)
            
            output_file_name = self.outfolder+"Epoch_"+str(epoch)+"_EMB_All.txt"
            Bless_output_file_name = self.outfolder+"Epoch_"+str(epoch)+"_EMB_Bless.txt"
        
            epochLoss = 0
            
            process_bar = tqdm(range(int(batch_count)))
            
            for i in process_bar:
            
                pos_pairs = self.get_batch_pairs(i)
                neg_v = self.get_neg_v(i) 
                
                pos_u = [pair[0] for pair in pos_pairs]   #index to the pair of Nouns
                pos_v = [pair[1] for pair in pos_pairs]   #a context word (for instance, inbetween word)

                pos_u = Variable(torch.LongTensor(pos_u))
                pos_v = Variable(torch.LongTensor(pos_v))

                neg_v = Variable(torch.LongTensor(neg_v)) #a negative context word from unigram distribution
                
                      
                if self.use_cuda:
                    pos_u = pos_u.cuda()
                    pos_v = pos_v.cuda()
                    neg_v = neg_v.cuda()

                
                
                self.optimizer.zero_grad()
                loss = self.skip_gram_model.forward(pos_u, pos_v, neg_v)
                
                loss.backward()
                self.optimizer.step()

                process_bar.set_description("Loss: %0.8f, lr: %0.6f" %
                                            (loss.data[0],self.optimizer.param_groups[0]['lr']))
                
                epochLoss += loss.data[0]
                
                if i * self.batch_size % 100000 == 0:
                    lr = self.initial_lr * (1.0 - 1.0 * i / batch_count)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr

            print("\n Average Epoch Loss: ", epochLoss/batch_count)
            
            self.skip_gram_model.save_embedding(self.id2pair, output_file_name, self.use_cuda)    
            self.skip_gram_model.save_embedding(self.Bless_id2pair, Bless_output_file_name, self.use_cuda)    
                
            
if __name__ == '__main__':
    w2v = Word2Vec(ifolder=sys.argv[1], ofolder=sys.argv[2])
    
    if w2v.pair_emb_size > 0 :
        w2v.read_pairs("Triplesets_positive","Triplesets_negative")
        w2v.train()
        pass
    else:
         print("Unable to train, doesn't have enough pair count")
