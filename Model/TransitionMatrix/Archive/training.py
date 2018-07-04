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
            os.makedirs(self.outfolder)
        except:
            print(self.outfolder+ " folder exists. Will be overwritten")
        
        self.emb_dimension = emb_dimension
        self.pair_emb_dimension = emb_dimension
            
        self.initial_lr = initial_lr
        self.iteration = iteration
        self.batch_size = batch_size
        self.fpos = 0
        self.fneg = 0
        
        self.id2word = dict()
        self.word2id = dict()
        self.id2pair = dict()
        self.pair2id = dict()
        
        self.read_word_dict(ifolder+"Word2Id")
        self.read_pair_dict(ifolder+"Pair2Id")
        
        self.pair_count = self.evaluate_pair_count()
        #self.positive_pairs = [[3242323, 13213121]] * self.pair_count
        self.positive_pairs = np.zeros((self.pair_count, 2), dtype=int)
       
        # Dummy values to ensure size does not change
        #self.negative_pairs = [[3242323,3242323,3242323,3242323,3242323]] * self.pair_count
        self.negative_pairs = np.zeros((self.pair_count, 5), dtype=int)
        
        print(" Size of :", sys.getsizeof(self.positive_pairs))
        print(" Size of :", sys.getsizeof(self.negative_pairs))
        #ipdb.set_trace()
        
        self.emb_size     = len(self.id2word)
        self.pair_emb_size = len(self.id2pair)
        
        
        self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension, self.pair_emb_dimension)
        self.use_cuda = torch.cuda.is_available()
        
        if self.use_cuda:
            self.skip_gram_model.cuda()
        self.optimizer = optim.SGD(self.skip_gram_model.parameters(), lr=self.initial_lr)
        
        print("Start reading pairs")
        
    def read_word_dict(self, wdictfile ):
        
        with open(wdictfile) as inputFile:
            
            for item in inputFile:
                word,wid = item.split()
                self.id2word[int(wid)] = word
                self.word2id[word] = int(wid)
                          
        print("\n Completed reading word dictionary.")
    def read_pair_dict(self, pdictfile ):
        
        with open(pdictfile) as inputFile:
            
            for item in inputFile:
                word1,word2,pid = item.split()
                
                wid1 = self.word2id[word1]
                wid2 = self.word2id[word2] 
                
                self.id2pair[int(pid)] = (wid1,wid2)
                self.pair2id[(wid1,wid2)] = int(pid)
                
                #print(self.id2pair[int(pid)],word1+':::'+word2)
        print("\n Completed reading pair dictionary.")
        
        #self.cross_verification_BLESS()
        #self.cross_verification_EVAL()
        
        
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
                self.positive_pairs[index] = [int(pid),int(wid)]
                index += 1
        print("Size of :", sys.getsizeof(self.positive_pairs))
        
        negDsfile = self.ifolder+negFile
        
        index = 0
        with open(negDsfile) as inputFile:
            for line in inputFile:
                temp = [int(i) for i in line.strip('[ ] \n').split(',')]
                self.negative_pairs[index] = temp
                index += 1
        print(" Size of :", sys.getsizeof(self.negative_pairs))
        
    def get_batch_pairs(self, batch_count):
        
        return self.positive_pairs[(batch_count)*self.batch_size:(batch_count+1)*self.batch_size]
        
        
    def get_neg_v(self, batch_count):
        
        return self.negative_pairs[(batch_count)*self.batch_size:(batch_count+1)*self.batch_size]
        
    
    def cross_verification_BLESS(self):

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
        
        with open("/home/achingacham/Model/GRID_data/Evaluation_Datasets/BLESS_UniqueTuples") as evalFile:
            testDataset = evalFile.readlines()
            
            for items in testDataset:
                nouns = items.split()
                search_key = (nouns[0],nouns[1])
                rev_search_key = (nouns[1],nouns[0])
                
                if (search_key in self.pair2id):
                    temp_id = self.pair2id[search_key]
                    self.Bless_id2pair[temp_id] = nouns[0]+':::'+nouns[1]
                    blessFile.write(items)
                
                else:
                    blessExceptFile.write(items)                
               
        
        print("Completed cross validation with Blessset")
        blessExceptFile.close()
        blessFile.close()

    def cross_verification_EVAL(self):

        #Remove the file if it already exists
        try:
            os.remove(self.outfolder+"EvalSet.txt")
        except:
            pass
        
        #Remove the file if it already exists
        try:
            os.remove(self.outfolder+"EvalSet_Except.txt")
        except:
            pass
        
        EVALExceptFile = open(self.outfolder+"EvalSet_Except.txt","w")
        EVALFile = open(self.outfolder+"EvalSet.txt","w")
        
        self.Eval_id2pair = dict()
        
        with open("/home/achingacham/Model/GRID_data/Evaluation_Datasets/EVAL_UniqueTuples") as evalFile:
            testDataset = evalFile.readlines()
            
            for items in testDataset:
                nouns = items.split()
                search_key = (nouns[0],nouns[1])
                rev_search_key = (nouns[1],nouns[0])
                
                if (search_key in self.pair2id):
                    temp_id = self.pair2id[search_key]
                    self.Eval_id2pair[temp_id] = nouns[0]+':::'+nouns[1]
                    EVALFile.write(items)
                
                else:
                    EVALExceptFile.write(items)                
               
        
        print("Completed cross validation with Blessset")
        EVALExceptFile.close()
        EVALFile.close()
        

        
    def train(self):
        """Multiple training.
        Returns:
            None.
        """
        
        batch_count = self.pair_count / self.batch_size
            
        for epoch in range(self.iteration):
            
            print("\n Epoch :", epoch)
            
            output_file_name = self.outfolder+"Epoch_"+str(epoch)+"_EMB_"+str(self.emb_dimension)+"_All.txt"
            #Bless_output_file_name = self.outfolder+"Epoch_"+str(epoch)+"_EMB_"+str(self.emb_dimension)+"_Bless.txt"
        
            epochLoss = 0
            
            process_bar = tqdm(range(int(batch_count)))
            
            for i in process_bar:
            
                pos_pairs = self.get_batch_pairs(i)
                neg_v = self.get_neg_v(i) 
                
                
                pos_u = np.array([pair[0] for pair in pos_pairs])   #index to the pair of Nouns
                
                pos_u_wid = np.array([self.id2pair[pid] for pid in pos_u])
                
                pos_v = np.array([pair[1] for pair in pos_pairs])   #a context word (for instance, inbetween word)
                
                
                
                #pos_u = Variable(torch.LongTensor(pos_u))
                pos_u_wid = Variable(torch.LongTensor(pos_u_wid))
                pos_v = Variable(torch.LongTensor(pos_v))
                neg_v = Variable(torch.LongTensor(neg_v)) #a negative context word from unigram distribution
                
                      
                if self.use_cuda:
                    pos_u_wid = pos_u_wid.cuda()
                    pos_v = pos_v.cuda()
                    neg_v = neg_v.cuda()

                
              
                self.optimizer.zero_grad()
                loss = self.skip_gram_model.forward(pos_u_wid, pos_v, neg_v)
                
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
            
            self.skip_gram_model.save_embedding(self.id2word, output_file_name, self.use_cuda)    
            #self.skip_gram_model.save_embedding(self.Bless_id2pair, Bless_output_file_name, self.use_cuda)    
                
            
if __name__ == '__main__':
    w2v = Word2Vec(ifolder=sys.argv[1], ofolder=sys.argv[2])
    
    if w2v.pair_emb_size > 0 :
        w2v.read_pairs("Triplesets_positive","Triplesets_negative")
        w2v.train()
        pass
    else:
         print("Unable to train, doesn't have enough pair count")
