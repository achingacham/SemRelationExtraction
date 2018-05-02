from model import SkipGramModel
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys
import re


class Word2Vec:
    
    def __init__(self, ifolder, output_file, emb_dimension=500,
                 batch_size=8,
                 iteration=2,
                 initial_lr=0.025):
        
        self.ifolder = ifolder
        self.output_file_name = ifolder+output_file
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
        
        self.emb_size     = len(self.id2word)
        self.pair_emb_size = len(self.id2pair)
        
        
        self.skip_gram_model = SkipGramModel(self.pair_emb_size,self.emb_size, self.emb_dimension)
        self.use_cuda = torch.cuda.is_available()
        
        if self.use_cuda:
            self.skip_gram_model.cuda()
        self.optimizer = optim.SGD(self.skip_gram_model.parameters(), lr=self.initial_lr)

    def read_word_dict(self, wdictfile ):
        
        with open(wdictfile) as inputFile:
            
            for item in inputFile:
                word = item.split('[}')[0]
                wid = item.split('[}')[1]
                self.id2word[int(wid)] = word
                
    def read_pair_dict(self, pdictfile ):
        
        with open(pdictfile) as inputFile:
            
            for item in inputFile:
                pair = item.split('[}')[0:2]
                pid = item.split('[}')[2]
                self.id2pair[int(pid)] = pair
                self.pair2id[(pair[0].strip("' "),pair[1].strip("' "))] = int(pid)
        
        self.cross_verification()
        
    def evaluate_pair_count(self):
        
        self.datasets = dict()
        
        dsfile = self.ifolder+"Statistics"
        with open(dsfile) as inputFile:
             
            for item in inputFile:
                
                if re.match("Dataset",item):
                    i = item.split(':')[1]
        print(i)        
        return int(i)
    
    def get_batch_pairs(self):
        
        pos_pairs = []
        count = 0
        dsfile = self.ifolder+"Triplesets_positive"
        
        while count < self.batch_size:
            with open(dsfile) as inputFile:
                inputFile.seek(self.fpos)
                line = inputFile.readline().strip('\n')
                if not line:
                    self.fpos = 0
                    inputFile.seek(self.fpos)
                    line = inputFile.readline().strip('\n')
                
                #print("...",line,"....")

                pid = line.split(',')[0].strip('( )')
                wid = line.split(',')[1].strip('( )')
                
                self.fpos = inputFile.tell()
                pos_pairs.append([int(pid),int(wid)])
               
            count +=1
        
        return pos_pairs
    
    def get_neg_v(self):
        
        neg_v = []
        count = 0
        dsfile = self.ifolder+"Triplesets_negative"
        
        while count < self.batch_size:
            with open(dsfile) as inputFile:
                inputFile.seek(self.fneg)
                line = inputFile.readline().strip('\n')
                
                if not line:
                    self.fneg = 0
                    inputFile.seek(self.fneg)
                    line = inputFile.readline().strip('\n')

                self.fneg = inputFile.tell()
                neg_v.append([int(i.strip('[ ]')) for i in line.split(',')])
               
            count +=1
        
        return neg_v
    
    def cross_verification(self):

        #Remove the file if it already exists
        try:
            os.remove("BlessSet.txt")
        except:
            pass
        
        
        blessFile = open("BlessSet.txt","w")
    
        with open("/home/achingacham/Model/GRID_data/Evaluation_Datasets/BLESS/UniqueTuples") as evalFile:
            testDataset = evalFile.readlines()
            
            for items in testDataset:
                nouns = items.split()
                search_key = (nouns[0],nouns[1])
                rev_search_key = (nouns[1],nouns[0])
                
                if (search_key in self.pair2id) or  (rev_search_key in self.pair2id):
                    blessFile.write("\n"+str(search_key) +"[}"+ str(rev_search_key))
                    
                    
        evalFile.close()
        blessFile.close()

    def train(self):
        """Multiple training.
        Returns:
            None.
        """
        pair_count = self.evaluate_pair_count() 
        #pair_count = 5214432
        batch_count = self.iteration * pair_count / self.batch_size
        process_bar = tqdm(range(int(batch_count)))
        
        # self.skip_gram_model.save_embedding(
        #     self.data.id2word, 'begin_embedding.txt', self.use_cuda)
        
            
        for i in process_bar:
        
            pos_pairs = self.get_batch_pairs()
            neg_v = self.get_neg_v()  
            
          
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
            if i * self.batch_size % 100000 == 0:
                lr = self.initial_lr * (1.0 - 1.0 * i / batch_count)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

        self.skip_gram_model.save_embedding(self.id2pair, self.output_file_name, self.use_cuda)    

if __name__ == '__main__':
    w2v = Word2Vec(ifolder=sys.argv[1], output_file=sys.argv[2])
    
    if w2v.pair_emb_size > 0 :
        w2v.train()
        pass
    else:
        print("Unable to train, doesn't have enough pair count")
