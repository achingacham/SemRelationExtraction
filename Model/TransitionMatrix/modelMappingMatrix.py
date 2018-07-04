import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import ipdb

class SkipGramModel(nn.Module):
    """Skip gram model of word2vec.

    Attributes:
        pair_emb_size : Number of pairs
        emb_size: Number of context words
        emb_dimention: Embedding dimention, typically from 50 to 500.
        
    """

    def __init__(self, pair_emb_size, emb_size, emb_dimension, id2word):
        """Initialize model parameters.

        Apply for two embedding layers.
        Initialize layer weight

        Args:
            emb_size: Embedding size.
            emb_dimention: Embedding dimention, typically from 50 to 500.

        Returns:
            None
        """
        self.id2word = id2word
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.pair_emb_size = pair_emb_size
        self.emb_dimension = emb_dimension
        
        ###
        self.word_embeddings = nn.Embedding(self.emb_size,self.emb_dimension) # |V|x400  
        self.mapping_matrix = nn.Linear(self.emb_dimension, self.emb_dimension) # 400x400
        ###
        
        self.init_emb()
        #self.create_dictWordVectors(pTrainedfile, self.emb_dimension)
        
        print("Model parameters")
        print("self.emb_dimension :\t",self.emb_dimension)
        print("self.emb_size :\t",self.emb_size)
        
        
        
    def init_emb(self):
        """Initialize embedding weight like word2vec.

        The u_embedding is a uniform distribution in [-0.5/emb_size, 0.5/emb_size], and the elements of v_embedding are zeroes.

        Returns:
            None
        """
        initrange = 0.5 / self.emb_dimension
        self.word_embeddings.weight.data.uniform_(-initrange, initrange)
        self.mapping_matrix.weight.data.uniform_(-0, 0)

    def forward(self, pos_u1,pos_u2, pos_v, neg_v):
        """Forward process.

        As pytorch designed, all variables must be batch format, so all input of this method is a list of word id.

        Args:
            pos_u: list of center word ids for positive word pairs.
            pos_v: list of neighbor word ids for positive word pairs.
            neg_u: list of center word ids for negative word pairs.
            neg_v: list of neighbor word ids for negative word pairs.

        Returns:
            Loss of this process, a pytorch variable.
        """
        
        
        
        word_1 = self.word_embeddings(pos_u1)
        word_2 = self.word_embeddings(pos_u2)
        word_context = self.word_embeddings(pos_v) #PreTrained vector for every word in Vocabulary doesn't exist, so try this.
        neg_context = self.word_embeddings(neg_v)
        
        
        #Method2
        
        relation_vector = word_1 + word_2          #why not use this directly?
        pred_relation = self.mapping_matrix(relation_vector)      #current prediction
        
        score = torch.mul(pred_relation, word_context)
        score = torch.sum(score, dim=1)
        score = F.logsigmoid(score)
        
        neg_score = torch.bmm(neg_context, pred_relation.unsqueeze(2)).squeeze() 
        neg_score = F.logsigmoid(-1 * neg_score)
        
        return -1 * (torch.sum(score)+torch.sum(neg_score)) #this has k values being summed up

        
    def create_dictWordVectors(self, preTrainedVectors, dim):
        
        self.dictWordVectors = dict()
        
        with open(preTrainedVectors) as inputFile:
            
            for Vectors in inputFile:
               
                vec = Vectors.split()
                
                try:
                    if len(vec) == dim+1:
                        self.dictWordVectors[vec[0]] = vec[1:dim+1]
                except:
                    print(vec[0],len(vec))
                    
        inputFile.close()
        
    
    def save_embedding(self, id2word, file_name, use_cuda):
        """Save all embeddings to file.

        As this class only record pair id, so the map from id to pair has to be transfered from outside.

        Args:
            id2pair: map from pair id to pair.
            file_name: file name.
        Returns:
            None.
        """
        if use_cuda:
            embedding = self.word_embeddings.weight.cpu().data.numpy()
        else:
            embedding = self.word_embeddings.weight.data.numpy()
            
        fout = open(file_name, 'w')
        fout.write('%d %d\n' % (len(id2word), self.emb_dimension))
        
        for wid, w in id2word.items():
            e = embedding[wid]
            e = ' '.join(map(lambda x: str(x), e))
            fout.write('%s %s\n' % (w, e))
            
        
        if use_cuda:
            matrixWeights = self.mapping_matrix.weight.cpu().data.numpy()
            matrixBias = self.mapping_matrix.bias.cpu().data.numpy()
        else:
            matrixWeights = self.mapping_matrix.weight.data.numpy()
            matrixBias = self.mapping_matrix.bias.data.numpy()
            
        fout = open(file_name+"_MM.txt", 'w')
        fout.write('%d %d\n' % (len(id2word), self.emb_dimension))
        
       
        for index,value in enumerate(matrixWeights):
            for weight in value:
                fout.write(str(weight)+"\t")
            fout.write(str(matrixBias[index])+"\n")


def test():
    model = SkipGramModel(100, 100)
    id2pair = dict()
    for i in range(100):
        id2pair[i] = str(i)
    model.save_embedding(id2pair)


if __name__ == '__main__':
    test()
