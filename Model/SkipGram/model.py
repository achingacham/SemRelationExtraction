import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class SkipGramModel(nn.Module):
    """Skip gram model of word2vec.

    Attributes:
        emb_size: Embedding size.
        emb_dimention: Embedding dimention, typically from 50 to 500.
        u_embedding: Embedding for center word.
        v_embedding: Embedding for neibor words.
    """

    def __init__(self, pair_emb_size, emb_size, emb_dimension):
        """Initialize model parameters.

        Apply for two embedding layers.
        Initialize layer weight

        Args:
            emb_size: Embedding size.
            emb_dimention: Embedding dimention, typically from 50 to 500.

        Returns:
            None
        """
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.pair_emb_size = pair_emb_size
        self.emb_dimension = pair_emb_size + 100
        self.u_embeddings = nn.Embedding(self.pair_emb_size, self.emb_dimension, sparse=True)
        
        self.v_embeddings = nn.Embedding(self.emb_size, self.emb_dimension, sparse=True)
        self.init_emb()
        
    def init_emb(self):
        """Initialize embedding weight like word2vec.

        The u_embedding is a uniform distribution in [-0.5/emb_size, 0.5/emb_size], and the elements of v_embedding are zeroes.

        Returns:
            None
        """
        initrange = 0.5 / self.emb_dimension
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, pos_u, pos_v, neg_v):
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
        
       
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)                 #context vector is for a pair of words.
        neg_emb_v = self.v_embeddings(neg_v)
        
        #print("Positive U embed :",emb_u.data.shape,"\n")
        #print("Positive V embed:",emb_v.data.shape,"\n")
        #print("Negative V embed:",neg_emb_v.data.shape,"\n")
        
        score = torch.mul(emb_u, emb_v).squeeze()
        score = torch.sum(score, dim=1) # summed up for the dot product sum
        score = F.logsigmoid(score)
        
        neg_score = torch.bmm(neg_emb_v, emb_u.unsqueeze(2)).squeeze() 
        neg_score = F.logsigmoid(-1 * neg_score)
        return -1 * (torch.sum(score)+torch.sum(neg_score)) #this has k values being summed up

    def save_embedding(self, id2pair, file_name, use_cuda):
        """Save all embeddings to file.

        As this class only record pair id, so the map from id to pair has to be transfered from outside.

        Args:
            id2pair: map from pair id to pair.
            file_name: file name.
        Returns:
            None.
        """
        if use_cuda:
            embedding = self.u_embeddings.weight.cpu().data.numpy()
        else:
            embedding = self.u_embeddings.weight.data.numpy()
        fout = open(file_name, 'w')
        fout.write('%d %d\n' % (len(id2pair), self.emb_dimension))
        for wid, w in id2pair.items():
            e = embedding[wid]
            e = ' '.join(map(lambda x: str(x), e))
            fout.write('%s %s\n' % (w, e))


def test():
    model = SkipGramModel(100, 100)
    id2pair = dict()
    for i in range(100):
        id2pair[i] = str(i)
    model.save_embedding(id2pair)


if __name__ == '__main__':
    test()
