import numpy
from collections import deque
import time
import progressbar

numpy.random.seed(12345)

class InputData:
    """Store data for word2vec, such as word map, sampling table and so on.

    Attributes:
        word_frequency: Count of each word, used for filtering low-frequency words and sampling table
        word2id: Map from word to word id, without low-frequency words.
        id2word: Map from word id to word, without low-frequency words.
        sentence_count: Sentence count in files.
        word_count: Word count in files, without low-frequency words.
    """

    def __init__(self, file_name, min_count, pair_min_count, window_size):
        
        self.input_file_name = file_name
        self.get_words(min_count, window_size)
        #self.get_noun_pairs(pair_min_count, window_size)  # Considering only Nouns        
        #self.word_pair_batch = deque()
        #self.init_sample_table()
        
        print('\nWord Count: %d' % self.word_count)
        #print('\nPair Count: %d' % self.pair_count)
        #print('\nSentence Length: %d' % (self.sentence_length))

    def get_words(self, min_count, window_size):

        word_frequency = dict()
        self.word2id = dict()
        self.id2word = dict()
        self.word_frequency = dict()
        
        # for triplets
        pair_frequency = dict()
        self.pair2id = dict()
        self.id2pair = dict()
        self.pair_frequency = dict()
        
        self.input_file = open(self.input_file_name)
        
        self.sentence_length = 0
        self.sentence_count = 0
        self.word_count = 0
        self.pair_count = 0
        
        
        print("Reading from file.. ")
        bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
      
        for line in self.input_file:
            
            
            if self.sentence_count % 1000 == 0:
                bar.update(self.sentence_count)
           
            self.sentence_count += 1
            line = line.strip().split()
            
            self.sentence_length += len(line)
               
            for i,w in enumerate(line):
            
                if i % 2 == 0:                          #ignore all POS tags
           
                    try:
                        word_frequency[w] += 1
                    except:
                        word_frequency[w] = 1
            
            Noun_positions = numpy.array([], dtype = int)
            
            for pos in ['NNS','NN','NP']:
                if pos in line:
                    line_numpy_arr = numpy.array(line)
                    Noun_positions = numpy.append(Noun_positions,(numpy.where(line_numpy_arr == pos)[0]-1))
                    
                    
            Noun_positions.sort()
            print(Noun_positions)
            
            
            for index_1, n1_position in enumerate(Noun_positions):
                for index_2, n2_position in enumerate(Noun_positions[index_1+1:]):
                    if abs(n1_position - n2_position) > 2*window_size:
                        break
                    else:
                        print("\n WW",n1_position,n2_position,"::",line[n1_position],line[n2_position])
                        n1 = line[n1_position]
                        n2 = line[n2_position]
                        
                        
                        try:
                            
                            pair_frequency[n1][n2] += 1
                        except:
                            
                            pair_frequency[n1] = {n2:1}
                        
     
        wid = 0
        
        for w, c in word_frequency.items():
            if c < min_count:
                self.sentence_length -= c
                
                
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1
            
        self.word_count = len(self.word2id)
        
               
        
    def get_noun_pairs(self,pair_min_count, window_size):
        
        # for triplets
        self.pair_frequency = dict()
        pair_frequency = dict()
        self.pair2id = dict()
        self.id2pair = dict()
        
    
        self.input_file.seek(0)
        
        self.pair_count = 0
        sentence_count = 0
        
        print("\nMaking noun pair samples..")
        bar_pairs = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
        
        
        for line in self.input_file:
            
            #time.sleep(0.001)
            if sentence_count % 1000 == 0:
                bar_pairs.update(sentence_count)
            
            sentence_count += 1
            
            temp_arr = line.split()
            
            temp_pos2NN = dict()
        
            #print(temp_arr)
            
            # Also can club with previous function
            
            for i,w in enumerate(temp_arr):
                if w in ['NNS','NN','NP']:
                    temp_pos2NN[i-1] = temp_arr[i-1]
                    #print("\n >>>>>",i, w)
                 
                
                
            # Can improve the loop by other way
            #for pos in ['NNS','NN','NP']:
            #    if pos in temp_arr:
            #        temp_numpy_arr = numpy.array(temp_arr)
            #        print("\n #########",numpy.where(temp_numpy_arr == pos),pos)
            #
            #    print(pos)
                
            #print(temp_pos2NN)
            
            temp_positions = [i for i in temp_pos2NN.keys()]
        
            temp_positions.sort()
        
            for i, items in enumerate(temp_pos2NN.items()):
                
                for j, pos in enumerate(temp_positions[i:]):
                    
                    if items[0] == pos:
                        continue
                        
                    if abs(items[0]-pos) < 2 * window_size:
                        #print(i,j,'::',temp_pos2NN[items[0]],temp_pos2NN[pos])
                        
                        w1 = temp_pos2NN[items[0]]
                        w2 = temp_pos2NN[pos]
                        
                        if w1 in self.word2id and  w2 in self.word2id:
                       
                            key = [self.word2id[w1], self.word2id[w2]]
                            #key.sort()
                            pair_key = str(key[0])+':'+str(key[1])

                            try:
                                pair_frequency[pair_key] += 1
                            except:
                                pair_frequency[pair_key] = 1
                            
         
        #print(pair_frequency)

        for key,value in pair_frequency.items():
            if value > pair_min_count:
                self.pair_frequency[key] = value                
        
        self.pair_count = len(self.pair_frequency)

        
        for index,pair in enumerate(self.pair_frequency):
            self.pair2id[pair] = index
            self.id2pair[index] = [self.id2word[int(w)] for w in pair.split(":")]
                        
                    
    
        
    def init_sample_table(self):
        self.sample_table = []
        sample_table_size = 1e8
        
        pow_frequency = numpy.array(list(self.word_frequency.values()))**0.75  # 3/4 of the power of pairs
        words_pow = sum(pow_frequency) # initial calculation
        ratio = pow_frequency / words_pow # initial calculation  
        count = numpy.round(ratio * sample_table_size) # This is for sampling indices on the same ratio over 1e8  # initial implementation

        print("\nMaking negative samples..")
        bar_samples = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
        
        for wid, c in enumerate(count):
            self.sample_table += [wid] * int(c)
            
            #time.sleep(0.001)
            if wid % 1000 == 0:
                bar_samples.update(wid)
                
        self.sample_table = numpy.array(self.sample_table)
        

    # @profile
    def get_batch_pairs(self, batch_size, window_size):
        
        while len(self.word_pair_batch) < batch_size:
            
            sentence = self.input_file.readline()
            
            if sentence is None or sentence == '':
                self.input_file = open(self.input_file_name)
                sentence = self.input_file.readline()
            
            word_ids = []
            
            for word in sentence.strip().split():
                
                try:
                    word_ids.append(self.word2id[word])
                except:
                    continue
            
            #print("word_IDS \n",word_ids)
        
            
            for i, l_u in enumerate(word_ids):
                
                temp_k = word_ids[i+1:i+window_size]
                
                for k, r_u in enumerate(temp_k[1:]): 
                    temp_j = temp_k[0:k+1]
                    
                    for j, v in enumerate(temp_j):
                    
                        assert l_u < self.word_count
                        assert v < self.word_count
                        assert r_u < self.word_count
                        
                        #print(i,j,k,'::',l_u,v,r_u)
                        
                        search_key = str(l_u)+':'+str(r_u)
                        if search_key in self.pair_frequency.keys() and v in self.word2id.values():
                            
                            self.word_pair_batch.append((self.pair2id[search_key],v))
                            
                            #print('::',self.word2id(l_u),self.word2id(v),self.word2id(r_u))
                            #print("\n ",sentence)
                            #print('\n::',l_u,v,r_u)
                        
                    
        batch_pairs = []
        
        for _ in range(batch_size):
            batch_pairs.append(self.word_pair_batch.popleft())
        
        return batch_pairs

    # @profile
    def get_neg_v_neg_sampling(self, pos_word_pair, count):
        neg_v = numpy.random.choice(
            self.sample_table, size=(len(pos_word_pair), count)).tolist()
        return neg_v

    def evaluate_pair_count(self, window_size):
        return self.sentence_length * (2 * window_size - 1) - (
            self.sentence_count - 1) * (1 + window_size) * window_size


def test():
    a = InputData('./zhihu.txt')


if __name__ == '__main__':
    test()
