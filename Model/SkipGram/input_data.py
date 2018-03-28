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
        self.get_words(min_count)
        #self.get_pairs(pair_min_count, window_size)
        self.get_noun_pairs(pair_min_count, window_size)  # Considering only Nouns
        ##for triplets contains tuples (indexto_pair of words, middle word)
        
        self.word_pair_batch = deque()
        self.init_sample_table()
        
        print('\nWord Count: %d' % self.word_count)
        print('\nPair Count: %d' % self.pair_count)
        print('\nSentence Length: %d' % (self.sentence_length))

    def get_words(self, min_count):
        self.input_file = open(self.input_file_name)
        
        self.sentence_length = 0
        self.sentence_count = 0
        word_frequency = dict()
        #for triplets
        
        pair_frequency = dict()
        print("Reading from file.. ")
       
        bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
      
        for line in self.input_file:
            
            #time.sleep(0.001)
            
            if sentence_count % 1000 == 0:
                bar.update(self.sentence_count)
           
            self.sentence_count += 1
            line = line.strip().split()
            self.sentence_length += len(line)
               
            for i,w in enumerate(line):
                
                if i % 2 == 0:
           
                    try:
                        word_frequency[w] += 1
                    except:
                        word_frequency[w] = 1

            
                
        self.word2id = dict()
        self.id2word = dict()
        
        wid = 0
        self.word_frequency = dict()
        for w, c in word_frequency.items():
            if c < min_count:
                self.sentence_length -= c
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1
        self.word_count = len(self.word2id)
              
        #print(self.word2id)
        
    def get_noun_pairs(self,pair_min_count, window_size):
        
        # for triplets
        self.pair_frequency = dict()
        pair_frequency = dict()
        self.pair2id = dict()
        self.id2pair = dict()
        
    
        self.input_file.seek(0)
        
        bar_pairs = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
        
        print("\nMaking noun pair samples..")
        
        sentence_count = 0
        
        for line in self.input_file:
            
            #time.sleep(0.001)
            if sentence_count % 1000 == 0:
                bar_pairs.update(sentence_count)
            
            sentence_count += 1
            
            temp_arr = line.split()
            
            temp_pos2NN = dict()
        
            #print(temp_arr)
            
            for i,w in enumerate(temp_arr):
                if w in ['NNS','NN','NP']:
                    temp_pos2NN[i-1] = temp_arr[i-1]
                    #print(i,w)
        
            
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
                        
                    
    
    def get_pairs(self, pair_min_count, window_size):
        
        # for triplets
        self.pair_frequency = dict()
        pair_frequency = dict()
        self.pair2id = dict()
        self.id2pair = dict()
        
       
        self.input_file.seek(0)
        
        bar_pairs = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
        
        print("\nMaking pair samples..")
        
        sentence_count = 0
        
        for line in self.input_file:
            
            #time.sleep(0.001)
            if sentence_count % 1000 == 0:   
               bar_pairs.update(sentence_count)

            sentence_count += 1
            
            temp_arr = line.split('\t')
            
            for i,w1 in enumerate(temp_arr):
                
                for j,w2 in enumerate(temp_arr[max(i-window_size,0):i+window_size]):
                    
                    if w1 == w2:
                        continue
                    
                    if w1 in self.word2id and  w2 in self.word2id:
                       
                        key = [self.word2id[w1], self.word2id[w2]]
                        #key.sort()
                        pair_key = str(key[0])+':'+str(key[1])
                        
                        try:
                            pair_frequency[pair_key] += 1
                        except:
                            pair_frequency[pair_key] = 1
                        
        
        for key,value in pair_frequency.items():
            if value > pair_min_count:
                self.pair_frequency[key] = value
                
        self.pair_count = len(self.pair_frequency)
        
        for index,pair in enumerate(self.pair_frequency):
            self.pair2id[pair] = index
            self.id2pair[index] = [self.id2word[int(w)] for w in pair.split(":")]
        
        
        #print(self.pair2id)
        
    def init_sample_table(self):
        self.sample_table = []
        sample_table_size = 1e8
        
        #print("Id2Word\n")
        #print(self.id2word)
        #print("word_frequency\n")
        #print(self.pair_frequency)
        
        pow_frequency = numpy.array(list(self.word_frequency.values()))**0.75  # 3/4 of the power of pairs
        
        #print("Pow_freq\n")
        #print(pow_frequency)
        
        words_pow = sum(pow_frequency) # initial calculation
        
        #print("Sum of word pow\n")
        #print(words_pow)
        
        ratio = pow_frequency / words_pow # initial calculation  
        
        #print("Ratio\n")
        #print(ratio)
        
        
        count = numpy.round(ratio * sample_table_size) # This is for sampling indices on the same ratio over 1e8  # initial implementation

        #print("Count\n")
        #print(count)

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
            print("\n ", sentence)
            
            for i, l_u in enumerate(word_ids):
                temp_k = word_ids[i+1:i+window_size]
                
                for k, r_u in enumerate(temp_k[1:]): #
                    temp_j = temp_k[0:k+1]
                    
                    for j, v in enumerate(temp_j):
                    
                        assert l_u < self.word_count
                        assert v < self.word_count
                        assert r_u < self.word_count
                        
                        #print(i,j,k,'::',l_u,v,r_u)
                        
                        search_key = str(l_u)+':'+str(r_u)
                        if search_key in self.pair_frequency.keys() and v in self.word2id.values():
                            
                            self.word_pair_batch.append((self.pair2id[search_key],v))
                            
                            print('::',self.word2id(l_u),self.word2id(v),self.word2id(r_u))
                        
                        
                    
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
