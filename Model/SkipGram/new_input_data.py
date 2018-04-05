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
        self.word_pair_batch = deque()
        self.get_words_pairs(min_count, window_size, pair_min_count)
        self.init_sample_table()
        
        print('\nWord Count: %d' % self.word_count)
        print('\nPair Count: %d' % self.pair_count)
        print('\nSentence Length: %d' % (self.sentence_length))

    def get_words_pairs(self, min_count, window_size, pair_min_count):

        # must include tokenization, lower cased words
        
        word_frequency = dict()
        self.word2id = dict()
        self.id2word = dict()
        self.word_frequency = dict()
        
        # for triplets
        pair_frequency = dict()
        paired_nouns = dict()
        self.pair2id = dict()
        self.id2pair = dict()
        self.pair_frequency = dict()
        
        self.between_words = dict()
        
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
                        word_frequency[w.lower()] += 1
                    except:
                        word_frequency[w.lower()] = 1
            
            Noun_positions = numpy.array([], dtype = int)
            
            for pos in ['NNS','NN','NP']:
                if pos in line:
                    line_numpy_arr = numpy.array(line)
                    Noun_positions = numpy.append(Noun_positions,(numpy.where(line_numpy_arr == pos)[0]-1))
                    
            Noun_positions.sort()
            
            
            for index_1, n1_position in enumerate(Noun_positions):
                for index_2, n2_position in enumerate(Noun_positions[index_1+1:]):
                    if (abs(n1_position - n2_position) > (2*window_size)) or (abs(n1_position - n2_position) < 3):
                        break
                    
                    else:
                        
                        
                        n1 = line[n1_position].lower()
                        n2 = line[n2_position].lower()

                        if n1 != n2:
                            try:
                            
                                pair_frequency[n1,n2] += 1
                                paired_nouns[n1][0].update([n2])
                                paired_nouns[n2][0].update([n1])
                                self.between_words[n1,n2].append(line[n1_position+2:n2_position:2])
                            
                            
                            except:
                            
                                pair_frequency[n1,n2] = 1
                                paired_nouns[n1] = [set([n2]),1]
                                paired_nouns[n2] = [set([n1]),2]
                                self.between_words[n1,n2] = [line[n1_position+2:n2_position:2]]
                            
                            
        self.input_file.close()
    
        wid = 0
        
        for w, c in word_frequency.items():
            if c < min_count:
                self.sentence_length -= c
                
                if w in paired_nouns:
                    
                    if paired_nouns[w][1] == 1:
                        for wTemp in paired_nouns[w][0]:
                            pair_frequency[w,wTemp] = -1
                            self.between_words[w,wTemp] = -1
                        
                        
                    if paired_nouns[w][1] == 2:
                        for wTemp in paired_nouns[w][0]:
                            pair_frequency[wTemp,w] = -1
                            self.between_words[wTemp,w] = -1
                        
                continue
             
            
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1
            
        
        self.word_count = len(self.word2id)
        
        
        pid = 0
        
        for key,value in pair_frequency.items():
            
            if value >= pair_min_count:
                self.pair2id[key] = pid
                self.id2pair[pid] = [":".join(key)]
                self.pair_frequency[key] = value
                
                
                for inbetween_words in self.between_words[key]:
                    for words in inbetween_words:
                        words = words.lower()
                        l = len(self.word_pair_batch)
                        try:
                            self.word_pair_batch.append((pid,self.word2id[words]))
                            
                        except:
                            
                            pass
                pid += 1
                
                
        self.pair_count = len(self.pair_frequency)
        
        
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
        
        batch_pairs = []
        
        if self.word_pair_batch:
            
            for _ in range(batch_size):
                batch_pairs.append(self.word_pair_batch.popleft())
        
        return batch_pairs
    
    
    def get_neg_v_neg_sampling(self, pos_word_pair, count):
        neg_v = numpy.random.choice(
            self.sample_table, size=(len(pos_word_pair), count)).tolist()
        return neg_v

    def evaluate_pair_count(self):
            
        return len(self.word_pair_batch)


def test():
    a = InputData('./zhihu.txt')


if __name__ == '__main__':
    test()
