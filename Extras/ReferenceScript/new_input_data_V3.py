import numpy
from collections import deque
import time
import progressbar
import os
import re

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

    def __init__(self, input_file_name, min_count, pair_min_count, window_size,output_file_name):
        
        self.input_file_name = input_file_name
        self.output_file_name = output_file_name
        self.word_pair_batch = deque()
        self.window_size = window_size
        self.min_count = min_count
        self.pair_min_count = pair_min_count
        
        self.get_words_pairs()
        #self.init_sample_table()
        
        print('\nWord Count: %d' % self.word_count)
        print('\nPair Count: %d' % self.pair_count)
        print('\nSentence Length: %d' % (self.sentence_length))

    
    def get_words_pairs(self):

        # must include tokenization, lower cased words
        
        #for words
        self.initial_word_frequency = dict()
        self.word2id = dict()
        self.id2word = dict()
        self.word_frequency = dict()
        
        #for pairs
        self.initial_pair_frequency = dict()
        self.pair2id = dict()
        self.id2pair = dict()
        
        #context for pairs
        self.between_words = dict()
        self.word_pair_batch_count = dict()
        
        self.input_file = open(self.input_file_name)
        
        #counters
        self.sentence_length = 0
        self.sentence_count = 0
        self.word_count = 0
        self.pair_count = 0
        
        
        print("Reading from file.. ")
        bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
      
        for line in self.input_file:
            
            if self.sentence_count % 100000 == 0:
                bar.update(self.sentence_count)
           
            self.sentence_count += 1
            line = line.strip().split()
            
            #every sentence include POS for each token
            self.sentence_length += (len(line)/2) 
               
            for i,w in enumerate(line):
            
                #ignore all POS tags, skip alternate words
                if i % 2 == 0:                          
                    
                    try:
                        self.initial_word_frequency[w.lower()] += 1
                    except:
                        self.initial_word_frequency[w.lower()] = 1
                    
            Noun_positions = numpy.array([], dtype = int)
            
            #track the position of Nouns in the sentences
            for pos in ['NNS','NN','NP']:
                if pos in line[1::2]:
                    line_numpy_arr = numpy.array(line[1::2])
                    Noun_positions = numpy.append(Noun_positions,(numpy.where(line_numpy_arr == pos)[0]*2))
                    
            Noun_positions.sort()
            
            self.track_pairs(line, Noun_positions)
        
        
        # First parse of input file is done
        self.input_file.close()
        print("\nFirst parsing Done!")
        
        self.make_word_dictionaries()
        print("\n Made word dictionaries")
        self.remove_infrequent_pairs()
        print("\n Removed infrequent pairs")
        self.make_pair_dictionaries()
        print("\n Made pair dictionarieste")
    
    def track_pairs(self, line, Noun_positions):

        #find the possible Nouns pairs in the sentence, with in the window. Ignore Noun pairs apprearing adjascent
        for index_1, n1_position in enumerate(Noun_positions):
            for index_2, n2_position in enumerate(Noun_positions[index_1+1:]):

                if (abs(n1_position - n2_position) > (2*self.window_size)) or (abs(n1_position - n2_position) < 3):
                    break

                #track the possible inbetween words for any pair of Nouns
                else:

                    n1 = line[n1_position].lower()
                    n2 = line[n2_position].lower()

                    #avoid all pairs of same noun eg: ('school','school')
                    if n1 == n2:
                        continue

                    if (n1,n2) in self.initial_pair_frequency:

                        self.initial_pair_frequency[n1,n2] += 1

                        for items in line[n1_position+2:n2_position:2]:
                            if items in self.between_words[n1,n2]:
                                self.between_words[n1,n2][items] += 1
                            else:
                                self.between_words[n1,n2].update({items:1})

                    else:
                        #track pair frequency for each Noun pair
                        #store all words inbetween the Noun pairs with their individual count.

                        self.initial_pair_frequency[n1,n2] = 1
                        self.between_words[n1,n2] = {}

                        for items in line[n1_position+2:n2_position:2]:
                            self.between_words[n1,n2].update({items:1})
    
    def make_word_dictionaries(self):
        
        wid = 0
        
        for word,wCount in self.initial_word_frequency.items():
            
            if wCount < self.min_count:
                self.sentence_length -= wCount
                continue
                
            if not (word in self.word2id):
                self.word2id[word] = wid
                self.id2word[wid] = word
                self.word_frequency[wid] = wCount
                wid += 1
        
        self.word_count = len(self.word2id)
        
        self.init_sample_table()
        
        #Check for any error is parsing
        [print("\n",item,value) for (item,value) in self.initial_word_frequency.items() if item in ['NNS','NN','NP','POS','CD','ADV','ADJ']]
        
        #Release memory associated with this dictionary
        self.initial_word_frequency = dict()
    
    def remove_infrequent_pairs(self):
        
        for pair, count in self.initial_pair_frequency.items():
            
            word_1 = pair[0]
            word_2 = pair[1]
            
            try:
                wid_1 = self.word2id[word_1]
            except:
                #print(word_1,":",self.initial_word_frequency[word_1])
                self.initial_pair_frequency[pair] = -1
                self.between_words[pair] = -1
                
            try:
                wid_2 = self.word2id[word_2]
            except:
                #print(word_2,":",self.initial_word_frequency[word_2])
                self.initial_pair_frequency[pair] = -1
                self.between_words[pair] = -1
                
    
    
    
        
    def make_pair_dictionaries(self):
        
        # Use pair_count file for testing only; display each Noun pairs with frequency, and the inbetween words
        try:
            os.remove(self.output_file_name)
        except:
            print("\n No previous Samples file existed")
        
        
        pid = 0
        for item in sorted(self.initial_pair_frequency, key = self.initial_pair_frequency.get, reverse=True):
            
            key = item
            value = self.initial_pair_frequency[item]
           
            if value >= self.pair_min_count:
                
                #testFile.write("\n"+str(pid)+"\t"+":".join(key)+"\t"+str(value)+"\t"+str(self.between_words[key]))
                
                self.pair2id[key] = pid
                self.id2pair[pid] = [":".join(key)]
                
                
                # generate a dictionary of pairs and inbetween words with their individual count
                for words,wCount in self.between_words[key].items():
                
                    if words == -1:
                        print("Unexpected Entry, Check pair sampling!")
        
                    # To avoid samples with inbetween word same as one of those in pair
                    if words in key:
                        continue

                    words = words.lower()
                    
                    try:
                        wid = self.word2id[words]

                        #self.word_pair_batch.append((pid,wid))  #Old method to push items to Deque

                        if pid in self.word_pair_batch_count:
                        
                            if wid in self.word_pair_batch_count[pid]:
                                self.word_pair_batch_count[pid][wid] += wCount
                            else:
                                self.word_pair_batch_count[pid].update({wid:wCount}) 
                                
                        else:

                            self.word_pair_batch_count[pid] = {wid:wCount}
                            
                    except:
                        #print("\n Inbetween word count is too less",words,self.initial_word_frequency[words])
                        pass    
                
                
                self.push_into_deque(pid)
                pid += 1
                
        
        print("\n Pushed into Deque")
        
        #Release memory with obselete dictionaries
        self.between_words = dict()
        self.initial_pair_frequency = dict()
        
        #testFile.close()

        self.cross_verification()
        print("\n Completed BLESS set verification")
        
        self.pair_count = len(self.pair2id)
        
    # New method to push items to Deque, ie; only those inbetween words which are top most frequent for any pair
    def push_into_deque(self,pid):
            
        if pid in self.word_pair_batch_count:

            pid_dict = self.word_pair_batch_count[pid]
            sorted_wid = sorted(pid_dict, key=pid_dict.get, reverse=True)
            top_50 = int((len(sorted_wid))/2)

            #mention the limit for consideration
            top_wid = sorted_wid[:] 

            for wid in top_wid:

                count = self.word_pair_batch_count[pid][wid]

                for _ in range(count):
                    self.word_pair_batch.append((pid,wid))
                    
                    with open(self.output_file_name,"a") as outputFile:
                        sample = '('+str(pid)+','+str(wid)+')'
                        outputFile.write(sample+'\n')
                    
                    self.get_neg_v_neg_sampling((pid,wid),6)
                    
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
    
    def get_neg_v_neg_sampling(self, pos_word_pair, count):
        neg_v = numpy.random.choice(
            self.sample_table, size=(len(pos_word_pair), count)).tolist()
        
    
    def cross_verification(self):

        blessFile = open("BlessSet.txt","w")
    
        with open("/home/achingacham/Model/GRID_data/Evaluation_Datasets/BLESS/UniqueTuples") as evalFile:
            testDataset = evalFile.readlines()
            
            for items in testDataset:
                nouns = items.split()
                search_key = (nouns[0],nouns[1])
                rev_search_key = (nouns[1],nouns[0])
                if (search_key in self.pair2id) or  (rev_search_key in self.pair2id):
                    blessFile.write("\n"+str(nouns))
                                
                    
                
        evalFile.close()
        blessFile.close()
   
   
def test():
    a = InputData('./zhihu.txt')


if __name__ == '__main__':
    test()
