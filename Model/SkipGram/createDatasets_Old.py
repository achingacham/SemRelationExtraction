import os
import sys
import re
import numpy
#import ipdb
#import cProfile
import time

class Datasets:
    
    def __init__(self, inputfolder, outfolder, minWordCount, minPairCount):
        
        listfiles = os.listdir(inputfolder)
        outputfolder = outfolder+str(int(time.time()))
        os.mkdir(outputfolder)
        
        self.posfile = outputfolder+'/Triplesets_positive'
        self.negfile = outputfolder+'/Triplesets_negative'
        self.statfile = outputfolder+"/Statistics"

        self.wdictfile = outputfolder+'/Word2Id'
        self.pdictfile = outputfolder+'/Pair2Id'
        
        wordfiles = []
        pairfiles = []
        
        self.word_frequency = dict()
        self.pair_frequency = dict()
        self.initial_data_sets = dict()
        
        for files in listfiles:

            if re.search('_out$',files):
                pairfiles.append(inputfolder+files)

            if re.search('_out_word$',files):
                wordfiles.append(inputfolder+files)

        for wfile in wordfiles:
            self.combineWordsFrequencies(wfile)
            
        self.word2id = dict()
        self.id2word = dict()
        self.wid_frequency = dict()
            
        self.words2Indices(minWordCount)
        
        self.init_sample_table()
        
        #ipdb.set_trace()
        
        
        for i,pfile in enumerate(pairfiles):
            #function = 'self.combinePairsFrequencies('+str(i)+',"'+pfile+'")'
            #print(cProfile.runctx(function,globals(),locals()))
            self.combinePairsFrequencies(i,pfile)
            
        self.pair2id = dict()
        self.id2pair = dict()
        
        self.pairs2Indices(minPairCount)
        
        
    def combineWordsFrequencies(self, wfile):

        with open(wfile) as inputFile:

            for lines in inputFile:
                item = lines.strip().split('>>>>')
                
                if item[0] in self.word_frequency:
                    self.word_frequency[item[0]] += int(item[1]) 
                else:
                    self.word_frequency[item[0]] = int(item[1])
                    
        
        inputFile.close()
        
    def combinePairsFrequencies(self, i,pfile):
        
        with open(pfile) as inputFile:

            for lines in inputFile:
                item = lines.strip().split('>>>>')
                
                if len(item) != 3:
                    print("Ëxceptional items!", item)
                    continue
                
                pair = item[0].split('[}')
                
                if len(pair) != 2:
                    print("Exceptional pairs",item[0])
                    continue
                    
                word1 = pair[0]
                word2 = pair[1]
                
                if word1 in self.word2id:
                    #does word1 satisfies minimum count?
                    wid1 = self.word2id[word1]
                else:
                    continue
                
                
                if word2 in self.word2id:
                    #does word2 satisfies minimum count?
                    wid2 = self.word2id[word2]
                else:
                    continue
                
                pairCount = int(item[1])
                
                if (wid1,wid2) in self.pair_frequency:
                    self.pair_frequency[(wid1,wid2)] += pairCount
                else:
                    self.pair_frequency[(wid1,wid2)] = pairCount
            
                
                words = item[2].strip().split('{]')
                
                for elements in words[:-1]:
                    elements = elements.strip().split('[}')
                    
                    in_word = elements[0]
                    
                    
                    if in_word in self.word2id:
                        in_wid = self.word2id[in_word]
                    else:
                        continue
                        #in between word is too less in frequency, hence ignored from dataset.
                    
                    in_wordCount = int(elements[1])
                    
                    if (wid1,wid2,in_wid) in self.initial_data_sets:
                        self.initial_data_sets[(wid1,wid2,in_wid)] += in_wordCount   
                    else:
                        self.initial_data_sets[(wid1,wid2,in_wid)] = in_wordCount 
                    
                    
                
                    
        inputFile.close()
        print("\n ",i,"data_sets size", sys.getsizeof(self.initial_data_sets)/(1024*1024), 'count:', len(self.initial_data_sets))
    
    def words2Indices(self, wordCount=200):
        
        #Remove the file if it already exists
        try:
            os.remove(self.wdictfile)
        except:
            pass
            
        wid = 0
        outputFile = open(self.wdictfile,"a")
        for word in self.word_frequency:
            if self.word_frequency[word] >= wordCount:
                self.word2id[word] = wid
                self.id2word[wid] = word
                self.wid_frequency[wid] = self.word_frequency[word]
                
                #with open(self.wdictfile,"a") as outputFile: 
                outputFile.write(word+"[}"+str(wid)+"\n")
                wid += 1
        
        #Release memory
        
        self.word_frequency = dict()
        outputFile.close()
        print( "\n # Words : ", len(self.word2id))
        
        
    def init_sample_table(self):
        
        print("\n Making sample table")
        
        self.sample_table = []
        sample_table_size = 1e8
        
        pow_frequency = numpy.array(list(self.wid_frequency.values()))**0.75  # 3/4 of the power of pairs
        words_pow = sum(pow_frequency) # initial calculation
        ratio = pow_frequency / words_pow # initial calculation  
        count = numpy.round(ratio * sample_table_size) # This is for sampling indices on the same ratio over 1e8  # initial implementation

        for wid, c in enumerate(count):
            self.sample_table += [wid] * int(c)
                
        self.sample_table = numpy.array(self.sample_table)

        
    def pairs2Indices(self, pairCount=100):
        
        #Remove the file if it already exists
        try:
            os.remove(self.pdictfile)
        except:
            pass
        
        pid = 0
        outputFile = open(self.pdictfile,"a")
        for pair in self.pair_frequency:
            if self.pair_frequency[pair] >= pairCount:
                self.pair2id[pair] = pid
                self.id2pair[pid] = pair
                
                wid1 = pair[0]
                wid2 = pair[1]
                w1 = self.id2word[wid1]
                w2 = self.id2word[wid2]
                word_pair = w1+"[}"+w2
                #with open(self.pdictfile,"a") as outputFile: 
                outputFile.write(word_pair+"[}"+str(pid)+"\n")
                
                pid += 1
    
        #Release memory
        self.pair_frequency = dict()
        outputFile.close()
        print( "\n # Pairs : ", len(self.pair2id))  
        
        
        
    def makeTriplesets(self,k=5):
        
        #Remove the file if it already exists
        try:
            os.remove(self.posfile)
        except:
            pass
        
        #Remove the file if it already exists
        try:
            os.remove(self.negfile)
        except:
            pass
        
        #Remove the file if it already exists
        try:
            os.remove(self.statfile)
        except:
            pass
            
        dataset_size = 0
        
        posOutputFile = open(self.posfile, "a")
        negOutputFile = open(self.negfile, "a")
        
        for dataset in self.initial_data_sets:
            
            count = self.initial_data_sets[dataset]
            
            
            wid1 = dataset[0]
            wid2 = dataset[1]
            iwid = dataset[2]
            
            pair = (wid1,wid2)
            
            if pair in self.pair2id:
                pid = self.pair2id[pair]
                for _ in range(count):
                    #for every positive sample, create 'k' negative samples
                    neg_v = self.get_neg_v_neg_sampling(k)
                    #with open(self.posfile,"a") as outputFile:
                    output = str((pid,iwid))
                    posOutputFile.write(output+"\n")
                        
                    #with open(self.negfile,"a") as outputFile:
                    output = neg_v
                    negOutputFile.write(str(output)+"\n")
                        
                    dataset_size += 1

        posOutputFile.close()
        negOutputFile.close()
        
        with open(self.statfile,"a") as statFile:
            statFile.write("Dataset :"+str(dataset_size))

                           
    #generate 'k'negative samples for every positive sample        
    def get_neg_v_neg_sampling(self, count):
        neg_v = numpy.random.choice(self.sample_table, size=(count)).tolist()
        return neg_v
         
if __name__ ==  '__main__':
    
    inputfolder = sys.argv[1]
    outputfolder = sys.argv[2]
    minWordCount = 700
    minPairCount = 200
    k = 5

    data = Datasets(inputfolder, outputfolder, minWordCount, minPairCount)
    #print(cProfile.runctx('data.makeTriplesets(k)',globals(),locals()))
    data.makeTriplesets(k)
    
