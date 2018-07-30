
# coding: utf-8
import sys
import os
import re
import pdb
import numpy 
#from multiprocessing import Pool
#from multiprocessing import Process

class Preprocess:
    
    '''
        This class is executed for every split file on multiple coresif available.
        Each file is preprocessed to track :
            count of every word (initial_word_frequency)
            count of every noun pair (initial_pair_frequency)
            distance between nouns in the pair (initial_pair_frequency)
            words in-between the noun pair (initial_pair_frequency)
            count of in-between words (initial_pair_frequency)
    '''
    
    def __init__(self, inputfile, outputfile, threshold_distance):
        
        self.ifile = inputfile
        self.ofile = outputfile
        self.threshold_distance = threshold_distance
        
        self.initial_pair_frequency = dict()
        self.initial_word_frequency = dict()
        
        self.trackTriples()
        self.writeOutput()
        
    
    def writeOutput(self):
        
        #Remove the file if it already exists
        try:
            os.remove(self.ofile)
        except:
            pass
            
        
        with open(self.ofile,"a") as outputFile:
            
            for pair_distance in self.initial_pair_frequency:
                
                
                for pair_key in self.initial_pair_frequency[pair_distance]:
                    
                    output = str(pair_distance)+"\t\t"+str(pair_key)+"\t\t"+str(self.initial_pair_frequency[pair_distance][pair_key][0])+"\t\t"

                    for word in self.initial_pair_frequency[pair_distance][pair_key][1]:
                        count = self.initial_pair_frequency[pair_distance][pair_key][1][word]
                        output += word+"\t"+str(count)+"\t"           #delimiters '[}' and  '{]'

                    outputFile.write(output+"\n")
                
        outputFile.close()
        
        #Remove the file if it already exists
        
        try:
            os.remove(self.ofile+"_word")
        except:
            pass
            
        
        
        with open(self.ofile+"_word","a") as outputFile:
            
            for words in self.initial_word_frequency:
                #print(words)
                output = str(words)+"\t"+str(self.initial_word_frequency[words]) #delimiter '>>>>'
                outputFile.write(output+"\n")
                
        outputFile.close()
        
        
    
    def trackTriples(self):

        with open(self.ifile) as inputFile:
            count = 0

            for lines in inputFile:

                line = lines.strip().split()
                
                for i,w in enumerate(line):
                    #ignore all POS tags, skip alternate words
                    if i % 2 == 0:                          

                        try:
                            self.initial_word_frequency[w.lower()] += 1
                        except:
                            self.initial_word_frequency[w.lower()] = 1

                Noun_positions = numpy.array([], dtype = int)
                
                for pos in ['NNS','NN','NP','NPS']:
                    if pos in line[1::2]:
                        line_numpy_arr = numpy.array(line[1::2])
                        Noun_positions = numpy.append(Noun_positions,(numpy.where(line_numpy_arr == pos)[0]*2))

                Noun_positions.sort()
                
                
                self.trackPairs(line, Noun_positions)

                count += 1

        
        inputFile.close()

    def trackPairs(self, line, Noun_positions):

        #find the possible Nouns pairs in the sentence, with in the window. Ignore Noun pairs apprearing adjascent
        for index_1, n1_position in enumerate(Noun_positions):
            for index_2, n2_position in enumerate(Noun_positions[index_1+1:]):
                
                
                pair_distance = int(abs(n1_position - n2_position)/2)
                
                #Ignore all compound words 
                
                if (pair_distance < 3):  
                    
                    continue
                
                #Exit inner loop once the distance between nouns exceeds 'threshold_distance'
                
                if (pair_distance > self.threshold_distance):
                    
                    break
                #track the possible inbetween words for any pair of Nouns
                else:

                    n1 = line[n1_position].lower()
                    n2 = line[n2_position].lower()
                    
                    
                    #avoid all pairs of same noun eg: ('school','school')
                    if n1 == n2:
                        continue
                    
                    pair_key = n1+"\t"+n2 # PairKey is (n1 n2) 
                    
                    
                    if pair_distance in self.initial_pair_frequency:

                        # Update pair_count
                        if pair_key in self.initial_pair_frequency[pair_distance]:
                            
                            self.initial_pair_frequency[pair_distance][pair_key][0] += 1
                            
                        else:
                            
                            
                            self.initial_pair_frequency[pair_distance][pair_key] = [1,{}]
                            
                            
                        for items in line[n1_position+2:n2_position:2]:
                            ###
                            items = items.lower()
                            ###
                            if items in self.initial_pair_frequency[pair_distance][pair_key][1]:

                                self.initial_pair_frequency[pair_distance][pair_key][1][items] += 1
                            else:
                                self.initial_pair_frequency[pair_distance][pair_key][1].update({items:1})

                    else:
                        #track pair frequency for each Noun pair, with respective distnace between them
                        #store all words inbetween the Noun pairs with their individual count.

                        
                        self.initial_pair_frequency[pair_distance] = dict()
                        
                        self.initial_pair_frequency[pair_distance][pair_key] = [1,{}]
                        
                        for items in line[n1_position+2:n2_position:2]:
                            self.initial_pair_frequency[pair_distance][pair_key][1].update({items:1})
                
                
        
if __name__ ==  '__main__':
    
    '''
    Inputs:
    
            inputfile          :  The corpora file(or a small chunk of it) with sentences per line and each word followed by its POS tag
            outputfile         :  Files with word dictionaries and pair dictionaries with their repective counts
                                  Pair dictionary also incluses the dictances and inbetween word with their counts information.
            threshold_distance :  Maximum distance possible between a noun pair to be considered. 
                                  POS tags are ignored for distance calculation
    
    
    '''
    
    inputfile = sys.argv[1]
    outputfile = sys.argv[2]
    threshold_distance = int(sys.argv[3])
    
    pp = Preprocess(inputfile, outputfile, threshold_distance)  
    
    
   
    
    
