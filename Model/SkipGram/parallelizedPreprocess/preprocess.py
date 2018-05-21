
# coding: utf-8
import sys
import os
import re
import pdb
import numpy 
#from multiprocessing import Pool
#from multiprocessing import Process

class Preprocess:
    
    def __init__(self, inputfile, outputfile, window_size):
        
        self.ifile = inputfile
        self.ofile = outputfile
        self.window = window_size
        
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
            
            for pairs in self.initial_pair_frequency:
                
                output = str(pairs)+">>>>"+str(self.initial_pair_frequency[pairs][0])+">>>>"
                
                for word in self.initial_pair_frequency[pairs][1]:
                    count = self.initial_pair_frequency[pairs][1][word]
                    output += word+"[}"+str(count)+"{]"           #delimiters '[}' and  '{]'
               
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
                output = str(words)+">>>>"+str(self.initial_word_frequency[words]) #delimiter '>>>>'
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
                
                for pos in ['NNS','NN','NP']:
                    if pos in line[1::2]:
                        line_numpy_arr = numpy.array(line[1::2])
                        Noun_positions = numpy.append(Noun_positions,(numpy.where(line_numpy_arr == pos)[0]*2))

                Noun_positions.sort()

                self.trackPairs(line, Noun_positions)

                count += 1

        print("\n", self.ifile,":", count)
        
        inputFile.close()

    def trackPairs(self, line, Noun_positions):

        #find the possible Nouns pairs in the sentence, with in the window. Ignore Noun pairs apprearing adjascent
        for index_1, n1_position in enumerate(Noun_positions):
            for index_2, n2_position in enumerate(Noun_positions[index_1+1:]):

                if (abs(n1_position - n2_position) > (2*self.window)) or (abs(n1_position - n2_position) < 3):
                    break

                #track the possible inbetween words for any pair of Nouns
                else:

                    n1 = line[n1_position].lower()
                    n2 = line[n2_position].lower()

                    #avoid all pairs of same noun eg: ('school','school')
                    if n1 == n2:
                        continue
                    
                    pair_key = n1+"[}"+n2
                    
                    if pair_key in self.initial_pair_frequency:

                        self.initial_pair_frequency[pair_key][0] += 1

                        for items in line[n1_position+2:n2_position:2]:
                            ###
                            items = items.lower()
                            ###
                            if items in self.initial_pair_frequency[pair_key][1]:
                                
                                self.initial_pair_frequency[pair_key][1][items] += 1
                            else:
                                self.initial_pair_frequency[pair_key][1].update({items:1})
                            
                    else:
                        #track pair frequency for each Noun pair
                        #store all words inbetween the Noun pairs with their individual count.

                        self.initial_pair_frequency[pair_key] = [1,{}]
                        
                        for items in line[n1_position+2:n2_position:2]:
                            self.initial_pair_frequency[pair_key][1].update({items:1})
                
                    
if __name__ ==  '__main__':
    
    inputfile = sys.argv[1]
    outputfile = sys.argv[2]
    window_size = int(sys.argv[3]) #7
    
    pp = Preprocess(inputfile, outputfile,window_size)  
    
    
    '''
    # multi threading
    listfiles = os.listdir(folder)
    poolList = []
    for file in listfiles:
        
        if re.match('splitFile_',file):
            
            p = Process(target=trackTriples, args=(folder,file,))
            p.start()
            p.join()
    '''
    #p = Pool(10)
    #p.map(trackTriples, poolList)cat
   
    
    
