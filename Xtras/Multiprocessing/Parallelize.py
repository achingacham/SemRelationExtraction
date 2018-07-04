

#parallelize the search of a word in huge file:
import sys
import re

class Parallelize(file_name, word):
    
    def __int__(self):
        self.file_name = file_name
        self.word = word
        self.count = 0
        readFile(self.file_name)
        
        print("\n ", self.word, self.count)
        
    def preprocess(self,line):
        if re.match(self.word,line):
            self.count += 1

    def readFile(self):
        with open(input_file_name) as inputFile:

            for sentence in inputSentences[0:100]:
                preprocess(sentence)
            
        inputFile.close()

        
if __name__ == '__main__':
    input_file_name=sys.argv[1]
    word = sys.argv[2]
    p = Parallelize(input_file_name,word)