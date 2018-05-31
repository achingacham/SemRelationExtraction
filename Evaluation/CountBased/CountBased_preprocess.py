import sys
import os
import numpy as np

def readDictionaries(pairFile, wordFile):
    
    
    with open(pairFile) as inputFile:
        
        for line in inputFile:
            word1,word2,pid = line.split()
            pairDict[int(pid)] = word1+'\t'+word2
    
    inputFile.close()    
    
    with open(wordFile) as inputFile:
        
        for line in inputFile:
            word,wid = line.split()
            wordDict[int(wid)] = word
            wordDictRev[word] = wid
    
    inputFile.close()    
    
    
    print("Completed reading Pair and Word dictionaries of respective sizes: ", len(pairDict), len(wordDict))

def convertId2Words(posTripleFile):
    
    inputFile = open(posTripleFile)
    
    try:
        os.remove(posTripleFile+"_converted")
    except:    
        pass
    
    outputFile = open(posTripleFile+"_converted","a")
    
    for index_lines,lines in enumerate(inputFile):
        count,pid,wid = lines.split()
        pid = int(pid.strip('(,)'))
        wid = int(wid.strip('(,)'))
        
        outputFile.write(pairDict[pid]+"\t\t"+wordDict[wid]+"\t\t"+count+"\n")
        
        if index_lines % 1000000 == 0:
            print(index_lines, "completed")
            
    
    outputFile.close()
    inputFile.close()
    print("Completed convertion of positive tripleset")

def convertPid2Wids(posTripleFile):
    
    inputFile = open(posTripleFile)
    
    try:
        os.remove(posTripleFile+"_Pid2Wid")
    except:    
        pass
    
    outputFile = open(posTripleFile+"_Pid2Wid","a")
    
    for index_lines,lines in enumerate(inputFile):
        pid,wid = lines.split()
        pid = int(pid.strip('(,)'))
        word1, word2 = pairDict[pid].split()
        wid = wid.strip('(,)')
        #wordID1 | wordID2 | contextwordID | [Count]
        
        outputFile.write(wordDictRev[word1]+"\t\t"+wordDictRev[word2]+"\t\t"+wid+"\n")
        
        if index_lines % 1000000 == 0:
            print(index_lines, "completed")
            
    
    outputFile.close()
    inputFile.close()
    print("Completed convertion of positive tripleset")
    

if __name__ == "__main__":
    
    ifolder = sys.argv[1]
    pairCount = int(sys.argv[2])
    wordCount = int(sys.argv[3])
    
    temp = np.array([1] * pairCount)
    pairindices = np.where(temp==1)[0]
    pairDict = dict.fromkeys(pairindices,'a\tb')
    
    temp = np.array([1] * wordCount)
    wordindices = np.where(temp==1)[0]
    wordDict = dict.fromkeys(wordindices,'a')
    wordDictRev = dict()
    
    pairFile = ifolder+"/Pair2Id"
    wordFile = ifolder+"/Word2Id"
    posTripleFile = ifolder+"Triplesets_positive"
    
    readDictionaries(pairFile, wordFile)
    
    ######convertId2Words(posTripleFile)
    
    convertPid2Wids(posTripleFile)