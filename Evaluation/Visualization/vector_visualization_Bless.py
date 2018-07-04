
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from modelData import modelData

import os
import sys
#get_ipython().run_line_magic('matplotlib', 'inline')


def calculateDistanceWith(k):
    
    p1 = x_coord[k]
    p2 = y_coord[k]
    
    return (np.sqrt(np.square(x_coord-p1)+(np.square(y_coord-p2))))       

def findNearestNeighbor(k,t,nPairs):
    
    relativeDistance = calculateDistanceWith(k)
    dtype = [('distance',float),('label','U50')]
    distances = []
    
    for i,d in enumerate(relativeDistance):
        distances.append((d, label[i]))
    
    temp_array = np.array(distances, dtype=dtype)
    ordered_array = np.sort(temp_array, order='distance')
    
    return ordered_array[:t+1]

# In[7]:
def drawPlot(finalDf, y, nearestNeighbors, title, col1, col2):

    fig = plt.figure(figsize = (20,20))
    ax = fig.add_subplot(1,1,1)

    ax.set_xlabel('Principal Component 1', fontsize = 20)
    ax.set_ylabel('Principal Component 2', fontsize = 20)
    ax.set_title('2 component PCA', fontsize = 35)

    targets = y

    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    classes = finalDf['class'].unique()
    colors = []

    for index,value in enumerate(classes):
        finalDf.loc[finalDf['class']==value, 'color'] = 'C'+str(index)
        colors.append('C'+str(index))

    #randomness = np.random.randint(0,len(y),100)
    
    for r in range(len(y)):

        target = targets[r]
        pair = target[0]
        indicesToKeep = finalDf['0'] == pair
        
        ax.scatter(finalDf.loc[indicesToKeep, col1]
               , finalDf.loc[indicesToKeep, col2]
               , c = finalDf.loc[indicesToKeep, 'color']
               , s = 50)


        ax.annotate( pair, (finalDf.loc[indicesToKeep, col1]
               , finalDf.loc[indicesToKeep, col2]))


    for items in nearestNeighbors:

        pair = items[1]
        indicesToKeep = finalDf['0'] == pair


        ax.scatter(finalDf.loc[indicesToKeep, col1]
               , finalDf.loc[indicesToKeep, col2]
               , c = finalDf.loc[indicesToKeep, 'color']
               , marker = 'x'  
               , s = 50)

        ax.annotate( pair, (finalDf.loc[indicesToKeep, col1]
               , finalDf.loc[indicesToKeep, col2]))



    recs = []
    for i in range(0,len(colors)):
        recs.append(mpatches.Rectangle((0,0),1,1,fc=colors[i]))
    ax.legend(recs,classes,loc=4)

    ax.grid()
    fig.savefig(outfolder+title+"_PC_Analysis_Bless.png")

    
if __name__ == '__main__':
    
    ifolder = sys.argv[1]
    ofolder = sys.argv[2]
    validationfile = sys.argv[3]
    preTrainedWordEmbedding = sys.argv[4]

    validatedBlessSet = ifolder+validationfile+'.txt'
    outfolder = ofolder+ifolder.rsplit('/',2)[1]+'/Visualization/'

    try:
        os.mkdir(outfolder)
    except:
        print(outfolder+ " folder exists. Will be overwritten")

    with open(ifolder+"Epoch_0_EMB_Bless.txt") as getParam:
        line = getParam.readline()

    line = line.split()

    numberPairs = int(line[0])
    emb_dimension = int(line[1])
    emb_dimension += 1

    file = pd.read_csv(ifolder+"Epoch_0_EMB_Bless.txt",names=[str(i) for i in range(emb_dimension)],delimiter=' ', skiprows=1)

    features = [str(i) for i in range(1,emb_dimension)]

    x = file.loc[:, features].values
    y = file.loc[:,['0']].values
    
    # Standardizing the features
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, file[['0']]], axis=1)
    
    
    
    #DATA
    inputData = modelData(validatedBlessSet, "visualization", finalDf)
    inputDim = 400
    inputData.create_dictWordVectors(preTrainedWordEmbedding, inputDim)
    
    
    
    ## DiffVector visualization
    inputData.create_diffVectors(finalDf)
    index_keys = 0
    
    for keys in inputData.dictDiffVectors:
        x[index_keys] = inputData.dictDiffVectors[keys]
        y[index_keys] = finalDf.loc[finalDf['0']==keys, '0']
        index_keys += 1
    
    # Standardizing the features
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    
    principalDf = pd.DataFrame(data = principalComponents, columns = ['new principal component 1','new principal component 2'])
    finalDfDiff = pd.concat([finalDf, principalDf], axis=1)
    
    
    # In[3]:
    x_coord = finalDf['principal component 1']
    y_coord = finalDf['principal component 2']
    label = finalDf['0']
    # In[18]:
    list_label = [i for i in label]
    random = list_label.index('phone:::computer')
    top = 10
    nearestNeighbors = findNearestNeighbor(random,top,numberPairs)
    print("\n Top ",top, " Nearest neighbors of ",label[random] , " with JustRel vectors")
    for items in nearestNeighbors:
        print(items)

    drawPlot(finalDf, y, nearestNeighbors, "RelVectors", 'principal component 1', 'principal component 2')
    
    # In[3]:
    x_coord = finalDfDiff['new principal component 1']
    y_coord = finalDfDiff['new principal component 2']
    label = finalDfDiff['0']
    # In[18]:
    list_label = [i for i in label]
    random = list_label.index('phone:::computer')
    top = 10
    nearestNeighbors = findNearestNeighbor(random,top,numberPairs)
    print("\n Top ",top, " Nearest neighbors of ",label[random], " with JustWord vectors")
    for items in nearestNeighbors:
        print(items)

    drawPlot(finalDfDiff, y, nearestNeighbors, "DiffVector",'new principal component 1', 'new principal component 2')
    