import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

file = pd.read_csv("sample_word_embedding.txt",names=[str(i) for i in range(101)],delimiter=' ', skiprows=1)
features = [str(i) for i in range(1,101)]


x = file.loc[:, features].values
y = file.loc[:,['0']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)
#print(y)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])


finalDf = pd.concat([principalDf, file[['0']]], axis=1)


import matplotlib.pyplot as plt
%matplotlib inline

fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(1,1,1)

ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

targets = y

#colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
for target in zip(targets):
    indicesToKeep = finalDf['0'] == target[0][0]

    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
    ax.annotate( target[0][0], (finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']))

#ax.legend(targets)
ax.grid()

#pca.explained_variance_ratio_ 
