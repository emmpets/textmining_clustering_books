__author__ = 'Manos'

import re
import nltk
from nltk.corpus import stopwords
import pickle
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances,cosine_similarity,manhattan_distances
# from sklearn.metrics.pairwise import cosine_similarity

import os  # for os.path.basename
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import linkage,centroid,ward,complete ,dendrogram

import os  # for os.path.basename
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.decomposition import PCA



dtm = pickle.load( open( "tfidf_matrix.p", "rb" ) )
vocab = pickle.load( open( "voc.p", "rb" ) )


print(dtm.shape)
# dtm = dtm.toarray()


# PCA
pca = PCA(n_components=6000, copy=True)
pca.fit(dtm.toarray())
dtm = pca.transform(dtm.toarray())

vocab = np.array(vocab)

n, _ = dtm.shape



dist = np.zeros((n, n))
def zero_space(dist):
    for i in range(n):
        for j in range(n):
            x, y = dtm[i, :], dtm[j, :]
            dist[i, j] = np.sqrt(np.sum((x - y)**2))
    return dist

dist_eukl= zero_space(dist)
dist_cos = zero_space(dist)
dist_man = zero_space(dist)


dist_eukl = euclidean_distances(dtm)
np.round(dist_eukl, 1)

dist_cos = 1 - cosine_similarity(dtm)
np.round(dist_cos, 2)

dist_man = manhattan_distances(dtm)
np.round(dist_man, 1)

norms = np.sqrt(np.sum(dtm * dtm, axis=1, keepdims=True))

dtm_normed = dtm / norms
similarities = np.dot(dtm_normed, dtm_normed.T)
np.round(similarities, 2)



# Visualizing distances




mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
# pos = mds.fit_transform(dist_eukl)  # shape (n_components, n_samples)
pos = mds.fit_transform(dist_cos)  # shape (n_components, n_samples)
# pos = mds.fit_transform(dist_man)  # shape (n_components, n_samples)


xs, ys = pos[:, 0], pos[:, 1]
names = ['Dictionary Greek and Roman Geography','WORKS OF CORNELIUS TACITUS:an Essay on his Life and Genius',
         'THE HISTORY OF THE PELOPONNESIAN WAR','THE HISTORY OF THE DECLINE AND FALL OF THE ROMAN EMPIRE I ,GIBBON',
         'THE HISTORY OF ROME BY TITUS LIVIUS','THE WHOLE GENUINE WORKS OF FLAVIUS JOSEPHUS',
         'THE HISTORY OF THE DECLINE AND FALL OF THE ROMAN EMPIRE III ,GIBBON','THE DESCRIPTION OF GREECE',
         'THE HISTORY OF ROME by THEODOR MOMMSEN','HISTORY OF ROME III by LIVY ','THE HISTORY OF THE PELOPONNESIAN WAR I, BY THUCYDIDES',
         'HISTORY OF THE DECLINE AND FALL OF THE ROMAN EMPIRE IV , GIBSON','HISTORY OF THE DECLINE AND FALL OF THE ROMAN EMPIRE II , GIBSON',
         'ROMAN HISTORY by TITUS LIVIUS','The Historical Annals of Cornelius Tacitus I','THE WORKS Of JOSEPHUS LIFE IV',
         'THE WORKS OF CORNELIUS TACITUS IV: WITH AN ESSAY ON HIS LIFE AND GENIUS','HISTORY OF ROME V by LIVY','THE FLAVIUS JOSEPHUS Jewish Antiquities',
         'Plinys Natural History','THE HISTORY OF THE THE ROMAN EMPIRE V, GIBSON','THE HISTORIES CAIUS CORNELIUS TACITUS',
         'THE HISTORY DECLINE AND FALL ROMAN EMPIRE VI , GIBSON','THE WORKS OF FLAVIUS JOSEPHUS III']

cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e',5:'#7FFF00'}
cluster_names = {0: 'Josephus',
                 1: 'Tacitus',
                 2: 'Roman Empire',
                 3: 'History of Rome,Peloponisian War',
                 4: 'Livy',
                 5: 'Nothing'}


# linkage_matrix = ward(dist_eukl)
# linkage_matrix = ward(dist_cos)
# linkage_matrix = ward(dist_man)
# linkage_matrix = centroid(dist_eukl)
# linkage_matrix = centroid(dist_cos)
# linkage_matrix = linkage(dist_eukl,'weighted')
linkage_matrix = linkage(dist_cos,'weighted')




dendrogram(linkage_matrix, orientation="right", labels=names)
# dendrogram(linkage_matrix_centroid, orientation="right", labels=names)
# plt.tight_layout()  # fixes margins
plt.show()

num_clusters = 6

km = KMeans(n_clusters=num_clusters)
km.fit(dtm)
clusters = km.labels_.tolist()


df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=names))

#group by cluster
groups = df.groupby('label')


# set up plot
fig, ax = plt.subplots(figsize=(17, 9)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
            label=cluster_names[name], color=cluster_colors[name],
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')

ax.legend(numpoints=1,loc=3)  #show legend with only 1 point

#add label in x,y position with the label as the film title
for i in range(len(df)):
    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)

plt.show()