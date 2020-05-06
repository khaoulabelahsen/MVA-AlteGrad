"""
Deep Learning on Graphs - ALTEGRAD - Dec 2019
"""

import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import accuracy_score
from deepwalk import deepwalk
import os 
os.chdir("/content/drive/My Drive/TP6/code/data/")

# Loads the karate network
G = nx.read_weighted_edgelist('karate.edgelist', delimiter=' ', nodetype=int, create_using=nx.Graph())
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

n = G.number_of_nodes()

# Loads the class labels
class_labels = np.loadtxt('../data/karate_labels.txt', delimiter=',', dtype=np.int32)
idx_to_class_label = dict()
for i in range(class_labels.shape[0]):
    idx_to_class_label[class_labels[i,0]] = class_labels[i,1]

y = list()
for node in G.nodes():
    y.append(idx_to_class_label[node])

y = np.array(y)



############## Task 5

colors = []

for node in G.nodes():
  if idx_to_class_label[node] == 1 :
    colors.append('r')
  else : 
    colors.append('b')

nx.draw_networkx(G, node_color = colors)




############## Task 6
n_dim = 128
n_walks = 10
walk_length = 20
model = deepwalk(G, n_walks, walk_length, n_dim)  # your code here

embeddings = np.zeros((n, n_dim))
for i, node in enumerate(G.nodes()):
    embeddings[i,:] = model.wv[str(node)]

idx = np.random.RandomState(seed=42).permutation(n)
idx_train = idx[:int(0.8*n)]
idx_test = idx[int(0.8*n):]

X_train = embeddings[idx_train,:]
X_test = embeddings[idx_test,:]

y_train = y[idx_train]
y_test = y[idx_test]



############## Task 7

lr = LogisticRegression(C=1, solver= 'lbfgs')
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print("The accurary score:", accuracy_score(y_test, y_pred)) # 0.8571428571428571




############## Task 8

model = SpectralEmbedding(affinity= 'rbf')  # your code here
embeddings = model.fit_transform(G)

embeddings = to_numpy_matrix(embeddings)
idx = np.random.RandomState(seed=42).permutation(n)
idx_train = idx[:int(0.8*n)]
idx_test = idx[int(0.8*n):]

X_train_se = embeddings[idx_train,:]
X_test_se = embeddings[idx_test,:]

y_train_se = y[idx_train]
y_test_se = y[idx_test]

lr = LogisticRegression(C=1, solver= 'lbfgs')
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print("The accurary score:", accuracy_score(y_test, y_pred)) # 0.8571428571428571