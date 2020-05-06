"""
Graph Mining - ALTEGRAD - Dec 2019
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
os.chdir('/content/drive/My Drive/TP5/code/datasets')

# ############# Task 1


G = nx.read_edgelist('CA-HepTh.txt', comments='#',
                     delimiter='\t', create_using=nx.Graph())
G
# print("Nodes:", G.number_of_nodes())
# print("Edges:", G.number_of_edges())

# ############# Task 2

num_cc = nx.number_connected_components(G)
print("Number of connected components :", num_cc)

largest_cc = max(nx.connected_components(G), key=len)
gcc = G.subgraph(largest_cc)

print("Nodes:", gcc.number_of_nodes())
print("Edges:", gcc.number_of_edges())

print("Fraction of nodes in gcc:",
      round(gcc.number_of_nodes()/G.number_of_nodes() * 100, 2), "%")
print("Fraction of edges in gcc:",
      round(gcc.number_of_edges()/G.number_of_edges() * 100, 2), "%")

# ############# Task 3

# Degree
degree_sequence = [G.degree(node) for node in G.nodes()]
degree_array = np.array(degree_sequence)

print("Min:", np.min(degree_array))
print("Max:", np.max(degree_array))
print("Mean:", round(np.mean(degree_array), 0))
print("Median:", np.median(degree_array))

# ############# Task 4

vals = nx.degree_histogram(G)

plt.plot(vals)
plt.xlabel("Degree")
plt.ylabel("Count")
plt.title("Degree histogram of graph G")
plt.show()
