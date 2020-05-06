"""
@author: Khaoula Belahsen
@email : khaoula.belahsen@yahoo.fr
"""

import string
from nltk.corpus import stopwords
import igraph
import matplotlib.pyplot as plt 
#os.chdir('/home/khaoula/Documents/MVA/ALTEGRAD/for_moodle/') # to change working directory to where functions live
# import custom functions
from library import clean_text_simple, terms_to_graph, core_dec

stpwds = stopwords.words('english')
punct = string.punctuation.replace('-', '')

my_doc = 'A method for solution of systems of linear algebraic equations \
with m-dimensional lambda matrices. A system of linear algebraic \
equations with m-dimensional lambda matrices is considered. \
The proposed method of searching for the solution of this system \
lies in reducing it to a numerical system of a special kind.'

my_doc = my_doc.replace('\n', '')

# pre-process document
my_tokens = clean_text_simple(my_doc,my_stopwords=stpwds,punct=punct)

g = terms_to_graph(my_tokens, 4)


# number of edges
print(len(g.es))

"""
@author: Khaoula Belahsen
@email : khaoula.belahsen@yahoo.fr
"""

# the number of nodes should be equal to the number of unique terms
len(g.vs) == len(set(my_tokens))
edge_weights = []
for edge in g.es:
    source = g.vs[edge.source]['name']
    target = g.vs[edge.target]['name']
    weight = edge['weight']
    edge_weights.append([source, target, weight])
    
print(edge_weights)
g.vs["label"]=g.vs["name"]
layout = g.layout("kk")
g4 = igraph.plot(g)
g4.show()

gd=[]
for w in range(2,10):
    g = terms_to_graph(my_tokens, w)
    gd.append(g.density())
    print(g.density())
    ### fill the gap (print density of g) ###
plt.plot(gd)

plt.xlabel('window size')
plt.ylabel('graph density')

plt.title('Graph density as a function of window size')

plt.savefig("q1.png")

plt.show()
# decompose g

core_numbers = core_dec(g,False)
print('core_numbers :',core_numbers,'\n')

### fill the gap (compare 'core_numbers' with the output of the .coreness() igraph method) ###

print(list(core_numbers.values()) == g.coreness())
print('coreness : ',dict(zip(g.vs['name'],g.coreness())),'\n')

# retain main core as keywords

max_c_n = max(core_numbers.values())
keywords = [kwd for kwd, c_n in core_numbers.items() if c_n == max_c_n]
print(keywords)
