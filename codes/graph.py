#returns the adjacency list read from .mat file

from scipy.io import loadmat
from collections import defaultdict
import numpy as np
import random
import networkx as nx

class Graph(defaultdict):
	def __init__(self):
		super(Graph, self).__init__(list)
	
	def nodes_number(self):
		return len(self)
	
	def random_walk(self, path_len, start = None, alpha = 0, rand = random.Random()):
		adlist = self
		if start:
			path = [start]
		else:
			path = [rand.choice(list(adlist.keys()))]
		while len(path) < path_len:
			cur = path[-1]
			if len(adlist[cur]) > 0:
				if rand.random() >= alpha:
					path.append(rand.choice(adlist[cur]))
				else:
					path.append(path[0])
			else:
				break
		return [node  for node in path]


	def build_deepwalk_corpus(self, num_paths, path_length, alpha=0,
		              rand=random.Random(0)):
		adlist = self
	 	walks = []

	  	nodes = list(adlist.keys())
	  
	  	for cnt in range(num_paths):
	    		rand.shuffle(nodes)
	    		for node in nodes:
	      			walks.append(adlist.random_walk(path_length,rand = rand,alpha = alpha,start=node))
		return walks


def readfile(filename):
		matfile = loadmat(filename)
		mat = matfile['network']
		return numpy_sparse(mat)
def numpy_sparse(x):
		G = Graph()
		sx = x.tocoo()
		for i,j,v in zip(sx.row, sx.col, sx.data):
			G[i].append(j)
		return G
def from_edgelist(filename):      #to read from .edgelist file
		G = Graph()		
		t = nx.read_edgelist(filename)
		t = nx.to_dict_of_lists(t)
		t = {int(k):[int(i) for i in v] for k,v in t.items()}
		for key,value in t.items():
			G[key].extend(value)
		return G
#G = from_edgelist('karate.edgelist')
#G = readfile('dataset/bc/blogcatalog.mat')
#print G[33]
#walks = G.build_deepwalk_corpus(20,80)
#print len(walks)
#np.savetxt( 'walks_kc_new.txt',np.array(walks), delimiter = ',', fmt = '%d')
#np.save( 'walks_kc.npy', np.array(walks))
