# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 20:41:53 2023

@author: Pandadada
"""

import numpy as np
import os,sys
sys.path.append(os.path.realpath('E:/Master Thesis/FGWD_on_Graphs_basics/lib_1.0'))
from graph import graph_colors,draw_rel,draw_transp,Graph,wl_labeling
from ot_distances import Fused_Gromov_Wasserstein_distance,Wasserstein_distance
import copy
# from data_loader import load_local_data,histog,build_noisy_circular_graph
import matplotlib.pyplot as plt
import networkx as nx


#%%
def build_comunity_graph(N=30,mu=0,sigma=0.3,pw=0.8):
    v=mu+sigma*np.random.randn(N);
    g=Graph()
    g.add_nodes(list(range(N)))
    for i in range(N):
         g.add_one_attribute(i,v[i])
         for j in range(N):
             r=np.random.rand()
             if  r<pw:
                 g.add_edge((i,j))
    return g

N=5
mu1=-1.5
mu2=1.5
vmin=-3
vmax=2
np.random.seed(12)
g1=build_comunity_graph(N=N,mu=mu1,sigma=0.8,pw=0.5)
g2=build_comunity_graph(N=N,mu=mu2,sigma=0.8,pw=0.5)

def merge_graph(g1,g2):
    gprime=nx.Graph(g1)
    N0=len(gprime.nodes())
    g2relabel=nx.relabel_nodes(g2, lambda x: x +N0)
    gprime.add_nodes_from(g2relabel.nodes(data=True))
    gprime.add_edges_from(g2relabel.edges(data=True)) 
    gprime.add_edge(N0-1,N0)
    
    return gprime

g3=merge_graph(merge_graph(g1.nx_graph,merge_graph(g1.nx_graph,g2.nx_graph)),g2.nx_graph)

g4=merge_graph(merge_graph(g2.nx_graph,merge_graph(g1.nx_graph,g2.nx_graph)),g1.nx_graph)

plt.figure(figsize=(8,5))
draw_rel(g3,vmin=vmin,vmax=vmax,with_labels=False,draw=False)
draw_rel(g4,vmin=vmin,vmax=vmax,with_labels=False,shiftx=3,draw=False)
plt.title('Two graphs. Color indicates the label')
plt.show()

G1=Graph(g3)
G2=Graph(g4)

#%%
nodes1=G1.nodes()
nodes2=G2.nodes()
p1=np.ones(len(nodes1))/len(nodes1)
p2=np.ones(len(nodes2))/len(nodes2)


alld=[]
x=np.linspace(0,1,10)
for alpha in x:
    d,log,transp=Fused_Gromov_Wasserstein_distance(alpha=alpha,features_metric='sqeuclidean').graph_d(G1,G2,p1,p2)
    alld.append(d)
plt.plot(x,alld)
plt.title('Evolution of FGW dist in wrt alpha \n max={}'.format(x[np.argmax(alld)]))
plt.xlabel('Alpha')
plt.xlabel('FGW dist')
plt.show()

# FGWD
fig=plt.figure(figsize=(10,8))
thresh=0.004
dfgw,log_FGWD,transp_FGWD=Fused_Gromov_Wasserstein_distance(alpha=0.8,features_metric='sqeuclidean').graph_d(G1,G2,p1,p2)
# d=gwdist.graph_d(G1,G2)
# plt.title('FGW coupling, dist : '+str(np.round(d,3)),fontsize=15)
draw_transp(G1,G2,transp_FGWD,shiftx=2,shifty=0.5,thresh=thresh,
            swipy=True,swipx=False,with_labels=False,vmin=vmin,vmax=vmax)
plt.show()

# GWD
fig=plt.figure(figsize=(10,8))
thresh=0.004
dgw,log_GWD,transp_GWD=Fused_Gromov_Wasserstein_distance(alpha=1,features_metric='sqeuclidean').graph_d(G1,G2,p1,p2)
# d=gwdist.graph_d(G1,G2)
# plt.title('GW coupling, dist : '+str(np.round(d,3)),fontsize=15)
draw_transp(G1,G2,transp_GWD,shiftx=2,shifty=0.5,thresh=thresh,
            swipy=False,swipx=False,with_labels=False,vmin=vmin,vmax=vmax)

plt.show()