# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 20:41:53 2023

@author: Pandadada
"""

import numpy as np
import os,sys
sys.path.append(os.path.realpath('E:/Master Thesis/FGWD_on_Graphs_basics/lib_2.0'))
from graph import graph_colors,draw_rel,draw_transp,Graph,wl_labeling
from ot_distances import Fused_Gromov_Wasserstein_distance,Wasserstein_distance
import copy
# from data_loader import load_local_data,histog,build_noisy_circular_graph
import matplotlib.pyplot as plt
import networkx as nx

#%% Example 1
g1=Graph()
g1.add_attributes({0:1,1:7,2:1,3:7})
g1.add_edge((0,1))
g1.add_edge((1,2))
g1.add_edge((3,0))
g1.add_edge((3,2))
g1.add_edge((0,2))
g2=Graph()
g2.add_attributes({0:7,1:1,2:7,3:1})
g2.add_edge((0,1))
g2.add_edge((1,2))
g2.add_edge((3,0))
g2.add_edge((3,2))
g2.add_edge((2,0))

plt.figure(figsize=(5,4))
vmin=0
vmax=7
draw_rel(g1.nx_graph,draw=False,vmin=vmin,vmax=vmax,with_labels=True)
draw_rel(g2.nx_graph,draw=False,vmin=vmin,vmax=vmax,with_labels=True,shiftx=5,swipy=True)
plt.title('Two graphs. Color indicates the label')
plt.show()

#%% Example 2
# g1=Graph()
# g1.add_attributes({0:1,1:3,2:5,3:7})
# g1.add_edge((0,1))
# g1.add_edge((1,2))
# g1.add_edge((3,0))
# g1.add_edge((3,2))
# g1.add_edge((0,2))
# g2=Graph()
# g2.add_attributes({0:1,1:3,2:5})
# g2.add_edge((0,1))
# g2.add_edge((1,2))
# g2.add_edge((2,0))

# plt.figure(figsize=(5,4))
# vmin=0
# vmax=7
# draw_rel(g1.nx_graph,draw=False,vmin=vmin,vmax=vmax,with_labels=True)
# draw_rel(g2.nx_graph,draw=False,vmin=vmin,vmax=vmax,with_labels=True,shiftx=5,swipy=True)
# plt.title('Two graphs. Color indicates the label')
# plt.show()

#%%
nodes1=g1.nodes()
nodes2=g2.nodes()
p1=np.ones(len(nodes1))/len(nodes1)
p2=np.ones(len(nodes2))/len(nodes2)

# fea_metric = 'dirac'
# fea_metric = 'hamming'
fea_metric = 'sqeuclidean'

#%%
# FGWD
alpha=0.5
dfgw,log_FGWD,transp_FGWD,M,C1,C2=Fused_Gromov_Wasserstein_distance(alpha=alpha,features_metric=fea_metric,method='shortest_path').graph_d(g1,g2,p1,p2)
# WD
dw,transp_WD=Wasserstein_distance(features_metric=fea_metric).graph_d(g1,g2,p1,p2)
# GWD
dgw,log_GWD,transp_GWD,M,C1,C2=Fused_Gromov_Wasserstein_distance(alpha=1,features_metric=fea_metric,method='shortest_path').graph_d(g1,g2,p1,p2)
fig=plt.figure(figsize=(10,8))
plt.title('GWD coupling')
thresh=0.004
draw_transp(g1,g2,transp_GWD,shiftx=2,shifty=0.5,thresh=thresh,
            swipy=True,swipx=False,with_labels=True,vmin=vmin,vmax=vmax)
plt.show()

print('Wasserstein distance={}, Gromov distance={} \nFused Gromov-Wasserstein distance for alpha {} = {}'.format(dw,dgw,alpha,dfgw))

# matching when alpha = 0.5 
fig=plt.figure(figsize=(10,8))
thresh=0.004
plt.title('FGW coupling, alpha = 0.5')
draw_transp(g1,g2,transp_FGWD,shiftx=2,shifty=0.5,thresh=thresh,
            swipy=True,swipx=False,with_labels=True,vmin=vmin,vmax=vmax)
plt.show()

#%% FGWD, find alpha
alld=[]
x=np.linspace(0,1,100)
for alpha in x:
    d,log,transp,M,C1,C2=Fused_Gromov_Wasserstein_distance(alpha=alpha,features_metric=fea_metric).graph_d(g1,g2,p1,p2)
    alld.append(d)
plt.plot(x,alld)
plt.title('Evolution of FGW dist in wrt alpha \n max={}'.format(x[np.argmax(alld)]))
plt.xlabel('Alpha')
plt.xlabel('FGW dist')
plt.show()

# optimal matching
fig=plt.figure(figsize=(10,8))
thresh=0.004
alpha_opt=x [ alld.index(max(alld)) ]
dfgw_opt,log_FGWD_opt,transp_FGWD_opt,M,C1,C2=Fused_Gromov_Wasserstein_distance(alpha=alpha_opt,features_metric=fea_metric).graph_d(g1,g2,p1,p2)
# d=dfgw.graph_d(g1,g2)
# plt.title('FGW coupling, dist : '+str(np.round(dfgw,3)),fontsize=15)
plt.title('FGW coupling, alpha = opt')
draw_transp(g1,g2,transp_FGWD_opt,shiftx=2,shifty=0.5,thresh=thresh,
            swipy=True,swipx=False,with_labels=True,vmin=vmin,vmax=vmax)
plt.show()

#%% Entropic GWD
# from entropic import entropic_gromov_wasserstein

# eps=0.01
# G0=np.outer(p1,p2)
# log_GWD_ent, trans_GWD_ent = entropic_gromov_wasserstein(C1, C2, p1, p2, loss_fun='square_loss', epsilon=eps, symmetric=None, G0=G0,
#                                 max_iter=1000, tol=1e-9, verbose=False, log=True)
# fig=plt.figure(figsize=(10,8))
# plt.title('entropic GWD')
# draw_transp(g1,g2,trans_GWD_ent,shiftx=2,shifty=0.5,thresh=thresh,
#             swipy=True,swipx=False,with_labels=True,vmin=vmin,vmax=vmax)
# plt.show()
#%% POT entropic
import ot
eps=0.1
G0=np.outer(p1,p2)
T, log = ot.gromov.entropic_gromov_wasserstein(C1, C2, p1, p2, loss_fun='square_loss', epsilon=eps, symmetric=None, G0=G0,
                                max_iter=1000, tol=1e-9, verbose=False, log=True)
fig=plt.figure(figsize=(10,8))
plt.title('entropic GWD')
draw_transp(g1,g2,T,shiftx=2,shifty=0.5,thresh=thresh,
            swipy=True,swipx=False,with_labels=True,vmin=vmin,vmax=vmax)
plt.show()

dgw_ent_disc, log = ot.gromov.entropic_gromov_wasserstein2(C1, C2, p1, p2, loss_fun='square_loss', epsilon=eps, symmetric=None, G0=G0,
                                max_iter=1000, tol=1e-9, verbose=False, log=True)   # entropic gromov-wasserstein discrepancy
fig=plt.figure(figsize=(10,8))
plt.title('entropic GWD disc')
draw_transp(g1,g2,log['T'],shiftx=2,shifty=0.5,thresh=thresh,
            swipy=True,swipx=False,with_labels=True,vmin=vmin,vmax=vmax)
plt.show()

#%% POT normal
# import ot
# G0=np.outer(p1,p2)
# T1, log1 = ot.gromov.gromov_wasserstein(C1, C2, p1, p2, loss_fun='square_loss', symmetric=None, G0=G0,
#                                 max_iter=1000, tol=1e-9, verbose=False, log=True)
# fig=plt.figure(figsize=(10,8))
# plt.title('GWD')
# draw_transp(g1,g2,T1,shiftx=2,shifty=0.5,thresh=thresh,
#             swipy=True,swipx=False,with_labels=True,vmin=vmin,vmax=vmax)
# plt.show()
