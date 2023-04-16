# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 15:24:58 2023

@author: Pandadada
"""

# Finding subgraph

import numpy as np
import os,sys

# sys.path.append(os.path.realpath('../lib'))
sys.path.append(os.path.realpath('E:/Master Thesis/FGWD_on_Graphs_basics/lib_0.0'))

from graph import graph_colors,draw_rel,draw_transp,Graph,wl_labeling
from ot_distances import Fused_Gromov_Wasserstein_distance,Wasserstein_distance
import copy
# from data_loader import load_local_data,histog,build_noisy_circular_graph
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
from matplotlib import cm
import networkx as nx
import ot


plt.close("all")

#%% Define two graphs 
G1=Graph()
G1.add_attributes({0:1,1:7,2:5,3:3})    # add color to nodes
G1.add_edge((0,1))
G1.add_edge((1,2))
G1.add_edge((2,3))
G1.add_edge((0,3))

G2=Graph()
G2.add_attributes({0:1,1:7})
G2.add_edge((0,1))

g1=G1.nx_graph
g2=G2.nx_graph

#%% Show the graphs
vmin=0
vmax=9  # the range of color
plt.figure(figsize=(8,5))
draw_rel(g1,vmin=vmin,vmax=vmax,with_labels=True,draw=False)
draw_rel(g2,vmin=vmin,vmax=vmax,with_labels=True,shiftx=3,draw=False)
plt.title('Two graphs. Color indicates the label')
plt.show()

#%% compare GWD and FGWD use and package and show the couplings

p1=ot.unif(4)
p2=ot.unif(2)

fea_metric = 'dirac'
# fea_metric = 'hamming'
# fea_metric = 'sqeuclidean'

thresh=0.004

# WD           
fig = plt.figure()
dw,transp_WD=Wasserstein_distance(features_metric=fea_metric).graph_d(G1,G2,p1,p2)
plt.title('WD coupling')
draw_transp(G1,G2,transp_WD,shiftx=2,shifty=0.5,thresh=thresh,swipy=True,swipx=False,with_labels=True,vmin=vmin,vmax=vmax)
plt.show()

# GWD
fig = plt.figure()
dgw,log_GWD,transp_GWD,M,C1,C2=Fused_Gromov_Wasserstein_distance(alpha=1,features_metric=fea_metric,method='shortest_path').graph_d(G1,G2,p1,p2)
plt.title('GWD coupling')
draw_transp(G1,G2,transp_GWD,shiftx=2,shifty=0.5,thresh=thresh,swipy=True,swipx=False,with_labels=True,vmin=vmin,vmax=vmax)
plt.show()

# FGWD
alpha=0.2
fig = plt.figure()
dfgw,log_FGWD,transp_FGWD,M,C1,C2=Fused_Gromov_Wasserstein_distance(alpha=alpha,features_metric=fea_metric,method='shortest_path').graph_d(G1,G2,p1,p2)
plt.title('FGWD coupling')
draw_transp(G1,G2,transp_FGWD,shiftx=2,shifty=0.5,thresh=thresh,swipy=True,swipx=False,with_labels=True,vmin=vmin,vmax=vmax)
plt.show()

#%%
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits import mplot3d

from FGW import cal_L,tensor_matrix,gwloss

A = np.linspace(0,1/3,100)
B = np.linspace(0,1/3,100)
C = np.linspace(0,1/3,100)

y = np.zeros([len(A),len(B)])
yy = np.zeros([len(A),len(B)])
y2 = np.zeros([len(A),len(B)])
# y_proj = np.zeros([len(A),len(B)])
# y_entropic = np.zeros([len(A),len(B)])

# C1=np.array([[0,1,1],[1,0,1],[1,1,0]])
# C2=np.array([[0,1],[1,0]])                               

L=cal_L(C1,C2)
                               
# for i in range(len(A)):
#     for j in range(len(B)):
#         a=A[i]
#         b=B[j]
#         T=np.array([[a,1/3-a],[b,1/3-b],[1/2-a-b,-1/6+a+b]])
#         y[j][i]=gwloss(L,T) # GWD
#         yy[j][i] = 2/3+1/2-2*( (1/2-a)*(1/3-a)+(1/6+a)*a+(1/2-b)*(1/3-b)+(1/6+b)*b+(a+b)*(-1/6+a+b)+(2/3-a-b)*(1/2-a-b) ) # formula by hand
#         y2[j][i]=(1-alpha)*np.sum(M * T) + alpha * y[j][i] # FGWD
                         
AA, BB, CC = np.meshgrid(A, B, C)
reg=0
for i in range(len(A)):
    for j in range(len(B)):
        for k in range(len(C)):
            a=AA[i,j,k]
            b=BB[i,j,k]
            c=CC[i,j,k]
            T=np.array([[a,1/4-a],[b,1/4-b],[c,1/4-c],[1/2-a-b-c,-1/4+a+b+c]])
            y[i][j]=gwloss(L,T) # GWD
            y2[i][j]=(1-alpha)*np.sum(M * T) + alpha * y[i][j] # FGWD
            # y_proj[i][j]=0 # projectiion points
            
            y[i][j]=y[i][j] + reg * np.sum(np.log(T)*T)
            y2[i][j]=y2[i][j] + reg * np.sum(np.log(T)*T)
        
#%% set all values outside condition to nan
# y[AA+BB > 1/2] = np.nan
# y[AA+BB < 1/6] = np.nan
# y2[AA+BB > 1/2] = np.nan
# y2[AA+BB < 1/6] = np.nan
# y_proj[AA+BB > 1/2] = np.nan
# y_proj[AA+BB < 1/6] = np.nan

#%% plot surface of GWD
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# surf = ax.plot_surface(AA,BB,y, cmap=cm.coolwarm,
#                         linewidth=0, antialiased=False)

# fig.colorbar(surf, shrink=0.5, aspect=8)

# # surf = ax.plot_surface(AA,BB,y_proj, color = '0.95',
# #                         linewidth=0, antialiased=False)

# plt.xlabel('a')
# plt.ylabel('b')

# plt.show()

#%% plot contour of GWD
fig = plt.figure()

y[np.isnan(y)] = 0 # set nan to zero

# levels = np.arange (-3/18, np.max(y), 1/18)
levels = np.arange (5/18, np.max(y), 1/18)

h = plt.contour(A, B, y, levels=levels, cmap=cm.coolwarm)
plt.clabel(h, inline=1, fontsize=10, colors='k')
plt.axis('scaled')
# plt.colorbar()
plt.xlabel('a')
plt.ylabel('b')

# # plot feasible set
# B1 = 1/6-A
# B2 = 1/2-A
# plt.plot(A,B1,A,B2, color = 'b', linewidth=1, linestyle="--")

# def plot_seg(point1,point2, color = 'b'):
#     x_values = [point1[0], point2[0]]
#     y_values = [point1[1], point2[1]]
#     plt.plot(x_values, y_values, color = color , linewidth=2, linestyle="--")
    
# plot_seg([0,1/6],[0,1/3])
# plot_seg([0,1/3],[1/6,1/3])
# plot_seg([1/6,0],[1/3,0])
# plot_seg([1/3,0],[1/3,1/6])

# # six notated points
# plt.annotate('A', xy=(0,1/6), xytext=(-0.02, 1/6), color = 'b', arrowprops=dict(facecolor='b', shrink=0.01))
# plt.annotate('B', xy=(0,1/3), xytext=(-0.02, 1/3), color = 'b', arrowprops=dict(facecolor='b', shrink=0.01))
# plt.annotate('C', xy=(1/6,1/3), xytext=(1/6, 1/3+0.02), color = 'b', arrowprops=dict(facecolor='b', shrink=0.01))
# plt.annotate('D', xy=(1/3,1/6), xytext=(1/3+0.02, 1/6), color = 'b', arrowprops=dict(facecolor='b', shrink=0.01))
# plt.annotate('E', xy=(1/3,0), xytext=(1/3+0.02, 0), color = 'b', arrowprops=dict(facecolor='b', shrink=0.01))
# plt.annotate('F', xy=(1/6,0), xytext=(1/6,-0.02), color = 'b', arrowprops=dict(facecolor='b', shrink=0.01))


#%% plot the optimization progress of GWD
T_log=log_GWD['G']

k=0
t=T_log[k]
pos_old=[t[0][0],t[1][0]]
plt.annotate('o', xy=pos_old, color = 'g')
while k<=len(T_log)-2:
    t=T_log[k+1]
    pos_new=[t[0][0],t[1][0]]
    plt.annotate('o', xy=pos_new, color = 'g')
    plot_seg(pos_old,pos_new, color = 'g')
    k+=1
    pos_old=pos_new
        
plt.show()

#%% plot surface of FGWD
fig = plt.figure()
ax = plt.axes(projection='3d')
AA, BB = np.meshgrid(A, B)
surf = ax.plot_surface(AA,BB,y2, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

fig.colorbar(surf, shrink=0.5, aspect=8)

# surf = ax.plot_surface(AA,BB,y_proj, color = '0.95',
#                         linewidth=0, antialiased=False)

#%% first order approximation 
# y3 = np.zeros([len(A),len(B)])
# for i in range(len(A)):
#     for j in range(len(B)):
#         a=AA[i][j]
#         b=BB[i][j]
#         T=np.array([[a,1/3-a],[b,1/3-b],[1/2-a-b,-1/6+a+b]])
#         Tk=np.outer(p1,p2)
#         grad=np.array([[0.5,1],[1,0.5],[1,1]])
#         y3[i][j]=gwloss(L,Tk)+np.sum( grad * (T-Tk) )
        
# surf = ax.plot_surface(AA,BB,y3, cmap=cm.coolwarm,
#                         linewidth=0, antialiased=False)

plt.xlabel('a')
plt.ylabel('b')
plt.show()

#%% plot contour of FGWD
fig = plt.figure()

y2[np.isnan(y2)] = 0 # set nan to zero

# levels = np.arange (-3/18, np.max(y2), 1/18)
levels = np.arange (5/18, np.max(y2), 1/18)

h = plt.contour(A, B, y2, levels=levels, cmap=cm.coolwarm)
plt.clabel(h, inline=1, fontsize=10, colors='k')
plt.axis('scaled')
# plt.colorbar()
plt.xlabel('a')
plt.ylabel('b')

# plot feasible set
B1 = 1/6-A
B2 = 1/2-A
plt.plot(A,B1,A,B2, color = 'b', linewidth=1, linestyle="--")

def plot_seg(point1,point2, color = 'b'):
    x_values = [point1[0], point2[0]]
    y_values = [point1[1], point2[1]]
    plt.plot(x_values, y_values, color = color , linewidth=2, linestyle="--")
    
plot_seg([0,1/6],[0,1/3])
plot_seg([0,1/3],[1/6,1/3])
plot_seg([1/6,0],[1/3,0])
plot_seg([1/3,0],[1/3,1/6])

# six notated points
plt.annotate('A', xy=(0,1/6), xytext=(-0.02, 1/6), color = 'b', arrowprops=dict(facecolor='b', shrink=0.01))
plt.annotate('B', xy=(0,1/3), xytext=(-0.02, 1/3), color = 'b', arrowprops=dict(facecolor='b', shrink=0.01))
plt.annotate('C', xy=(1/6,1/3), xytext=(1/6, 1/3+0.02), color = 'b', arrowprops=dict(facecolor='b', shrink=0.01))
plt.annotate('D', xy=(1/3,1/6), xytext=(1/3+0.02, 1/6), color = 'b', arrowprops=dict(facecolor='b', shrink=0.01))
plt.annotate('E', xy=(1/3,0), xytext=(1/3+0.02, 0), color = 'b', arrowprops=dict(facecolor='b', shrink=0.01))
plt.annotate('F', xy=(1/6,0), xytext=(1/6,-0.02), color = 'b', arrowprops=dict(facecolor='b', shrink=0.01))


#%% plot the optimization progress of GWD
T_log=log_FGWD['G']

k=0
t=T_log[k]
pos_old=[t[0][0],t[1][0]]
plt.annotate('o', xy=pos_old, color = 'g')
while k<=len(T_log)-2:
    t=T_log[k+1]
    pos_new=[t[0][0],t[1][0]]
    plt.annotate('o', xy=pos_new, color = 'g')
    plot_seg(pos_old,pos_new, color = 'g')
    k+=1
    pos_old=pos_new
        
plt.show()

