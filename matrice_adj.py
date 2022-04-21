#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 13:26:29 2022

@author: thgerault
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
from networkx.algorithms.community.centrality import girvan_newman

G = nx.Graph()
    

def matrice_proba(n):
    mat = abs(np.random.random((n, n))) #genere une matrice nxn de valeur aléatoire
    proba=mat*0
    diag=np.copy(proba)
    for i in range(n):
        proba[i,i:]=mat[i,i:]
        diag[i,i]=mat[i,i]

    proba=proba + proba.T
    proba=proba-2*diag
    
    return proba


def matrice_adj(n,pin):
    proba=matrice_proba(n)
    adj=np.eye(n)*0 #on part d'une matrice avec que des 0
    pout=(16-31*pin)/96 #deduction de pout
    
    # communauté 1
    for i in range(32):
        for j in range(i+1,32):
            if pin >= proba[i][j]:
                adj[i,j]=1
                adj[j,i]=1 #remplir la transposé
            else: 
                adj[i,j]=0
                adj[j,i]=0
    
    for i in range(32):
        for j in range(32,n):
            if pout > proba[i][j] and j!=i:
                    adj[i,j]=1
                    adj[j,i]=1
            else: 
                adj[i,j]=0
                adj[j,i]=0
    
    #communauté 2
    for i in range(32,64):
        for j in range(i+1,64):
            if pin >= proba[i][j]:
                adj[i,j]=1
                adj[j,i]=1
            else: 
                adj[i,j]=0
                adj[j,i]=0

    for i in range(32,64):
        for j in range(64,n):
            if pout > proba[i][j]:
                    adj[i,j]=1
                    adj[j,i]=1
            else: 
                adj[i,j]=0
                adj[j,i]=0
    
    #communauté 3
    for i in range(64,96):
        for j in range(i+1,96):
            if pin >= proba[i][j]:
                adj[i,j]=1
                adj[j,i]=1
            else: 
                adj[i,j]=0
                adj[j,i]=0

    for i in range(64,96):
        for j in range(96,n):
            if pout > proba[i][j]:
                    adj[i,j]=1
                    adj[j,i]=1
            else: 
                adj[i,j]=0
                adj[j,i]=0

    #communauté 4
    for i in range(96,n):
        for j in range(i+1,n):
            if pin >= proba[i][j]:
                adj[i,j]=1
                adj[j,i]=1
            else: 
                adj[i,j]=0
                adj[j,i]=0

    return adj   


#affiche le graphe et on distingue bien 4 communautés.

G = nx.from_numpy_matrix(matrice_adj(128, 16/31))
fix, ax = plt.subplots(1, 1,figsize=(4,4))
nx.draw(G, with_labels=True, ax=ax)

#verification que degre moyen = 16
t=[]
for i in range(1000):
    t.append(np.sum(matrice_adj(128, 0.4)[0]))
np.sum(t)/1000 # return 16.045
 


# estimation de pout pour voir a partir de quelle valeur de pout ca deconne
# fct degre renvoie tte les 

def degre(n,pin):
    matrice=matrice_adj(n, pin)
    pout=(16-31*pin)/96
    deg_out=0
    for i in range(n):
        if i<32:
            for j in range(32,n):
                if matrice[i,j]==1:
                    deg_out+=1
            if deg_out>=6:
                    return pout
            else: deg_out=0
        if i>=32 and i<64:
            for j in range(32):
                if matrice[i,j]==1:
                    deg_out+=1
            if deg_out>=6:
                    return pout
            else: deg_out=0
            for j in range(64,n):
                if matrice[i,j]==1:
                    deg_out+=1
            if deg_out>=6:
                    return pout
            else: deg_out=0
        
        if i>=64 and i<96:
            for j in range(64):
                if matrice[i,j]==1:
                    deg_out+=1
            if deg_out>=6:
                    return pout
            else: deg_out=0

            for j in range(96,n):
                if matrice[i,j]==1:
                    deg_out+=1
            if deg_out>=6:
                    return pout
            else: deg_out=0
        
        if i>=96:
            for j in range(96):
                if matrice[i,j]==1:
                    deg_out+=1
            if deg_out>=6:
                    return pout
            else: deg_out=0
        


def simulation(N):
    minimum=[]
    for i in range(N):
        l=[]
        simu=(16/31)*npr.rand(N)
        for k in simu:
            p=degre(128,k)
            if p != None:
                l.append(p)
        minimum.append(min(l))
    
    return min(minimum) 

simulation(100) # renvoie 0.008943845753291749 c'est long de ouf !!!


# test avec networkx la fct GN mais fait nimp, distingue que 2 communautés
"""
G = nx.from_numpy_matrix(matrice_adj(128, 0.51)) #attribution de la matrice à un graphe
communities = girvan_newman(G)

node_groups = []
for com in next(communities):
  node_groups.append(list(com))

print(node_groups)

color_map = []
for node in G:
    if node in node_groups[0]:
        color_map.append('blue')
    elif node in node_groups[1]: 
        color_map.append('green') 
    elif node in node_groups[2]:
        color_map.append('red')
    elif node in node_groups[3]:
        color_map.append('yellow')


fix, ax = plt.subplots(1, 1,figsize=(4,4))
nx.draw(G, node_color=color_map, with_labels=True, ax=ax)
plt.show()
"""

# http://www.xavierdupre.fr/app/ensae_teaching_cs/helpsphinx/notebooks/2021_random_graph.html#version-2          

#pin doit etre <=16/31 pour que pout>=0
#pout doit etre <=1/6 pour que pin >=0                 
    