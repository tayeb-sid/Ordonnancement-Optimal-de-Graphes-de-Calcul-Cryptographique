# coding: utf-8
# clear;time python3 fonction_cout.17.tris.py -b 'Graphe(4).txt' -i 360184 -n 2 -c 1 -N 0.00157 -X 0.00564 -A 0.82265 -O 0.82866 -r 512 -g 612 -l 7818028 
################################################################################
import random
import operator
import time
import matplotlib.pyplot as plt
import sys
import math
import networkx as nx
import numpy as np
import multiprocessing as mp
import json
import os
from itertools import accumulate
import argparse
################################################################################
parser = argparse.ArgumentParser(description="Permets de trouver une distribution quasi optimal de la répartition des portes logiques d'un circuit cingulata BLIF sur un certain nombre de processeurs sous le modèle BSP. Le nombre de processeurs p = nodes * cores.",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-b','--blif',help="Chemin d'accès vers le fichier blif",required=True)
parser.add_argument('-i','--in-length',help="Taille maximale de l'entrée du circuit cingulata (en octets).",required=True)

parser.add_argument('-n','--nodes',help="Nombre de noeuds",required=True)
parser.add_argument('-c','--cores',help="Nombre de cores par noeud",required=True)

parser.add_argument('-N','--time-NOT',help="Durée d'execution d'une porte ['not'] (en seconde)",required=True)
parser.add_argument('-X','--time-XOR',help="Durée d'execution d'une porte ['xor','xnor'] (en seconde)",required=True)
parser.add_argument('-A','--time-AND',help="Durée d'execution d'une porte ['and','nand','andny','andyn'] (en seconde)",required=True)
parser.add_argument('-O','--time-OR',help="Durée d'execution d'une porte ['or','nor','orny','oryn'] (en seconde)",required=True)

parser.add_argument('-r','--cpu-speed',help="Paramètre BSP: Le nombre d'opérations en virgule flottante par seconde par processeurs (en Mflops)",required=True)
parser.add_argument('-g','--comm-speed',help="Paramètre BSP: Durée nécessaire pour échanger un octet sur le réseau (en flop/octet)",required=True)
parser.add_argument('-l','--latency',help="Paramètre BSP: Durée de la barrière de synchronisation (en flop).",required=True)

parser.add_argument('-S','--size-population',help="Paramètre génétique: La taille de la population.",default=100)
parser.add_argument('-G','--number-of-generation',help="Paramètre génétique: Le nombre de générations.",default=100)
parser.add_argument('-M','--chance-of-mutation',help="Paramètre génétique: La probabilité de mutation d'un gène.",default=0.05)
parser.add_argument('-B','--ratio-best-population',help="Paramètre génétique: Le ratio des individus à sélectionner parmi les meilleurs.",default=0.10)
parser.add_argument('-W','--ratio-lucky-population',help="Paramètre génétique: Le ratio des individus à sélectionner parmi les mauvais.",default=0.02)
args = parser.parse_args()
################################################################################
time_NOT=float(args.time_NOT)
time_XOR=float(args.time_XOR)
time_AND=float(args.time_AND)
time_OR =float(args.time_OR)
time_MOY=(time_NOT+time_XOR+time_AND+time_OR)/4
################################################################################
nodes=int(args.nodes)
cores=int(args.cores)
machines_number=nodes*cores
################################################################################
BSP_r=float(args.cpu_speed)
BSP_g=float(args.comm_speed)
BSP_L=float(args.latency)
BSP_rs=BSP_r/BSP_r
BSP_gs=BSP_g/(BSP_r*10**6)
BSP_Ls=BSP_L/(BSP_r*10**6)
################################################################################
blif1=os.path.abspath(args.blif)
in_length=float(args.in_length)
################################################################################
size_population       = int(args.size_population)
number_of_generation  = int(args.number_of_generation)
chance_of_mutation    = float(args.chance_of_mutation)
ratio_best_population = float(args.ratio_best_population)
ratio_lucky_population= float(args.ratio_lucky_population)
################################################################################
tt0=time.time()
################################################################################
name=os.path.basename(blif1)
fileout=name+'_'+str(nodes)+'x'+str(cores)+'_processor_'+str(int(time.time()))
################################################################################
args_dict = vars(args)
################################################################################
def poids(truth_table):
	# hypothese simplificatrice pour ter
	# le poids d'une porte depend de la taille de sa table de verité
	ref="00 1\n01 0\n10 0\n11 0"
	return(time_MOY*len(truth_table)/len(ref))
################################################################################
# Fonction qui calcul le niveau de chaque porte logique dans le circuit logique
################################################################################
def niveau(G):
	def in_degree(G):
		try:return G.in_degree().items()
		except:return G.in_degree()
	indegree_map = {v: d for v, d in in_degree(G) if d > 0}
	zero_indegree = [v for v, d in in_degree(G) if d == 0]
	while zero_indegree:
		yield zero_indegree
		new_zero_indegree = []
		for v in zero_indegree:
			for _, child in G.edges(v):
				indegree_map[child] -= 1
				if not indegree_map[child]:
					new_zero_indegree.append(child)
		zero_indegree = new_zero_indegree
################################################################################
# fonction permetant de lire le circuit de calcul depuis un fichier blif
################################################################################
def read_blif_file(file_name):
	def parse_blif(lines):
		taches={}
		cmds = "".join(lines).split('.')
		G = nx.DiGraph()
		for cmd in cmds:
			if cmd.startswith('names'):
				cmd = cmd.strip().split('\n')
				var = cmd[0].split()[1:]
				# print(var)
				out = var[-1]
				ins = var[:-1]
				edges = [(v, out) for v in ins]
				G.add_nodes_from(var)
				G.add_edges_from(edges)
				truth_table="\n".join(cmd[1:]).strip()
				taches[out]=(ins,truth_table)
		# print(G)
		return G,taches
	f = open(file_name)
	lines = f.readlines()
	f.close()
	return parse_blif(lines)
################################################################################
# cette fonction calcule le cout BSP d'une association l
################################################################################
def fitness(association):
	W={}
	H={}
	cost=0
	for level in levels:
		for gate in level:
			idd=gate_to_id[gate]
			pi=association[idd]
			poid=poids(taches[gate][-1])
			if pi in W:
				W[pi]+=poid
			else:
				W[pi]=poid
			for gate1 in G.successors(gate):
				idd1=gate_to_id[gate1]
				pi1=association[idd1]
				if (pi,pi1) in H:
					H[pi,pi1]+=in_length
				else:
					H[pi,pi1]=in_length
		# fin super etape
		# ne pas prendre en compte les comm dans la meme machine
		for p in range(machines_number):H[p,p]=0
		w=max(list(W.values())+[0])
		h=max(list(H.values())+[0])
		t=w+h*BSP_gs+BSP_Ls
		cost+=t
	return float(cost)
################################################################################
# generer une association aleatoir
################################################################################
def generate_coloration_aleatoire():
	association=[]
	for level in levels:
		for gate in level:
			pi=random.randint(0,machines_number-1)
			association.append(pi)
	return association
################################################################################
# variables
################################################################################
G,taches=read_blif_file(blif1)
levels=list(niveau(G))
levels[0]=[gate for gate in levels[0] if gate in taches.keys()]
################################################################################
# give an idd between 0 and n for all gates
gate_to_id={}
id_to_gate={}
idd=0
for level in levels:
	for gate in level:
		gate_to_id[gate]=idd
		id_to_gate[idd]=gate
		# print(idd,gate)
		idd+=1
################################################################################
c=generate_coloration_aleatoire()
print(fitness(c))
################################################################################
