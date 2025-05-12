def extract_truth_tables_as_matrices(blif_file):
    tables = []

    with open(blif_file, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith('.names'):
            parts = line.split()
            inputs = parts[1:-1]
            output = parts[-1]
            truth_table = []

            i += 1
            while i < len(lines):
                logic_line = lines[i].strip()
                if not logic_line or logic_line.startswith('.'):
                    break
                # Convert each character to int
                row = list(map(int, logic_line.replace(" ", "")))
                truth_table.append(row)
                i += 1

            tables.append((output, truth_table))
        else:
            i += 1

    return tables


truth_tables=extract_truth_tables_as_matrices('test.txt')

# for gate_id, matrix in truth_tables:
#     print(f"Gate: {gate_id}")
#     for row in matrix:
#         print(row)
    
def and_check(truth_table):
    n_inputs= len(truth_table[1])-1
    if n_inputs <= 1:
        return False
    derniere_ligne= truth_table[pow(2,n_inputs)-1]
    cpt_zeros=0
    for line in truth_table:
        if line[-1]==0:
            cpt_zeros+=1
    if derniere_ligne[n_inputs]== 1  and cpt_zeros == (pow(2,n_inputs)-1):
        return True
    return False
    
def nand_check(truth_table):
    n_inputs= len(truth_table[1])-1
    if n_inputs <= 1:
        return False
    derniere_ligne= truth_table[pow(2,n_inputs)-1]
    cpt_ones=0
    for line in truth_table:
        if line[-1]==1:
            cpt_ones+=1
    if derniere_ligne[n_inputs]== 0  and cpt_ones == (pow(2,n_inputs)-1) :
        return True
    return False
 
def or_check(truth_table):
    n_inputs= len(truth_table[1])-1
    if n_inputs <= 1:
        return False
    premiere_ligne= truth_table[0]
    cpt_ones=0
    for line in truth_table:
        if line[-1]==1:
            cpt_ones+=1
    if premiere_ligne[n_inputs]== 0  and cpt_ones == (pow(2,n_inputs)-1):
        return True
    return False

def nor_check(truth_table):
    n_inputs= len(truth_table[1])-1
    if n_inputs <= 1:
        return False
    premiere_ligne= truth_table[0]
    cpt_zeros=0
    for line in truth_table:
        if line[-1]==0:
            cpt_zeros+=1
    if premiere_ligne[n_inputs]== 1  and cpt_zeros == (pow(2,n_inputs)-1):
        return True
    return False

def xor_check(truth_table):
    for line in truth_table:
        if(sum(line)%2!=0) :
            return False
    return True

def xnor_check(truth_table):
    n_inputs= len(truth_table[1])-1
    if n_inputs <= 1:
        return False
    for line in truth_table:
        if(sum(line)%2==0):
            return False
    return True

def not_check(truth_table):
    n_inputs= len(truth_table[1])-1
    return n_inputs==1

import networkx as nx
import matplotlib.pyplot as plt

def plot_directed_graph(sources, destinations):
    """
    Plots a directed graph from source and destination node lists.
    
    Parameters:
        sources (list): List of source node IDs.
        destinations (list): List of destination node IDs.
    """
    if len(sources) != len(destinations):
        raise ValueError("Sources and destinations must be of equal length.")
    
    G = nx.DiGraph()
    edges = list(zip(sources, destinations))
    G.add_edges_from(edges)
    
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', arrows=True)
    plt.title("Directed Graph")
    plt.show()


def check_gate(truth_table):
    if and_check(truth_table):
        return "AND"
    if or_check(truth_table):
        return "OR"
    if xor_check(truth_table):
        return "XOR"
    if not_check(truth_table):
        return "NOT"
    return 0