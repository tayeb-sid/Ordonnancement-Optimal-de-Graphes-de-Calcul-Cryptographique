import json
import torch
import os
import numpy as np
import pandas as pd
from extra import extract_truth_tables_as_matrices,check_gate
import os
import json
import torch
import re
def blif_to_graph(blif_file):
    from_nodes = []
    to_nodes = []
    logic_gates = set()
    original_order = []

    # First pass: collect logic gates in order of appearance
    with open(blif_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if line.startswith('.names'):
                parts = line.split()
                if len(parts) < 2:
                    continue

                output_gate = parts[-1]
                if output_gate.isdigit():
                    gate_id = int(output_gate)
                    logic_gates.add(gate_id)
                    if gate_id not in original_order:
                        original_order.append(gate_id)

    # Build mapping from original ID to new sequential ID
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(original_order)}

    # Second pass: collect actual edges between logic gates
    with open(blif_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if line.startswith('.names'):
                parts = line.split()
                if len(parts) < 2:
                    continue

                inputs = parts[1:-1]
                output = parts[-1]

                if not output.isdigit():
                    continue

                output = int(output)

                if output not in logic_gates:
                    continue

                for input_gate in inputs:
                    if input_gate.isdigit():
                        input_id = int(input_gate)
                        if input_id in logic_gates:
                            from_nodes.append(input_id)
                            to_nodes.append(output)

    return from_nodes, to_nodes, id_mapping

def create_graph_structure_json(from_nodes, to_nodes):
    return {"from": from_nodes, "to": to_nodes}
def compute_gate_degrees(blif_path):
    """Calculate degrees only for gates that have fan_in > 0 (active logic gates)"""
    fan_in = {}
    fan_out = {}
    gates = set()
    primary_inputs = set()
    
    # First pass: Identify primary inputs
    with open(blif_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('.inputs'):
                primary_inputs.update(line.split()[1:])
    
    # Second pass: Analyze gates
    with open(blif_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('.names'):
                parts = line.split()
                if len(parts) < 3:
                    continue
                
                output = parts[-1]
                inputs = parts[1:-1]
                
                # Only process if output is a gate (not primary input)
                if output not in primary_inputs:
                    # Count only non-primary inputs
                    gate_inputs = [inp for inp in inputs if inp not in primary_inputs]
                    fan_in[output] = len(gate_inputs)
                    gates.add(output)
                    
                    # Track gate-to-gate connections
                    for inp in gate_inputs:
                        if inp not in fan_out:
                            fan_out[inp] = []
                        fan_out[inp].append(output)
                        gates.add(inp)
    
    # Filter for gates with fan_in > 0
    active_gates = {
        gate: {
            'fan_in': fan_in[gate],
            'fan_out': len(fan_out.get(gate, [])),
            'total': fan_in[gate] + len(fan_out.get(gate, []))
        }
        for gate in gates 
        if fan_in.get(gate, 0) > 0  # THIS IS THE KEY FILTER
    }
    
    return active_gates
def print_gate_degrees(degrees):
    print("\nGate Connectivity Analysis:")
    print("-" * 40)
    print(f"{'Gate':<10} {'Fan-in':<10} {'Fan-out':<10} {'Total':<10}")
    print("-" * 40)
    
    for gate, data in degrees.items():
        print(f"{gate:<10} {data['fan_in']:<10} {data['fan_out']:<10} {data['total']:<10}")
    
    print("-" * 40)
    total_gates = len(degrees)
    avg_fan_in = sum(d['fan_in'] for d in degrees.values()) / total_gates
    print(f"Total gates: {total_gates}, Avg fan-in: {avg_fan_in:.2f}")
def compute_gate_depths(blif_path):
    """Calculate depths for gates only, excluding primary inputs and depth=1 gates"""
    # First pass: Build the circuit graph
    graph = {}
    primary_inputs = set()
    gates = set()
    
    with open(blif_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('.inputs'):
                primary_inputs.update(line.split()[1:])
            elif line.startswith('.names'):
                parts = line.split()
                if len(parts) < 3:
                    continue
                output = parts[-1]
                inputs = parts[1:-1]
                if output not in primary_inputs:
                    # Only track gate-to-gate connections
                    gate_inputs = [inp for inp in inputs if inp not in primary_inputs]
                    graph[output] = gate_inputs
                    gates.add(output)
                    gates.update(gate_inputs)
    
    # Remove primary inputs
    gates -= primary_inputs
    
    # Compute raw depths
    depths = {}
    def calculate_depth(node):
        if node in depths:
            return depths[node]
        if node in primary_inputs:
            return 0
        if node not in graph or not graph[node]:
            return 1  # Will be filtered later
        
        max_depth = max(calculate_depth(pred) for pred in graph[node])
        depths[node] = max_depth + 1
        return depths[node]
    
    for gate in gates:
        calculate_depth(gate)
    
    # Filter for depths ≥ 2
    return {gate: depth for gate, depth in depths.items() if depth >= 2}
def print_gate_depths(depths):
    print("\nGate Depth Analysis:")
    print("-" * 40)
    print(f"{'Gate':<10} {'Depth':<10}")
    print("-" * 40)
    
    for gate, depth in sorted(depths.items(), key=lambda x: x[1]):
        print(f"{gate:<10} {depth:<10}")
    
    print("-" * 40)
    max_depth = max(depths.values()) if depths else 0
    print(f"Total gates: {len(depths)}, Max depth: {max_depth}")

def extract_cluster_parameters(json_file_path):
    with open(json_file_path) as f:
        data = json.load(f)
    
    params = data.get("parameters", {})
    
    cluster_params = {
        "graph": params.get("blif", ""),  
        "time_NOT": float(params.get("time_NOT", 0.0)),
        "time_XOR": float(params.get("time_XOR", 0.0)),
        "time_AND": float(params.get("time_AND", 0.0)),
        "time_OR": float(params.get("time_OR", 0.0)),
        "latency": int(params.get("latency", 0)),
        "comm_speed": int(params.get("comm_speed", 0)),
        "cpu_speed": int(params.get("cpu_speed", 0)),
        "n_cpus": int(params.get("nodes", 0)),
        "in_length":int(params.get("in_length",0))
    }
    
    return cluster_params

def extract_target_repartition(json_file_path):
    with open(json_file_path) as f:
        data = json.load(f)
    
    optimal_repartition = data.get("solution", {})
    return optimal_repartition

def create_node_features_JSON(cluster_path, output_dir="graphes_JSON"):
    target_repartition = extract_target_repartition(cluster_path)
    cluster_parameters = extract_cluster_parameters(cluster_path)
    fitness_param=load_fitness_parameters(cluster_path)
    graph_path = "dataset000/" + cluster_parameters['graph']
    print(graph_path)
    truth_tables=extract_truth_tables_as_matrices(graph_path)
    from_nodes, to_nodes,id_mapping = blif_to_graph(graph_path)
    graph_structure = create_graph_structure_json(from_nodes, to_nodes)

    node_features_json = {}
    depths = compute_gate_depths(graph_path)
    degrees = compute_gate_degrees(graph_path)

    mean_delay = np.mean([
        cluster_parameters['time_NOT'],
        cluster_parameters['time_AND'],
        cluster_parameters['time_OR'],
        cluster_parameters['time_XOR']
    ])

    
    os.makedirs(output_dir, exist_ok=True)

    graph_name = os.path.splitext(os.path.basename(cluster_parameters['graph']))[0]
    
    graph_blif_name = os.path.basename(cluster_parameters['graph'])  # e.g., Graphe(1).blif
    match_graph = re.search(r'Graphe\((\d+)\)', graph_blif_name, re.IGNORECASE)
    graph_number = int(match_graph.group(1)) if match_graph else -1

    for node in degrees:
        filtered_lines = [line for line in truth_tables if line[0] == node]
        truth_table=filtered_lines[0][1]
        node_weight=len(truth_table)
        type="uknown"
        if check_gate(truth_table)=='AND':
            mean_delay=cluster_parameters['time_AND']
            type="AND"
        elif check_gate(truth_table)=='OR':
            type="OR"
            mean_delay=cluster_parameters['time_OR']

        elif check_gate(truth_table)=='XOR':
            mean_delay=cluster_parameters['time_XOR']
            type="XOR"

        elif check_gate(truth_table)=='NOT':
            mean_delay=cluster_parameters['time_NOT']
            type="NOT"

        node_features_json[node] = {
            "node_id": int(node),
            "fan_in": degrees[node]["fan_in"],
            "fan_out": degrees[node]["fan_out"],
            "depth": depths.get(node, 0),
            "weight": node_weight,
            "computation_time": mean_delay * degrees[node]["fan_in"] * node_weight,
            "cpu_speed": cluster_parameters["cpu_speed"],
            "comm_speed": cluster_parameters["comm_speed"],
            "latency": cluster_parameters["latency"],
            "in_length":cluster_parameters["in_length"],
            "time_NOT":        cluster_parameters['time_NOT'],
            "time_AND":cluster_parameters['time_AND'],
            "time_OR": cluster_parameters['time_OR'],
            "time_XOR":cluster_parameters['time_XOR'],
            "n_cpus": cluster_parameters["n_cpus"],
            "graphe_number":graph_number
        }
       
  

    base_filename = f"{graph_name}_features.json"
    output_path = os.path.join(output_dir, base_filename)

    # Check for name collision and create a new filename if needed
    if os.path.exists(output_path):
        counter = 2
        while True:
            new_filename = f"{graph_name}_{counter}_features.json"
            new_path = os.path.join(output_dir, new_filename)
            if not os.path.exists(new_path):
                output_path = new_path
                break
            counter += 1

    final_json = {
        graph_name: {
            "graph_structure": graph_structure,
            "node_features": node_features_json,
            "optimal_repartition": target_repartition,
            "fitness_params": fitness_param,
            "id_mapping":id_mapping
        }
    }

    with open(output_path, 'w') as f:
        json.dump(final_json, f, indent=4)

    print(f"Node features exported to: {output_path}")
    return final_json

def process_all_clusters(cluster_dir, output_dir="graphes_JSON"):
    cpt=0
    success=0
    errors=0
    for filename in os.listdir(cluster_dir):
        cpt+=1
        if filename.endswith(".json"): 
            cluster_path = os.path.join(cluster_dir, filename)
            try:
                print(f"Processing {filename}...")
                create_node_features_JSON(cluster_path, output_dir)
                success+=1
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                errors+=1
    print("processed: ",cpt," successful: ",success, "errors: ",errors)

def extract_mapped_edges_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    graph_name = next(iter(data))
    graph_data = data[graph_name]

    graph_structure = graph_data["graph_structure"]  # Dict: { "from": [...], "to": [...] }
    mapping = graph_data["id_mapping"]  # Dict: original_id (str) -> mapped_id (int)

    # Prepare lists for 'from', 'to', 'original_from', and 'original_to'
    from_nodes = []
    to_nodes = []
    original_from = []
    original_to = []

    for original_src, original_dst in zip(graph_structure["from"], graph_structure["to"]):
        mapped_src = mapping[str(original_src)]  # Map the original source node to the new ID
        mapped_dst = mapping[str(original_dst)]  # Map the original destination node to the new ID

        from_nodes.append(mapped_src)
        to_nodes.append(mapped_dst)
        original_from.append(original_src)  # Store the original source node
        original_to.append(original_dst)  # Store the original destination node

    # Create a pandas DataFrame with 'from', 'to', 'original_from', and 'original_to' columns
    edge_df = pd.DataFrame({
        'from': from_nodes,
        'to': to_nodes,
        'original_from': original_from,
        'original_to': original_to
    })

    # Sort the DataFrame by the 'from' column
    edge_df = edge_df.sort_values(by='from').reset_index(drop=True)

    return edge_df

def extract_node_features_from_json_file(json_file):
    # Load the JSON data from the file
    with open(json_file, 'r') as f:
        graph_data = json.load(f)
    
    # Get the first key of the graph_data (this will be the graph name)
    graph_name = next(iter(graph_data))  # Extract the first key (graph name)
    
    # Extract the "node_features" dictionary from the graph_data
    node_features = graph_data[graph_name]["node_features"]
    
    # Extract the "id_mapping" dictionary to use for sorting
    id_mapping = graph_data[graph_name].get("id_mapping", {})
    
    # Prepare a list of node features for the DataFrame
    node_data = []
    for node_id, features in node_features.items():
        # Add the node_id and its associated features to the node_data list
        features["node_id"] = node_id  # Append node_id as a column
        node_data.append(features)
    
    # Convert the list of node data into a pandas DataFrame
    df = pd.DataFrame(node_data)
    
    # Sort the DataFrame based on the "id_mapping" of node_ids (the new mapped order)
    if id_mapping:
        # Ensure that id_mapping is in a list of (old_id, new_id) format
        # Here, we assume id_mapping is a dictionary where key is the old node_id and value is the new node_id.
        sorted_node_ids = sorted(id_mapping, key=lambda x: id_mapping[x])
        
        # Reorder the rows of the DataFrame based on the sorted mapping
        df.set_index("node_id", inplace=True)
        df = df.loc[sorted_node_ids]
    
    return df

def extract_optimal_repartition_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    graph_name = next(iter(data))  # Extract graph name (assuming there's only one graph in the file)
    graph_data = data[graph_name]

    # Extract "optimal_repartition" and "id_mapping" from the JSON
    optimal_repartition = graph_data["optimal_repartition"]  # List of [original_id, cpu]
    mapping = graph_data["id_mapping"]  # Dict: original_id (str) -> mapped_id (int)

    # Prepare lists for the original ID, assigned CPU, and mapped ID
    original_ids = []
    assigned_cpus = []
    mapped_ids = []

    # Iterate over the "optimal_repartition" list
    for item in optimal_repartition:
        original_id = item[0]  # The original ID is the first element in each sublist
        cpu = item[1]  # The CPU is the second element in each sublist

        # Get the mapped ID from the id_mapping
        mapped_id = mapping[str(original_id)]  # Use str to match the keys in id_mapping

        original_ids.append(original_id)
        assigned_cpus.append(cpu)
        mapped_ids.append(mapped_id)

    # Create a pandas DataFrame with 'original_id', 'assigned_cpu', and 'mapped_id' columns
    repartition_df = pd.DataFrame({
        'original_id': original_ids,
        'assigned_cpu': assigned_cpus,
        'mapped_id': mapped_ids
    })

    # Sort the DataFrame by 'mapped_id' to ensure it's ordered by the mapped node IDs
    repartition_df = repartition_df.sort_values(by='mapped_id').reset_index(drop=True)

    return repartition_df

def rename_json_files(json_dir):
    # Create a set to keep track of filenames to avoid duplicates
    seen_files = set()

    # Iterate through all JSON files in the directory
    for filename in os.listdir(json_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(json_dir, filename)
            
            # Open and load the JSON file
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            blif_field = data.get("parameters", {}).get("blif", "")
            if blif_field:
                # Extract the part of the blif file name
                blif_name = os.path.basename(blif_field) 
                new_filename = f"optimal_{blif_name.replace('.txt', '.json')}"
                
                # If the filename already exists, add a suffix (_1, _2, etc.)
                suffix = 1
                while new_filename in seen_files:
                    new_filename = f"optimal_{blif_name.replace('.txt', f'_{suffix}.json')}"
                    suffix += 1
                
                # Rename the file
                os.rename(file_path, os.path.join(json_dir, new_filename))
                seen_files.add(new_filename)


def prepare_data_for_GNN(node_features_df, edges_df, target_df):
    from_nodes = edges_df['from'].values
    to_nodes = edges_df['to'].values

    
    node_features = node_features_df.values

    node_features_tensor = torch.tensor(node_features, dtype=torch.float32)

    y_target = target_df['assigned_cpu'].values
    y_target_tensor = torch.tensor(y_target, dtype=torch.long)

   
    edge_index = np.array([from_nodes, to_nodes], dtype=np.int64)
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)

    return node_features_tensor, edge_index_tensor, y_target_tensor


def load_fitness_parameters(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    p = data["parameters"]

    # Convert parameters to correct types
    params = {
        "blif": p["blif"],
        "in_length": int(p["in_length"]),
        "nodes": int(p["nodes"]),
        "cores": int(p["cores"]),
        "time_NOT": float(p["time_NOT"]),
        "time_XOR": float(p["time_XOR"]),
        "time_AND": float(p["time_AND"]),
        "time_OR": float(p["time_OR"]),
        "cpu_speed": float(p["cpu_speed"]),
        "comm_speed": float(p["comm_speed"]),
        "latency": float(p["latency"])
    }
    return params

import os
import json
import torch

def add_prediction_to_json(predictions, file_name, dir_path=""):
    # Build full path
    json_path = os.path.join(dir_path, file_name) if dir_path else file_name

    # Extract graph key: first word before '_'
    graph_key = file_name.split("_")[0]

    # Load JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    # Convert predictions to list if it's a tensor
    pred_list = predictions.tolist() if isinstance(predictions, torch.Tensor) else list(predictions)

    # Inject predictions
    data[graph_key]["prediction_list"] = pred_list

    # Save file
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)

import os
import json
import shutil

def organize_by_n_cpus(source_dir):
    for file_name in os.listdir(source_dir):
        if not file_name.endswith(".json"):
            continue

        file_path = os.path.join(source_dir, file_name)

        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            # Extract the first graph key
            graph_key = next(iter(data))
            node_features = data[graph_key]["node_features"]

            # Extract n_cpus from the first node
            first_node_key = next(iter(node_features))
            n_cpus = node_features[first_node_key]["n_cpus"]

            # Make destination directory
            dest_dir = os.path.join(source_dir, str(n_cpus))
            os.makedirs(dest_dir, exist_ok=True)

            # Move the file
            shutil.move(file_path, os.path.join(dest_dir, file_name))

        except Exception as e:
            print(f"Skipping {file_name}: {e}")

import os
import random
import shutil

def split_train_test(root_dir, train_ratio=0.8):
    for n_cpu_folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, n_cpu_folder)
        if not os.path.isdir(folder_path):
            continue

        # List only JSON files
        files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
        if not files:
            continue

        # random.shuffle(files)
        split_idx = int(len(files) * train_ratio)
        train_files = files[:split_idx]
        test_files = files[split_idx:]

        # Create subfolders
        train_path = os.path.join(folder_path, "train")
        test_path = os.path.join(folder_path, "test")
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        # Move files
        for f in train_files:
            shutil.move(os.path.join(folder_path, f), os.path.join(train_path, f))
        for f in test_files:
            shutil.move(os.path.join(folder_path, f), os.path.join(test_path, f))


import os
from torch_geometric.data import Data

def load_graphs_by_ncpu(n_cpu: int, base_dir="graphes_JSON_Complet"):
    cpu_dir = os.path.join(base_dir, str(n_cpu))
    train_dir = os.path.join(cpu_dir, "train")
    test_dir = os.path.join(cpu_dir, "test")

    def load_graphs_from_dir(directory):
        if not os.path.exists(directory):
            print(f"Warning: Directory does not exist: {directory}")
            return []
        files = [f for f in os.listdir(directory) if f.endswith('.json')]
        data_list = []
        for file in files:
            path = os.path.join(directory, file)
            node_features_df = extract_node_features_from_json_file(path)
            edges_df = extract_mapped_edges_from_json(path)
            target_df = extract_optimal_repartition_from_json(path)
            node_features_tensor, edge_index_tensor, y_target_tensor = prepare_data_for_GNN(
                node_features_df, edges_df, target_df
            )
            data = Data(x=node_features_tensor, edge_index=edge_index_tensor, y=y_target_tensor)
            data_list.append(data)
        return data_list

    train_graphs = load_graphs_from_dir(train_dir)
    test_graphs = load_graphs_from_dir(test_dir)

    print(f"Loaded {len(train_graphs)} training and {len(test_graphs)} testing graphs for n_cpus={n_cpu}")
    return train_graphs, test_graphs



clusters="dataset000/sol"
# graph4="dataset000/blif/Graphe(4).txt"
# test_file="test.txt"
# testJSON_file="test.json"
# json_dir = "dataset000/sol"

# json_dirComplet = "datasetComplet/dataset001/sol"
# rename_json_files(json_dir)



# graph4="dataset000/Logique/Graphe(1).txt"
# truth_tables= extract_truth_tables_as_matrices(graph4)
# for gate_id, matrix in truth_tables:
#     print(f"Gate: {gate_id} weight {len(matrix)}")
#     print(check_gate(matrix))
    # for row in matrix:
    #     print(row)
import os
import json
import re

def add_suffix_to_each_node_feature(folder_path):
    for file_name in os.listdir(folder_path):
        if not file_name.endswith('_features.json'):
            continue

        # Try to extract the suffix
        match = re.search(r'_(\d+)_features\.json$', file_name)
        suffix = int(match.group(1)) if match else -1

        file_path = os.path.join(folder_path, file_name)

        with open(file_path, 'r') as f:
            data = json.load(f)

        # Assuming the structure is: { "Graphe(n)": { ... } }
        graph_key = list(data.keys())[0]
        node_features = data[graph_key].get("node_features", {})

        for node_id, node_data in node_features.items():
            if isinstance(node_data, dict):  # safety check
                node_data["graph_suffix"] = suffix

        # Save back the updated JSON
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

    print("Suffixes added to all node features.")


# create_node_features_JSON("dataset000\sol\optimal_Graphe(1)_1.json","qsd")
# add_suffix_to_each_node_feature('qsd')

# process_all_clusters(clusters,"graphes_JSON_Complet")
# add_suffix_to_each_node_feature('graphes_JSON_Complet')
# organize_by_n_cpus('graphes_JSON_Complet')
# split_train_test('graphes_JSON_Complet')