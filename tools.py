import json
import os
import numpy as np
import pandas as pd

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
        "graph": params.get("blif", ""),  # Keeps full path "blif/Graphe(4346).txt"
        "n_cpus": int(params.get("nodes", 0)),
        "time_NOT": float(params.get("time_NOT", 0.0)),
        "time_XOR": float(params.get("time_XOR", 0.0)),
        "time_AND": float(params.get("time_AND", 0.0)),
        "time_OR": float(params.get("time_OR", 0.0)),
        "latency": int(params.get("latency", 0)),
        "comm_speed": int(params.get("comm_speed", 0)),
        "cpu_speed": int(params.get("cpu_speed", 0))
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
    graph_path = "dataset000/" + cluster_parameters['graph']
    print(graph_path)

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

    for node in degrees:
        node_features_json[node] = {
            "node_id": int(node),
            "fan_in": degrees[node]["fan_in"],
            "fan_out": degrees[node]["fan_out"],
            "depth": depths.get(node, 0),
            "n_cpus": cluster_parameters["n_cpus"],
            "cpu_speed": cluster_parameters["cpu_speed"]/100,
            "comm_speed": cluster_parameters["comm_speed"]/100,
            "latency": cluster_parameters["latency"] / 1_000_0000,
            "computation_time": mean_delay * degrees[node]["fan_in"]
        }

    os.makedirs(output_dir, exist_ok=True)

    graph_name = os.path.splitext(os.path.basename(cluster_parameters['graph']))[0]
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


clusters="dataset000/sol"
graph4="dataset000/blif/Graphe(4).txt"
test_file="test.txt"
testJSON_file="test.json"


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
            
            # Extract the "blif" field from the JSON (assuming the structure you provided)
            blif_field = data.get("parameters", {}).get("blif", "")
            if blif_field:
                # Extract the part of the blif file name
                blif_name = os.path.basename(blif_field)  # E.g., "Graphe(4346).txt"
                new_filename = f"optimal_{blif_name.replace('.txt', '.json')}"
                
                # If the filename already exists, add a suffix (_1, _2, etc.)
                suffix = 1
                while new_filename in seen_files:
                    new_filename = f"optimal_{blif_name.replace('.txt', f'_{suffix}.json')}"
                    suffix += 1
                
                # Rename the file
                os.rename(file_path, os.path.join(json_dir, new_filename))
                seen_files.add(new_filename)

# Specify the directory containing your JSON files
json_dir = "dataset000/sol"
# rename_json_files(json_dir)
