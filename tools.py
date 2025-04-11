import json
import math
import os
import numpy as np
def blif_to_graph(blif_file):
    sources = []
    targets = []
    logic_gates = set()  # Track all gate outputs (as ints)

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
                    logic_gates.add(int(output_gate))  # Convert to int

    # Second pass: Only keep edges between logic gates
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

                for input_gate in inputs:
                    if input_gate.isdigit() and int(input_gate) in logic_gates:
                        sources.append(int(input_gate))
                        targets.append(output)

    return sources, targets
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
        "latency": int(params.get("latency", 0))
    }
    
    return cluster_params

def extract_target_repartition(json_file_path):
    with open(json_file_path) as f:
        data = json.load(f)
    
    optimal_repartition = data.get("solution", {})
    return optimal_repartition

import numpy as np

def create_node_features_JSON(cluster_path, output_dir="graphes_JSON"):
    target_repartition = extract_target_repartition(cluster_path)
    cluster_parameters = extract_cluster_parameters(cluster_path)
    graph_path = "dataset000/" + cluster_parameters['graph']
    print(graph_path)

    from_nodes, to_nodes = blif_to_graph(graph_path)
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
            "optimal_repartition": target_repartition
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


clusters="dataset000/sol"
graph4="dataset000/blif/Graphe(4).txt"
test_file="test.txt"
