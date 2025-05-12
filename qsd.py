import json

# Load the JSON
with open("graphes_JSON_Complet\Graphe(1)_2_features.json", "r") as f:
    data = json.load(f)

# Get the graph key (e.g., "Graphe(1)")
graph_id = next(iter(data))
graph_data = data[graph_id]

# Construct graph path
graph_path = f"data000/Logique/{graph_id}.txt"

# Extract fitness parameters
fitness_params = graph_data["fitness_params"]
in_length = fitness_params["in_length"]
time_NOT = fitness_params["time_NOT"]
time_AND = fitness_params["time_AND"]
time_OR = fitness_params["time_OR"]
time_XOR = fitness_params["time_XOR"]
cpu_speed = fitness_params["cpu_speed"]
comm_speed = fitness_params["comm_speed"]
latency = fitness_params["latency"]
nodes = fitness_params["nodes"]
cores = fitness_params["cores"]


# Extract optimal repartition as a vector (drop keys, keep values)
optimal_vector = [v for _, v in graph_data["optimal_repartition"]]

# Extract first node features
first_node_key = next(iter(graph_data["node_features"]))
first_node_features = graph_data["node_features"][first_node_key]

# Print to verify
print("Graph path:", graph_path)
print("Optimal vector:", optimal_vector)
print("fitness params:", fitness_params)
