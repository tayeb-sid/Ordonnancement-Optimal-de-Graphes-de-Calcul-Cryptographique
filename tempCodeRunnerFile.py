from tools import * 
import torch

graph4JSON="graphes_JSON/Graphe(4)_features.json"
node_features_df = extract_mapped_edges_from_json(graph4JSON)
edges_df = extract_node_features_from_json_file(graph4JSON)
target_df = extract_target_repartition(graph4JSON)
print("hello")
def prepare_data_for_GNN(node_features_df, edges_df, target_df):
    from_nodes = edges_df['from'].values
    to_nodes = edges_df['to'].values

    node_features = node_features_df.drop(columns=["node_id"]).values
    node_features_tensor = torch.tensor(node_features, dtype=torch.float32)

    y_target = target_df['assigned_cpu'].values
    y_target_tensor = torch.tensor(y_target, dtype=torch.float32)

    edge_index = torch.tensor([from_nodes, to_nodes], dtype=torch.long)

    return node_features_tensor, edge_index, y_target_tensor

node_features_tensor, edge_index, y_target_tensor = prepare_data_for_GNN(node_features_df, edges_df, target_df)

# print(node_features_tensor.shape)
# print(edge_index.shape)
# print(y_target_tensor.shape)
print(edges_df.columns)