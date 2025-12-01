import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import os
import argparse

random.seed(42)
np.random.seed(42)

def euclidean_distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))

def initialize_weights(n_neurons, input_len):
    weights = np.ones((n_neurons, input_len)) / np.sqrt(input_len)
    return weights

def find_bmu_1(weights, input_vector):
    distances = [euclidean_distance(input_vector, weight) for weight in weights]
    bmu_index = np.argmin(distances)
    return bmu_index

def find_bmu_2(weights, input_vector, clusters):
    distances = [euclidean_distance(input_vector, weight[:len(input_vector)]) for weight in weights]
    bmu_index = np.argmin(distances)
    cluster_index = clusters[bmu_index]
    return bmu_index, cluster_index

def update_weight_convex(weight, input_vector, beta):
    return (1 - beta) * weight + beta * input_vector

def cluster_fragments(weights, fragments):
    clusters = {i: [] for i in range(len(weights))}
    for fragment in fragments:
        bmu_index = find_bmu_1(weights, fragment)
        clusters[bmu_index].append(fragment)
    return {k: v for k, v in clusters.items() if len(v) >= 4}

def train_kohonen_fragments_sync(fragments, n_neurons, input_len, initial_learning_rate=0.1, decay_rate=0.95,
                                 iterations=1000):
    weights = initialize_weights(n_neurons, input_len)
    learning_rate = initial_learning_rate
    beta = 1.0

    for iteration in range(iterations):
        prev_weights = np.copy(weights)
        for input_vector in fragments:
            bmu_index = find_bmu_1(weights, input_vector)
            weights[bmu_index] = update_weight_convex(weights[bmu_index], input_vector, beta)

        learning_rate *= decay_rate
        beta = max(0, beta - (1 / iterations))
        weight_change = np.mean(np.abs(weights - prev_weights))
        if iteration % 100 == 0 or iteration == iterations - 1:
            print(f"Iteration {iteration + 1}/{iterations}, weight change: {weight_change:.6f}, "
                  f"learning rate: {learning_rate:.4f} beta coeff: {beta}")
    return weights

def load_crypto_prices(file_path):
    if not os.path.exists(file_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = find_csv_candidates([script_dir, os.getcwd()])
        suggestion = "\n".join([f" - {c}" for c in candidates]) if candidates else "(no .csv files found in script directory or working directory)"
        raise FileNotFoundError(
            f"CSV file not found at: '{file_path}'.\n"
            f"Current working directory: '{os.getcwd()}'.\n"
            f"Script directory: '{script_dir}'\n"
            f"Candidate CSV files:\n{suggestion}\n\n"
            f"If you want to specify a CSV file, run the script like:\n"
            f"   python {os.path.basename(__file__)} --file " + '"<path/to/database_btc.csv>"' + "\n"
        )
    data = pd.read_csv(file_path, parse_dates=['Start', 'End'])
    data.sort_values('Start', inplace=True, ascending=True)
    return data[['Start', 'Close']]

def create_fragments_1(data, fragment_length=21, step=7):
    fragments = []
    for start in range(0, len(data) - fragment_length + 1, step):
        fragments.append(data.iloc[start:start + fragment_length]['Close'].values)
    return np.array(fragments)

def create_fragments_2(data, fragment_length=28, step=7):
    fragments = []
    min_max_values = []
    dates = []
    for start in range(0, len(data) - fragment_length + 1, step):
        fragment = data.iloc[start:start + fragment_length]['Close'].values
        min_val, max_val = np.min(fragment), np.max(fragment)
        min_max_values.append((min_val, max_val))
        normalized_fragment = (fragment - min_val) / (max_val - min_val)
        fragments.append(normalized_fragment)
        dates.append(data.iloc[start:start + fragment_length]['Start'].values)
    return np.array(fragments), np.array(min_max_values), np.array(dates)

def normalize_fragments(fragments):
    normalized = []
    for fragment in fragments:
        norm_fragment = (fragment - np.min(fragment)) / (np.max(fragment) - np.min(fragment))
        normalized.append(norm_fragment)
    return np.array(normalized)

def plot_clusters_separately_with_weights(clusters, weights, title_prefix="Cluster"):
    colors = ['red', 'green', 'purple', 'orange', 'blue', 'black', 'cyan', 'pink', 'brown', 'gray']
    for cluster_index, cluster_fragments in clusters.items():
        if not cluster_fragments:
            continue

        plt.figure(figsize=(10, 6))

        for fragment in cluster_fragments:
            plt.plot(fragment, color=colors[cluster_index % len(colors)], alpha=0.7, label="Fragment" if 'Fragment' not in plt.gca().get_legend_handles_labels()[1] else "")

        plt.plot(weights[cluster_index], color="black", linestyle="-", linewidth=5, label="Vector of weights")

        plt.title(f"{title_prefix} {cluster_index}")
        plt.xlabel("Number of day in fragment")
        plt.ylabel("Normilized value")
        plt.grid(True)
        plt.legend()
        plt.show()

def get_default_csv_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, 'database_btc.csv')


def find_csv_candidates(base_dirs):
    candidates = []
    for d in base_dirs:
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if fname.lower().endswith('.csv'):
                candidates.append(os.path.join(d, fname))
    return candidates


def parse_args():
    parser = argparse.ArgumentParser(description='Neural Network timelines prediction')
    parser.add_argument('--file', '-f', default=None, help='Path to the CSV file (optional)')
    return parser.parse_args()


args = parse_args()
if args.file:
    file_path = os.path.abspath(os.path.expanduser(args.file))
else:
    file_path = get_default_csv_path()
data = load_crypto_prices(file_path)
fragment_length = 28
step = 7
fragments = create_fragments_1(data, fragment_length, step)
normalized_fragments = normalize_fragments(fragments)
n_neurons = 30
weights_sync = train_kohonen_fragments_sync(
    normalized_fragments, n_neurons, fragment_length,
    initial_learning_rate=0.5, decay_rate=0.98, iterations=250
)

clusters_sync = cluster_fragments(weights_sync, normalized_fragments)

n_clusters = len([cluster for cluster in clusters_sync.values() if len(cluster) > 0])
print(f"Amount of clusters after filtering: {n_clusters}")
print("Amount of dots:")
for cluster_index, cluster_points in clusters_sync.items():
    print(f"Cluster {cluster_index}: {len(cluster_points)} dots")

plot_clusters_separately_with_weights(clusters_sync, weights_sync, title_prefix="Cluster")

print(weights_sync)


forecast_length = fragment_length - 7
fragments, min_max_values, fragment_dates = create_fragments_2(data, fragment_length, step)
n_neurons = 30

last_index = len(fragments) - 1
test_index = last_index
train_indices = list(range(0, last_index - 2))
train_fragments = fragments[train_indices]
train_min_max = min_max_values[train_indices]
train_dates = fragment_dates[train_indices]

weights_sync = train_kohonen_fragments_sync(
    train_fragments, n_neurons, fragment_length,
    initial_learning_rate=0.5, decay_rate=0.9, iterations=250
)
clusters_sync = cluster_fragments(weights_sync, normalized_fragments)
test_fragment = fragments[test_index]
test_dates = fragment_dates[test_index]

bmu_index, predicted_cluster = find_bmu_2(weights_sync, test_fragment[:forecast_length], clusters_sync)
cluster_weight_vector = weights_sync[bmu_index]
predicted_values_norm = cluster_weight_vector[-(fragment_length - forecast_length):]
real_values_norm = test_fragment[-(fragment_length - forecast_length):]
input_values_norm = test_fragment[:forecast_length]
min_val, max_val = min_max_values[test_index]

predicted_values = predicted_values_norm * (max_val - min_val) + min_val
real_values = real_values_norm * (max_val - min_val) + min_val
input_values = input_values_norm * (max_val - min_val) + min_val

print(f"The last testing fragment: {test_dates[0]} - {test_dates[-1]}")
mse_error = np.mean(abs((real_values - predicted_values))/real_values)*100
print(f"Prediction - relative error: {mse_error:.2f}%")
plt.figure(figsize=(10, 5))
full_x = list(range(fragment_length))

plt.subplot(1, 2, 1)
plt.plot(full_x[:forecast_length], input_values, color='red', label='Input data')
plt.title(f'Input data')
plt.xlabel('Days number of fradment')
plt.ylabel('Real values')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(full_x[forecast_length:], predicted_values, color='green', label='Prognosed vector of weights')
plt.plot(full_x[forecast_length:], real_values, color='blue', label='Real values')
for i in range(fragment_length - forecast_length):
    plt.plot([forecast_length + i, forecast_length + i], [predicted_values[i], real_values[i]], color='gray', linestyle='dotted')
plt.title(f'Prediction - relative error: {mse_error:.2f}%')
plt.xlabel('Number of day in fragment')
plt.ylabel('Real values')
plt.legend()

plt.ylim(min(min(real_values), min(predicted_values)) * 0.9, max(max(real_values), max(predicted_values)) * 1.2)

plt.tight_layout()
plt.show()
