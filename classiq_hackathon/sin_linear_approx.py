import numpy as np
import matplotlib.pyplot as plt

# Define the function and its derivative
f = np.sin
df = np.cos

def sin_linear_approximation(
        domain_start: float = 0,
        domain_stop : float = 1,
        ) -> None:
    """
    This function generates Chebyshev nodes as a fuction of the required n_qbits_precision.
    """
    # Find sin precision as a function of n_qbits_precision
    n_qbits_precision_arr = np.arange(2, 20)
    sin_precision = np.zeros(len(n_qbits_precision_arr))
    for precision_idx, n_qbits_precision in enumerate(n_qbits_precision_arr):
        x_arr = np.linspace(domain_start, domain_stop, 2**n_qbits_precision)
        sin_x = f(x_arr)
        sin_x_precision = np.round(sin_x * 2**n_qbits_precision) / 2**n_qbits_precision
        sin_precision[precision_idx] = max(np.abs(sin_x - sin_x_precision))
    # plot_sin_precision(n_qbits_precision_arr, sin_precision)

    chebyshev_nodes = dict()
    for nodes_idx, n_nodes in enumerate(range(2, 20)):
        chebyshev_nodes[nodes_idx] = generate_chebyshev_nodes(domain_start, domain_stop, n_nodes)
        for precision_idx, n_qbits_precision in enumerate(n_qbits_precision_arr):
            x_arr = np.linspace(domain_start, domain_stop, 2**n_qbits_precision)
            sin_x = f(x_arr)
            approx_values = np.array([
                piecewise_linear_approximation(x, chebyshev_nodes[nodes_idx]["edges"], chebyshev_nodes[nodes_idx]["slop"], chebyshev_nodes[nodes_idx]["intercept"])
                for x in x_arr]
                )
            approx_values = np.round(approx_values * 2**n_qbits_precision) / 2**n_qbits_precision
            chebyshev_error = np.max(np.abs(approx_values - sin_x))
            if chebyshev_error < sin_precision[precision_idx]:
                chebyshev_nodes[nodes_idx]["desired_precision"] = n_qbits_precision
            
        

# Generate Chebyshev nodes and their function values
def generate_chebyshev_nodes(domain_start: float, domain_stop: float, n_nodes: int):
    """
    This function generates Chebyshev nodes, including their linear approximation and edges
    """
    # Generate Chebyshev nodes
    nodes_idx = np.arange(0, n_nodes)
    chebyshev_nodes_locs = np.flip((domain_start + domain_stop) / 2 + (domain_stop - domain_start) / 2 * np.cos((2 * nodes_idx + 1) * np.pi / (2 * n_nodes)))
    chebyshev_nodes_locs = np.flip(np.array([(domain_start + domain_stop) / 2 + (domain_stop - domain_start) / 2 * np.cos((2 * i + 1) * np.pi / (2 * n_nodes)) for i in range(n_nodes)]))
    chebyshev_nodes_vals = f(chebyshev_nodes_locs)
    chebyshev_nodes_df = df(chebyshev_nodes_locs)
    
    # Calculate the linear approximation and edges
    slop = chebyshev_nodes_df
    intercept = chebyshev_nodes_vals - chebyshev_nodes_locs * chebyshev_nodes_df
    edges = find_edges(domain_start, domain_stop, chebyshev_nodes_locs, slop, intercept)

    # Return the Chebyshev nodes
    chebyshev_nodes = dict(
        n_nodes     = n_nodes,
        locs        = chebyshev_nodes_locs,
        vals        = chebyshev_nodes_vals,
        slop        = slop,
        intercept   = intercept,
        edges       = edges
    )
    return chebyshev_nodes

def find_edges(domain_start, domain_stop, nodes, slope, intercept):
    """
    This function calculates the Chebyshev nodes.
    """
    edge = np.zeros(len(nodes)+1)
    edge[0] = domain_start
    edge[-1] = domain_stop
    for i in range(1, len(nodes)):
        edge[i] = (intercept[i] - intercept[i-1]) / (slope[i-1] - slope[i])
    return edge

def piecewise_linear_approximation(x: np.ndarray, nodes: np.ndarray, slope: np.ndarray, intercept: np.ndarray):
    """
    This function calculates the piecewise linear approximation of sin(x) for a given x.
    """
    for i_node in range(0,len(nodes)-1):
        if nodes[i_node] <= x <= nodes[i_node+1]:
            return slope[i_node] * x + intercept[i_node]
    return None # This should never happen

# Plot the results
def plot_chebyshev_nodes(chebyshev_nodes):
    """
    This function plots the Chebyshev nodes for a given n_nodes.
    """
    # Plot the results
    plt.figure(figsize=(10, 8))

    # Subplot 1: Original vs. Approximation
    plt.subplot(2, 1, 1)
    plt.plot(test_points, true_values, label="True sin(x)", color="blue")
    plt.plot(test_points, approx_values, label="Piecewise Linear Approximation", color="red", linestyle="--")
    plt.scatter(all_edges, all_edges_vals, label="egese", color="black")
    plt.scatter(chebyshev_nodes, chebyshev_nodes_vals, label="Chebyshev Nodes", color="green")
    plt.legend()
    plt.title(f"Piecewise Linear Approximation of sin(x) with {n_nodes} Chebyshev Nodes")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)

    # Subplot 2: Approximation Error
    plt.subplot(2, 1, 2)
    plt.plot(test_points, error_values, label="Error (Approx - True)", color="purple")
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.title("Difference Between Approximation and True sin(x)")
    plt.xlabel("x")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show(block=True)

def plot_sin_precision(n_qbits_precision_arr, sin_precision):
    """
    This function plots the sin precision as a function of n_qbits_precision.
    """
    plt.figure(figsize=(10, 8))
    plt.plot(n_qbits_precision_arr, sin_precision, marker='x', linestyle='-', color='r', label="Sin Precision")
    plt.xticks(n_qbits_precision_arr)
    plt.xlim(n_qbits_precision_arr[0], n_qbits_precision_arr[-1])
    plt.ylim(0, max(np.ceil(sin_precision*100)/100))
    plt.xlabel("n_qbits_precision")
    plt.ylabel("Abs. Error")
    plt.title("Sin Abs. Error vs. n_qbits_precision")
    plt.grid(True)
    plt.show(block=True)

# test_points = np.arange(domain_start, domain_stop, 1/2**6)
# approx_values = np.array([piecewise_linear_approximation(x, all_edges, slop, intercept) for x in test_points])
# approx_values = np.round(approx_values * 2**6) / 2**6
# true_values = f(test_points)
# error_values = approx_values - true_values
# print(f"{chebyshev_nodes=}\nMaximum error:", np.max(np.abs(error_values)))


if __name__ == "__main__":
    sin_linear_approximation()
