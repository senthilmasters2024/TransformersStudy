import numpy as np

# Input Embeddings (from your PCA reduced values)
X = np.array([
    [-0.288779, 0.579001],   # My
    [-0.405061, -0.517733],  # First
    [ 0.693840, -0.061268]   # Project
])

# Define the number of heads and dimensions
num_heads = 2
d_model = X.shape[1] # 2
d_k_per_head = d_model // num_heads # 2 // 2 = 1

print(f"Input X (Shape: {X.shape}):\n{X}\n")

# --- Define Weight Matrices for each Head ---
# Head 1 Weights (d_model x d_k_per_head = 2x1)
W_Q1 = np.array([[0.5], [0.3]])
W_K1 = np.array([[0.7], [0.4]])
W_V1 = np.array([[0.6], [0.2]])

# Head 2 Weights (d_model x d_k_per_head = 2x1)
W_Q2 = np.array([[0.8], [0.2]])
W_K2 = np.array([[0.1], [0.5]])
W_V2 = np.array([[0.7], [0.3]])

# Final Output Weight Matrix (num_heads * d_k_per_head x d_model = 2x2)
W_O = np.array([[0.9, 0.1], [0.1, 0.9]])

print(f"Head 1 Weights (Shape: {W_Q1.shape}):")
print(f"W_Q1:\n{W_Q1}\nW_K1:\n{W_K1}\nW_V1:\n{W_V1}\n")
print(f"Head 2 Weights (Shape: {W_Q2.shape}):")
print(f"W_Q2:\n{W_Q2}\nW_K2:\n{W_K2}\nW_V2:\n{W_V2}\n")
print(f"Output Projection Matrix W_O (Shape: {W_O.shape}):\n{W_O}\n")

# --- Helper function for Self-Attention (single head) ---
def single_head_attention(X, W_Q, W_K, W_V, d_k):
    # 1. Linear projections
    Q = np.dot(X, W_Q)
    K = np.dot(X, W_K)
    V = np.dot(X, W_V)

    # 2. Attention scores (Q * K^T)
    M = np.dot(Q, K.T)

    # 3. Scaling
    scale_factor = np.sqrt(d_k)
    M_scaled = M / scale_factor

    # 4. Softmax (Attention Weights A)
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    A = softmax(M_scaled)

    # 5. Weighted sum of Values
    output_head = np.dot(A, V)
    return output_head, A # Also return A for inspection

# --- Perform Multi-Head Attention ---
attention_outputs = [] # To store outputs from each head
attention_weights = [] # To store attention weights from each head

# Head 1 Calculation
print("--- Calculating Head 1 ---")
output_h1, A_h1 = single_head_attention(X, W_Q1, W_K1, W_V1, d_k_per_head)
attention_outputs.append(output_h1)
attention_weights.append(A_h1)
print(f"Output Head 1 (Shape: {output_h1.shape}):\n{output_h1}\n")
print(f"Attention Weights Head 1 (Shape: {A_h1.shape}):\n{A_h1}\n")


# Head 2 Calculation
print("--- Calculating Head 2 ---")
output_h2, A_h2 = single_head_attention(X, W_Q2, W_K2, W_V2, d_k_per_head)
attention_outputs.append(output_h2)
attention_weights.append(A_h2)
print(f"Output Head 2 (Shape: {output_h2.shape}):\n{output_h2}\n")
print(f"Attention Weights Head 2 (Shape: {A_h2.shape}):\n{A_h2}\n")


# 3. Concatenate Outputs of all Heads
concatenated_output = np.concatenate(attention_outputs, axis=1)
print(f"--- Concatenated Output (Shape: {concatenated_output.shape}) ---")
print(concatenated_output)

# 4. Final Linear Projection (W_O)
final_multi_head_output = np.dot(concatenated_output, W_O)
print(f"\n--- Final Multi-Head Attention Output (Shape: {final_multi_head_output.shape}) ---")
print(final_multi_head_output)
