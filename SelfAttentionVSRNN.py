import numpy as np
import time
import matplotlib.pyplot as plt

# Fixed input embedding (3 tokens, 2 dimensions)
x_input = np.array([
    [-0.288779, 0.579001],   # "My"
    [-0.405061, -0.517733],  # "First"
    [0.693840, -0.061268]    # "Project"
])
n, d = x_input.shape

# ---------------------------------------------------------
# Simulates a basic RNN using your embeddings
# ---------------------------------------------------------
def simulate_rnn(x):
    n, d = x.shape
    h = np.zeros((n, d))
    w = np.random.rand(d, d)
    for t in range(1, n):
        h[t] = np.tanh(np.dot(x[t], w) + np.dot(h[t - 1], w))
    return h

# ---------------------------------------------------------
# Simulates self-attention using your embeddings
# ---------------------------------------------------------
def simulate_self_attention(x):
    n, d = x.shape
    q = x @ np.random.rand(d, d)
    k = x @ np.random.rand(d, d)
    v = x @ np.random.rand(d, d)
    attention_scores = q @ k.T
    attention_weights = attention_scores / np.sqrt(d)
    attention_output = attention_weights @ v
    return attention_output

# ---------------------------------------------------------
# Time both models on the same input
# ---------------------------------------------------------
start = time.time()
rnn_output = simulate_rnn(x_input)
rnn_time = time.time() - start

start = time.time()
attention_output = simulate_self_attention(x_input)
attention_time = time.time() - start

# ---------------------------------------------------------
# Print results
# ---------------------------------------------------------
print("RNN Output:")
print(rnn_output)
print(f"RNN Time: {rnn_time:.6f} seconds\n")

print("Self-Attention Output:")
print(attention_output)
print(f"Self-Attention Time: {attention_time:.6f} seconds")

# ---------------------------------------------------------
# Plotting the RNN and Self-Attention outputs
# ---------------------------------------------------------
import matplotlib.pyplot as plt

tokens = ["My", "First", "Project"]
rnn_x, rnn_y = rnn_output[:, 0], rnn_output[:, 1]
attn_x, attn_y = attention_output[:, 0], attention_output[:, 1]

plt.figure(figsize=(12, 5))

# RNN plot
plt.subplot(1, 2, 1)
plt.scatter(rnn_x, rnn_y, c='blue')
for i, token in enumerate(tokens):
    plt.annotate(token, (rnn_x[i], rnn_y[i]), fontsize=12)
plt.title("RNN Output Embeddings")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.grid(True)

# Attention plot
plt.subplot(1, 2, 2)
plt.scatter(attn_x, attn_y, c='green')
for i, token in enumerate(tokens):
    plt.annotate(token, (attn_x[i], attn_y[i]), fontsize=12)
plt.title("Self-Attention Output Embeddings")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.grid(True)

plt.tight_layout()
plt.show()
