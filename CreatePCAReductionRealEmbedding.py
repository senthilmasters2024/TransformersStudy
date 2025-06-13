import openai
import numpy as np
import json
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# === 1. API Setup ===
client = openai.OpenAI(api_key="")  # Use your OpenAI key

# === 2. Define tokens ===
tokens = ["My", "First", "Project"]

# === 3. Get Embeddings using new SDK client ===
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=tokens
)

# Extract embeddings from response
embeddings = [item.embedding for item in response.data]

# === 4. Save embeddings as JSON ===
embedding_dict = [{"token": token, "embedding": emb} for token, emb in zip(tokens, embeddings)]
json_path = "openai_embeddings_tokens.json"
with open(json_path, "w") as f:
    json.dump(embedding_dict, f, indent=2)
print(f"Saved embeddings to {json_path}")

# === 5. Reduce embeddings to 2D with PCA ===
pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)

# === 6. Plot the reduced embeddings ===
x, y = reduced[:, 0], reduced[:, 1]
plt.figure(figsize=(8, 6))
plt.scatter(x, y, c='blue')

# Annotate each point with the token
# Annotate each point with the token and print PCA coordinates
for i, (token, coords) in enumerate(zip(tokens, reduced)):
    plt.annotate(token, (x[i], y[i]), fontsize=12)
    print(f"{token}: [{coords[0]:.6f}, {coords[1]:.6f}]")

