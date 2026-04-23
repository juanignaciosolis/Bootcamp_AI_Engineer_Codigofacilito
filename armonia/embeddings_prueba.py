from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

load_dotenv()

client = OpenAI()

def get_embedding(
        text: str,
        model: str = "text-embedding-3-small" # 1536 dimensiones, barato y bueno
) -> list[float]:
    """
    Genera el embeddign de un texto
    """
    text = text.replace("\n", " ").strip()
    print(text)
    response = client.embeddings.create(
        input= text, model= model
    )
    return response.data[0].embedding

# Probamos

emb = get_embedding(input("Ingrese la oracion: "))
print("Dimensiones ", len(emb)) # 1536 dimensiones
print("Primeros 5 ", emb[:5])

# Graficamos

texts = [
    input("1 - primera frase:"),
    input("2 - primera frase:"),
    input("3 - primera frase:")
]


embeddings = [get_embedding(t) for t in texts]

X = np.array(embeddings)

# 🔹 PCA con 3 componentes
pca = PCA(n_components=3)
X_reduced = pca.fit_transform(X)

# 🔹 Crear figura 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 🔹 Scatter 3D
ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    X_reduced[:, 2]
)

# 🔹 Labels
for i, (x, y, z) in enumerate(X_reduced):
    label = f"{i} - {texts[i]}"
    ax.text(x, y, z, label, fontsize=9)

# 🔹 Decoración
ax.set_title("Embeddings en 3D (PCA)")
ax.set_xlabel("Componente 1")
ax.set_ylabel("Componente 2")
ax.set_zlabel("Componente 3")

plt.show()