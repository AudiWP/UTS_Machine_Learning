import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

# Data
X = np.array([[1, 3], [7, 10], [4, 4], [1, 6], [4, 8]])
names = ['N1', 'N2', 'N3', 'N4', 'N5']

# Inisialisasi model K-Means dengan 2 klaster
model = KMeans(n_clusters=2, random_state=42)

# Fit model dengan data
model.fit(X)

# Prediksi klaster untuk setiap data
labels = model.predict(X)

# Kelompokkan data berdasarkan klaster
cluster_a = [names[i] for i in range(len(labels)) if labels[i] == 0]
cluster_b = [names[i] for i in range(len(labels)) if labels[i] == 1]

# Tampilkan hasil anggota klaster
print("Klaster A:", cluster_a)
print("Klaster B:", cluster_b)

# Buat scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')

# Tambahkan label untuk setiap titik data
for i, name in enumerate(names):
    plt.annotate(name, (X[i, 0], X[i, 1]), textcoords="offset points", xytext=(0,10), ha='center')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Hasil Clustering K-Means')
plt.grid(False)
plt.show()