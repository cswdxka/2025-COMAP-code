import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

# Load data
data = pd.read_csv("C:/Users/dxw/Desktop/1.csv")

# Reshape the data to 2D array (needed for KMeans)
X = data['Total'].values.reshape(-1, 1)

wcss = []
for i in range(1, 11):  # Try k from 1 to 10
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the elbow graph
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

