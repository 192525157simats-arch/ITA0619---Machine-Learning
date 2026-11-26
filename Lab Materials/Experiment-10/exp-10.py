import numpy as np
from sklearn.mixture import GaussianMixture

X = np.array([
    [1,2],[1.5,1.8],[2,2.2],
    [8,8],[8.5,8.2],[9,8.5]
])

model = GaussianMixture(n_components=2,random_state=42)
model.fit(X)

print("Cluster Labels:")
print(model.predict(X))

print("Cluster Means:")
print(model.means_)
