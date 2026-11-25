from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

data = load_iris()

X = data.data
y = data.target

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X,y)

sample = [[5.1,3.5,1.4,0.2]]

pred = model.predict(sample)

print("Flower:",data.target_names[pred[0]])
