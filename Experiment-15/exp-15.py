from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB

data = load_iris()

X = data.data
y = data.target

model = GaussianNB()
model.fit(X,y)

sample = [[5.1,3.5,1.4,0.2]]

pred = model.predict(sample)

print("Flower:",data.target_names[pred[0]])
