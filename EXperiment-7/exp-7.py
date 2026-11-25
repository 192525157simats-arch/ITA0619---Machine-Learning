from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

data = load_iris()

X = data.data[:100]
y = data.target[:100]

model = LogisticRegression(max_iter=200)
model.fit(X,y)

sample = [[5.1,3.5,1.4,0.2]]

pred = model.predict(sample)

print("Predicted Class:",pred[0])
print("Flower:",data.target_names[pred[0]])
