import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

X = np.array([[1],[2],[3],[4],[5],[6]])
y = np.array([1,4,9,16,25,36])

m1 = LinearRegression()
m1.fit(X,y)

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

m2 = LinearRegression()
m2.fit(X_poly,y)

print("Linear Prediction:",m1.predict(X))
print("Polynomial Prediction:",m2.predict(X_poly))
