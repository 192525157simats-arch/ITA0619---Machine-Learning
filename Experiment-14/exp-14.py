from sklearn.linear_model import LinearRegression

X = [
    [600,1],[800,2],[1000,2],[1200,2],
    [1500,3],[1800,3],[2000,4],[2200,4]
]

y = [20,28,35,42,55,65,75,82]

model = LinearRegression()
model.fit(X,y)

a = float(input("Enter house area: "))
b = int(input("Enter bedrooms: "))

print("House Price:",model.predict([[a,b]])[0],"Lakhs")
