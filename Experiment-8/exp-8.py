from sklearn.linear_model import LinearRegression

X = [[1],[2],[3],[4],[5],[6]]
y = [35,40,50,55,65,70]

model = LinearRegression()
model.fit(X,y)

a = float(input("Enter study hours: "))

print("Predicted Marks:",model.predict([[a]])[0])
