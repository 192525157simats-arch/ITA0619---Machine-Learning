from sklearn.neighbors import KNeighborsClassifier

X = [[2,60],[3,65],[5,80],[6,85],[1,50],[7,90]]
y = ["Fail","Fail","Pass","Pass","Fail","Pass"]

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X,y)

a = float(input("Enter study hours: "))
b = float(input("Enter attendance: "))

print("Result:",model.predict([[a,b]])[0])
