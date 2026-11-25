from sklearn.tree import DecisionTreeClassifier

X = [
    [20,3,0],[25,2,0],[30,2,1],[40,1,1],
    [50,1,1],[60,0,1],[22,4,0],[45,2,1]
]

y = ["Low","Low","Medium","Medium",
     "High","High","Low","High"]

model = DecisionTreeClassifier()
model.fit(X,y)

a = int(input("Enter income: "))
b = int(input("Enter loans: "))
c = int(input("Payment history (1=Good, 0=Poor): "))

print("Credit Score:",model.predict([[a,b,c]])[0])
