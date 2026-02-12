import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv("data.csv")

X = data[["Hours_Studied", "Hours_Slept"]]
y = data["Test_Score"]

model = LinearRegression()
model.fit(X, y)

prediction = model.predict([[6, 8]])

print("The AI predicted a test score of " + str(prediction[0]) +
      " for a student who studied 6 hours and slept 8 hours.")




