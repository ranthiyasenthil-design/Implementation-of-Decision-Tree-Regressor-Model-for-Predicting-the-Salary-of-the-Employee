# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required Python libraries such as Pandas and Scikit-learn.

2.Load the Salary.csv dataset using Pandas.

3.Select Level as the independent variable and Salary as the dependent variable.

4.Create a Decision Tree Regressor model.

5.Train the model using the given dataset.

6.Predict the salary for the available employee levels.

7.Calculate the Mean Squared Error and R² score to evaluate the model.

8.Predict the salary for a new employee based on the given level.

9.Display the actual and predicted salary values.

10.Display the prediction for the new employee.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by:RANTHIYA S
RegisterNumber:  212225230225
*/
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("Salary.csv")

# Display the dataset
print("Employee Salary Dataset:")
print(data)

# Select the independent variable
X = data[["Level"]]

# Select the dependent variable
y = data["Salary"]

# Create Decision Tree Regressor
model = DecisionTreeRegressor(
    random_state=42
)

# Train the model
model.fit(X, y)

# Predict salaries
y_pred = model.predict(X)

# Display actual and predicted salaries
print("\nActual Salaries:")
print(y.values)

print("\nPredicted Salaries:")
print(y_pred)

# Calculate Mean Squared Error
mse = mean_squared_error(y, y_pred)

# Calculate R2 Score
r2 = r2_score(y, y_pred)

print("\nMean Squared Error:", mse)
print("R2 Score:", r2)

# Predict salary for a new employee
new_level = pd.DataFrame({
    "Level": [6]
})

predicted_salary = model.predict(new_level)

print("\nNew Employee Level:", 6)
print("Predicted Salary:", predicted_salary[0])
```

## Output:
<img width="1628" height="558" alt="image" src="https://github.com/user-attachments/assets/64452419-803c-431e-bd39-927432191222" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
