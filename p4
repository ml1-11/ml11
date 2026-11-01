import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

data = load_diabetes(as_frame=True)
X = data.data[["bmi"]]
y = data.target
print("Dataset Shape:", X.shape)
print("\nFirst 5 Records:\n", X.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

degrees = [1, 2, 3, 5, 8, 10]
train_mse, test_mse, models = [], [], []

for d in degrees:
    model = make_pipeline(PolynomialFeatures(d), StandardScaler(), LinearRegression())
    model.fit(X_train, y_train)
    train_mse.append(mean_squared_error(y_train, model.predict(X_train)))
    test_mse.append(mean_squared_error(y_test, model.predict(X_test)))
    models.append(model)

results = pd.DataFrame({"Degree": degrees, "Training MSE": train_mse, "Testing MSE": test_mse})
print("\nPolynomial Regression Results:\n", results.round(2))

best_index = 1 + np.argmin(test_mse[1:])
print("\nBest Polynomial Degree:", degrees[best_index])
print("Minimum Testing MSE:", round(test_mse[best_index], 2))

plt.figure(figsize=(8,5))
plt.plot(degrees, train_mse, marker="o", label="Training MSE")
plt.plot(degrees, test_mse, marker="s", label="Testing MSE")
plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.title("Bias-Variance Trade-off")
plt.legend()
plt.grid()
plt.show()

curve_model = models[degrees.index(8)]
X_plot = np.linspace(X["bmi"].min(), X["bmi"].max(), 500).reshape(-1, 1)
y_plot = curve_model.predict(pd.DataFrame(X_plot, columns=["bmi"]))

plt.figure(figsize=(9, 5))
plt.scatter(X_test["bmi"], y_test, label="Actual Data")
plt.plot(X_plot, y_plot, color="red", linewidth=2, label="Polynomial Degree 8")
plt.xlabel("BMI")
plt.ylabel("Disease Progression")
plt.title("Polynomial Regression Curve (Degree 8)")
plt.legend()
plt.grid()
plt.show()
