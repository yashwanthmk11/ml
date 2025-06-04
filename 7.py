import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

def linear_regression():
    data = fetch_california_housing(as_frame=True)
    X, y = data.data[["AveRooms"]], data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    plt.scatter(X_test, y_test, color="blue", label="Actual")
    plt.plot(X_test, y_pred, color="red", label="Predicted")
    plt.xlabel("AveRooms")
    plt.ylabel("House Value ($100k)")
    plt.title("Linear Regression")
    plt.legend()
    plt.show()

    print("Linear Regression Results")
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R² Score:", r2_score(y_test, y_pred))

def polynomial_regression():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    cols = ["mpg", "cyl", "disp", "hp", "wt", "acc", "year", "origin"]
    df = pd.read_csv(url, sep=r'\s+', names=cols, na_values="?").dropna()

    X = df[["disp"]].values
    y = df["mpg"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = make_pipeline(PolynomialFeatures(2), StandardScaler(), LinearRegression())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    plt.scatter(X_test, y_test, color="blue", label="Actual")
    plt.scatter(X_test, y_pred, color="red", label="Predicted")
    plt.xlabel("Displacement")
    plt.ylabel("MPG")
    plt.title("Polynomial Regression")
    plt.legend()
    plt.show()

    print("Polynomial Regression Results")
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R² Score:", r2_score(y_test, y_pred))

if __name__ == "__main__":
    print("Linear and Polynomial Regression Demo")
    linear_regression()
    polynomial_regression()
