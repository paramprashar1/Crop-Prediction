import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import max_error
from sklearn.metrics import explained_variance_score

df = pd.read_csv("joint.csv")

# print(df.info())
Xfeatures = ["district", "year", "crop", "season", "area", "AvgRain", "AvgTemp"]
X = df[Xfeatures]
# print(X)
Y = df["production"]
# print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
print(type(X_test), type(X_train), type(Y_test), type(Y_train))
X_test.to_csv("X_test.csv")
X_train.to_csv("X_train.csv")
Y_train.to_csv("Y_train.csv")
Y_test.to_csv("Y_test.csv")
