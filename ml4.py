import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from keras.optimizers import Optimizer
from keras import optimizers

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

df_train = pd.read_csv("X_train.csv")
df_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("Y_train.csv")
y_test = pd.read_csv("Y_test.csv")


cr = LabelEncoder()
se = LabelEncoder()
ds = LabelEncoder()
df_train["season"] = se.fit_transform(df_train["season"].values)
df_train["crop"] = cr.fit_transform(df_train["crop"].values)
df_train["district"] = ds.fit_transform(df_train["district"].values)


df_test["season"] = se.transform(df_test["season"].values)
df_test["crop"] = cr.transform(df_test["crop"].values)
df_test["district"] = ds.fit_transform(df_test["district"].values)

x_train = df_train.values
x_test = df_test.values

y_train = y_train.values
y_test = y_test.values

y_train = np.log10(y_train)
y_test = np.log10(y_test)

print(len(x_train))
print(len(y_train))

# print(y_train[:, 0].shape)
# index = 0
# for i in x_train[:, 1]:
#     if not np.isfinite(i):
#         print(index, i)
#     index += 1


reg = LinearRegression()
reg.fit(x_train, y_train)
training_accuracy = reg.score(x_train, y_train)
test_accuracy = reg.score(x_test, y_test)
rmse_train = np.sqrt(mean_squared_error(reg.predict(x_train), y_train))
rmse_test = np.sqrt(mean_squared_error(reg.predict(x_test), y_test))
print(
    "Training Accuracy = %0.3f, Test Accuracy = %0.3f, RMSE (train) = %0.3f, RMSE (test) = %0.3f"
    % (training_accuracy, test_accuracy, rmse_train, rmse_test)
)

y_true = y_test
y_pred = reg.predict(x_test)
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
plt.title("Linear Regression")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.savefig("Linear Regression Scatter")

print("Linear Regression")
print("exp variance err", explained_variance_score(y_true, y_pred))
print("max error", max_error(y_true, y_pred))
print("mae", mean_absolute_error(y_true, y_pred))
print("mse", mean_squared_error(y_true, y_pred))
print("mean sq log error", mean_squared_log_error(y_true, y_pred))
print("median absolute error", median_absolute_error(y_true, y_pred))
print("r2", r2_score(y_true, y_pred))
print("Success!")

# Xnew2 = [["MANSA", 2002, "Cotton(lint)", "Kharif", 70000, 19.445, 28.05]]
# predc = 10 ** reg.predict(Xnew2)
# print(predc)

# y_predB=reg.predict(X)


# print(df_test["district"])
