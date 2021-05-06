import pandas as pd

yield_df = pd.read_csv("joint.csv")
# print(df.head())
# print(df.describe())
# print(yield_df.info())

# Hot Encoding

from sklearn.preprocessing import OneHotEncoder

yield_df_onehot = pd.get_dummies(yield_df, columns=["district", "crop", "season"])
features = yield_df_onehot.loc[:, yield_df_onehot.columns != "yield"]
label = yield_df["yield"]
# print(features.info())
# print(label)
features = features.drop(["year"], axis=1)
# print(features.head())


# Scaling features
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
features = scaler.fit_transform(features)
# print(type(features))

from sklearn.model_selection import train_test_split

train_data, test_data, train_labels, test_labels = train_test_split(
    features, label, test_size=0.3, random_state=42
)
yield_df.to_csv("yield_df.csv")
train_data, test_data, train_labels, test_labels = train_test_split(
    features, label, test_size=0.3, random_state=42
)

# Model Comparison and Selection

from sklearn.metrics import r2_score


def compare_models(model):
    model_name = model.__class__.__name__
    fit = model.fit(train_data, train_labels)
    y_pred = fit.predict(test_data)
    r2 = r2_score(test_labels, y_pred)
    return [model_name, r2]


from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor

models = [
    GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=0),
    RandomForestRegressor(n_estimators=200, max_depth=3, random_state=0),
    svm.SVR(),
    DecisionTreeRegressor(),
]

print(compare_models)
print(models)
print(type(compare_models))
# model_train = list(map(compare_models, models))
