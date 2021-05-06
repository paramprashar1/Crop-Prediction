import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

import tk

# plt.use("TkAgg")

df = pd.read_csv("joint.csv")

# print(df.head())

corrMatrix = df.corr()
# print(corrMatrix)


# sn.heatmap(corrMatrix, annot=True)
# plt.savefig("correlation.png")
# plt.show()
# df["yield"].hist()
# df.hist()
# plt.savefig("dataHist.png")


print(df.describe())
print(df.groupby("crop").count())
# df_mod = df.drop([''], axis=1)
# print(df_mod.head())
# df.drop(
#     df.columns[df.columns.str.contains("unnamed", case=False)], axis=1, inplace=True
# )
print(df.info())
print(df.head())
