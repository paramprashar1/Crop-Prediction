# File 2
import pandas as pd

df = pd.read_csv("punjab.csv")
# print(df.head())
# df = df.replace(["BHATINDA"], "BATHINDA")
# df = df.replace(["MANSHA"], "MANSA")
# df = df.replace(["N.SHAHAR"], "NAWANSHAHR")

print(df["year"].unique())
# df.to_csv("punjab.csv")
