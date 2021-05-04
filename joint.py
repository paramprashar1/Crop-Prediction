import pandas as pd

# reading two csv files
data1 = pd.read_csv("punjab.csv")
data2 = pd.read_csv("TempRain.csv")

# using merge function by setting how='inner'
output1 = pd.merge(data1, data2, on=["district", "year"], how="inner")
# mergeDf = pd.concat([data1, data2], keys=["district", "year"])

# displaying result
print(output1)
output1.to_csv("joint.csv")
