# File 1


import pandas as pd

data = pd.read_csv("agridata.csv", index_col="state")
rows = data.loc["PUNJAB"]
rows["Year_ob"] = rows.year.str[:4]
del rows["year"]


rows = rows.reindex(
    ["state", "district", "Year_ob", "crop", "season", "area", "production", "yield"],
    axis=1,
)


def drop_col_n(df, col_n_to_drop):
    col_dict = {x: col for x, col in enumerate(df.columns)}
    return df.drop(col_dict[col_n_to_drop], 1)


rows = drop_col_n(rows, 0)
print(rows.head())
rows.rename(columns={"Year_ob": "year"}, inplace=True)
print(rows)

# Extracting Punjab's agri data into a new file
rows.to_csv("punjab.csv")
