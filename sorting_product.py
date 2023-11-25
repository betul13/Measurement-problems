import pandas as pd
from sklearn.preprocessing import MinMaxScaler
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format",lambda x : "%.5f" % x)

df = pd.read_csv(r"datasets/product_sorting.csv")

#Sorting by Rating

df.sort_values("rating",ascending = False).head(20)

#Sorting by comment count or purchase count

df.sort_values("purchase_count",ascending = False).head(20)

df.sort_values("commment_count",ascending = False).head(20)

# Sorting by Rating,Comment and Purchase

df["purchase_count_scaled"] = MinMaxScaler(feature_range=(1,5)).fit(df[["purchase_count"]]).transform(df[["purchase_count"]])

df["comment_count_scaled"] = MinMaxScaler(feature_range=(1,5)).fit(df[["commment_count"]]).transform(df[["commment_count"]])

def weighted_sorting_score(dataframe, w1 = 32, w2 = 26, w3 = 42):
    return (df["comment_count_scaled"] * w1 / 100 +
            df["purchase_count_scaled"] * w2/ 100 +
            df["rating"] * w3 / 100 )

df["weighted_sorting_score"] = weighted_sorting_score(df)

df.sort_values("weighted_sorting_score",ascending = False).head(20)