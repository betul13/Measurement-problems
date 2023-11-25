import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width",500)
pd.set_option("float_format",lambda x : "%5.f" %x)

df = pd.read_csv(r"datasets/movies_metadata.csv",low_memory=False)

df.columns

df = df[["title","vote_average","vote_count"]]

df.head()

df.dropna(inplace=True)

#weighted_rating = (v/(v+M) * r ) + (M/(v+M) * C)
#r = vote average
#v = vote_count
#M = minimum votes required to be listed in the top 250
#C = the mean vote across the whole report(currently 7.0)

M = 2500
C = df["vote_average"].mean()
def weighted_rating(r, v, M, C):
    return (v/(v+M) * r ) + (M/(v+M) * C)

df["weighted_rating"] = weighted_rating(df["vote_average"],df["vote_count"], M, C)

df.sort_values("weighted_rating",ascending = False).head(20)