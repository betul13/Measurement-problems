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

#Vote Average'a göre sıralama

df.sort_values("vote_average",ascending=False) #vote count göz ardı edildiği için mantıklı değil

df["vote_count"].describe([0.10, 0.25, 0.50, 0.70, 0.90, 0.95, 0.99]).T #filtreleme yapıcaz çünkü vote countu çok düşük olan ya da değerlendirilmeyenler var

#df[df["vote_count"] > 400].sort_values("vote_average",ascending=False).head(20) #pek verimli değil

df["vote_count_scaled"] = MinMaxScaler(feature_range=(1,10)).fit(df[["vote_count"]]).transform(df[["vote_count"]]) #vote_count u standartlaştırdık.

df["average_count_score"] = df["vote_average"] * df["vote_count_scaled"]

df.sort_values("average_count_score",ascending=False).head(20)