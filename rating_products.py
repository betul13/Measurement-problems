import pandas as pd
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler
import math
import datetime as dt
pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)
pd.set_option("display.width",500)
pd.set_option("display.float_format",lambda x : "%.5f" %x)

df = pd.read_csv(r"datasets/course_reviews.csv")
print(df.head())
print(df["Rating"].value_counts())
df.reset_index()

print(df.groupby("Questions Asked").agg({"Questions Asked":"count",
                                         "Rating": "mean"}))

# ortalama puan

df["Rating"].mean() #bu şekilde yaparsak son zamanlardaki müşteri memnuniyet trendini kaçırmış oluruz.

#Time-Based Weighted Average

print(df.info()) #zaman ifadeleri object bunları dönüştürmemiz gerek.

df["Enrolled"] = pd.to_datetime(df["Enrolled"])
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

current_date = dt.datetime(2021,2,10)

df["days"] = (current_date - df["Timestamp"]).dt.days

def time_based_weighted_average (dataframe, w1 = 28, w2 = 26 , w3 = 24, w4 = 22) :
    return  dataframe.loc[dataframe["days"] <= 30, "Rating"].mean() * w1 / 100 + \
            dataframe.loc[(dataframe["days"] > 30) & (dataframe["days"] <= 90), "Rating"].mean() * w2 / 100 + \
            dataframe.loc[(dataframe["days"] > 90) & (dataframe['days'] <= 180), "Rating"].mean() * w3 / 100 + \
            dataframe.loc[dataframe["days"] > 180, "Rating"].mean() * w4 / 100

#time_based_weighted_average(df)

#User-Based Weighted Average

df.groupby("Progress").agg({"Rating":"mean"})

def user_based_weighted_average (dataframe, w1 = 22, w2 = 24 , w3 = 26, w4 = 28) :
    return  dataframe.loc[(dataframe["Progress"] <= 10), "Rating"].mean() * w1 / 100 + \
            dataframe.loc[(dataframe["Progress"] > 10) & (dataframe["Progress"] <= 45), "Rating"].mean() * w2 / 100 + \
            dataframe.loc[(dataframe["Progress"] > 45) & (dataframe['Progress'] <= 75), "Rating"].mean() * w3 / 100 + \
            dataframe.loc[(dataframe["Progress"] > 75), "Rating"].mean() * w4 / 100

#Weighted Rating

def course_weighted_rating(dataframe, time_w = 50, user_w = 50):
    return time_based_weighted_average(dataframe) * time_w / 100 + user_based_weighted_average(dataframe) * user_w / 100

course_weighted_rating(df)

