#!/usr/bin/env python3
"""Script to visualize a transformed pd.DataFrame"""


import matplotlib.pyplot as plt
import pandas as pd

from_file = __import__("2-from_file").from_file

df = from_file("coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv", ",")
df = df.drop(columns=["Weighted_Price"])
df = df.rename(columns={"Timestamp": "Date"})
df["Date"] = pd.to_datetime(df["Date"], unit="s")
df = df.set_index("Date")
df["Close"] = df["Close"].ffill()
df["High"] = df["High"].fillna(df["Close"])
df["Low"] = df["Low"].fillna(df["Close"])
df["Open"] = df["Open"].fillna(df["Close"])
df["Volume_(BTC)"] = df["Volume_(BTC)"].fillna(0)
df["Volume_(Currency)"] = df["Volume_(Currency)"].fillna(0)

df_2017_beyond = df[df.index >= "2017"]
df_daily = df_2017_beyond.resample("D").agg(
    {
        "High": "max",
        "Low": "min",
        "Open": "mean",
        "Close": "mean",
        "Volume_(BTC)": "sum",
        "Volume_(Currency)": "sum",
    }
)
plt.plot(df_daily.index, df_daily["High"], label="High")
plt.plot(df_daily.index, df_daily["Low"], label="Low")
plt.plot(df_daily.index, df_daily["Open"], label="Open")
plt.plot(df_daily.index, df_daily["Close"], label="Close")
plt.plot(df_daily.index, df_daily["Volume_(BTC)"], label="Volume_(BTC)")
plt.plot(df_daily.index, df_daily["Volume_(Currency)"],
         label="Volume_(Currency)")

plt.xlabel("Date")
plt.legend()
ax = plt.gca()
months = pd.date_range(start="2017-01", end="2019-02", freq="3MS")
ax.set_xticks(months)
labels = []
for i, date in enumerate(months):
    if i == 0 or date.year != months[i - 1].year:
        labels.append(f"{date.strftime('%b')}\n{date.year}")
    else:
        labels.append(date.strftime("%b"))
ax.set_xticklabels(labels)

plt.show()
print(df_daily)
