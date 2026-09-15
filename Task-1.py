import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("API_SP.POP.TOTL_DS2_en_csv_v2_366238.csv", skiprows=4)

year = "2025"

data = df[["Country Name", year]].dropna()

data = data.sort_values(by=year, ascending=False).head(10)

plt.figure(figsize=(10, 6))
plt.bar(data["Country Name"], data[year])

plt.xlabel("Country")
plt.ylabel("Population")
plt.title("Population Distribution by Country - 2025")

plt.xticks(rotation=90)
plt.tight_layout()

plt.show()