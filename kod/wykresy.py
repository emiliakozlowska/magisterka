import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

poland = pd.read_csv(r"D:\Studia\0. SGH\MAGISTERKA\dane\poland-maternal-mortality-rate.csv", sep=";")
poland

x = poland["Year"]
y = poland["Per 100K Live Births"]
plt.figure(figsize=(10, 6))
sns.lineplot(x=x, y=y, marker="o", color="red")
plt.fill_between(x, y, color="pink", alpha=0.1)
plt.title('Zgony matek w Polsce w latach 2000-2020')
plt.xlabel('Rok')
plt.ylabel('Wskaźnik zgonów na 100 000 urodzeń')
plt.xticks(x[::2])
plt.show()

india = pd.read_csv(r"D:\Studia\0. SGH\MAGISTERKA\dane\india-maternal-mortality-rate.csv", sep=";")
india

sns.set(style="whitegrid")
x = india["Year"].round()
y = india["Per 100K Live Births"]
plt.figure(figsize=(10, 6))
sns.lineplot(x=x, y=y, marker="o", color="red")
plt.fill_between(x, y, color="pink", alpha=0.1)
plt.title('Zgony matek w Indiach w latach 2000-2020')
plt.xlabel('Rok')
plt.ylabel('Wskaźnik zgonów na 100 000 urodzeń')
plt.xticks(x[::2])
plt.show()