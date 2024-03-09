import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


risk_df = pd.read_csv(r"D:\Studia\0. SGH\MAGISTERKA\dane\Maternal Health Risk Data Set.csv", sep=';', decimal=',')
risk_df.shape
risk_df['RiskLevel'].unique()
risk_df.isnull().sum()
risk_df.info()

mapping = {'low risk': 1, 'mid risk': 2, 'high risk': 3}
risk_df['RiskLevel'] = risk_df['RiskLevel'].map(mapping)
risk_df[['RiskLevel']].value_counts()

X_col_names = risk_df.iloc[:, 0:6].columns

# Zamiana skali z Fahrenheita na Celcjusza
risk_df['BodyTemp'] = (risk_df['BodyTemp'] - 32) * 5/9
# Skosnosc
risk_df[X_col_names].skew()
# Statystyki opisowe
risk_df[X_col_names].describe()
# Wspolczynnik zmiennosci
cv_result = (risk_df[X_col_names].std() / risk_df[X_col_names].mean()) * 100
print(cv_result)

# Wykresy gestosci
sns.set(style="whitegrid")
for col in X_col_names:
    plt.figure(figsize=(8, 4))
    sns.kdeplot(risk_df[col], shade=True)
    plt.title(f'Wykres gęstości dla {col}')
    plt.xlabel(f'{col}')
    plt.ylabel('Gęstość')
    plt.show()
    
# Wykres czestosci dla RiskLevel
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.countplot(x='RiskLevel', data=risk_df, palette='plasma')
plt.title('Wykres częstości dla zmiennej RiskLevel')
plt.xlabel('Poziom ryzyka')
plt.ylabel('Liczba wystąpień')
plt.xticks(ticks=[0, 1, 2], labels=['Low Risk (1)', 'Medium Risk (2)', 'High Risk (3)']) # Opcjonalnie, aby lepiej opisać co oznaczają wartości 1, 2, 3
plt.show()

# Wartosci odstajace
q1 = risk_df[X_col_names].quantile(0.25)
q3 = risk_df[X_col_names].quantile(0.75)
iqr = q3 - q1
l_bound = (q1 - 1.5 * iqr)
u_bound = (q1 + 1.5 * iqr)
n_oo_l = (risk_df[X_col_names][iqr.index] < l_bound).sum()
n_oo_u = (risk_df[X_col_names][iqr.index] > u_bound).sum()
outliers = pd.DataFrame({'low_boundary':l_bound, 'up_boundary':u_bound, 'outliers l':n_oo_l, 'outliers U': n_oo_u})
print(outliers)

# Usuniecie wartosci odstajacych w HeartRate
risk_df_sorted = risk_df.sort_values(by='HeartRate', ascending=True)
risk_df_sorted
risk_df_filtered = risk_df.loc[risk_df['HeartRate'] != 7]
risk_df_filtered.shape

# Korelacja
correlation = risk_df_filtered.corr().round(2)
plt.figure(figsize = (10,6))
sns.heatmap(correlation, annot = True, cmap = 'YlOrBr')

risk_df_sieci = risk_df_filtered.copy()

# Standaryzacja danych
scaler = StandardScaler()
scaled_data = scaler.fit_transform(risk_df_filtered[X_col_names])
risk_df_scaled = pd.DataFrame(scaled_data, columns=X_col_names)
risk_df_filtered.update(risk_df_scaled)
risk_df_filtered








