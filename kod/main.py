import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree


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

###############################################################################

# accuracy, precision, recall, and F1 score

# Drzewa decyzyjne

X = risk_df_filtered[X_col_names]
y = risk_df_filtered['RiskLevel']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Definiowanie siatki parametrów do przeszukiwania
param_grid = {
    'max_depth': [None, 5, 10, 15, 20, 25],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 4, 6, 8]
}

# Przeszukiwanie siatki w celu znalezienia najlepszego zestawu parametrów
grid_search_params = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
grid_search_params.fit(X_train, y_train)
best_params = grid_search_params.best_params_
print(best_params)

# Budowanie wstępnego modelu drzewa decyzyjnego (bez przycinania)
initial_clf = DecisionTreeClassifier(random_state=42, **best_params)
initial_clf.fit(X_train, y_train)

# Ocena modelu
y_pred = initial_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Dokładność modelu: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Przycinanie Drzewa - znalezienie optymalnego ccp_alpha
ccp_alpha_values = np.linspace(start=0.001, stop=0.02, num=100)  # Przykładowy większy zakres
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42),
                           param_grid={'ccp_alpha': ccp_alpha_values}, cv=5)
grid_search.fit(X_train, y_train)
best_ccp_alpha = grid_search.best_params_['ccp_alpha']
print(best_ccp_alpha)

# Budowanie i trenowanie ostatecznego modelu drzewa decyzyjnego z optymalnym ccp_alpha
final_clf = DecisionTreeClassifier(random_state=42, ccp_alpha=best_ccp_alpha)
final_clf.fit(X_train, y_train)

# Ocena modelu
y_pred = final_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Dokładność modelu: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Wizualizacja ważności cech
feature_importances = final_clf.feature_importances_
features = X.columns
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importances in Decision Tree Model')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Wizualizacja Ostatecznego Drzewa Decyzyjnego
plt.figure(figsize=(20, 12))
plot_tree(final_clf, 
          filled=True, 
          feature_names=X.columns.tolist(), 
          fontsize=12,
          rounded=True, 
          impurity=True)
plt.show()

###############################################################################

# Lasy losowe
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)
y_pred = random_forest_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Dokładność modelu: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

###############################################################################

# Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
gradient_boosting_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gradient_boosting_model.fit(X_train, y_train)
y_pred = gradient_boosting_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Dokładność modelu: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

###############################################################################

# SVM
from sklearn.svm import SVC
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Dokładność modelu: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

###############################################################################

# SGD
from sklearn.linear_model import SGDClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
sgd_model = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, max_iter=1000, tol=1e-3, random_state=42)
sgd_model.fit(X_train, y_train)
y_pred = sgd_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Dokładność modelu: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

###############################################################################

# Sieci neuronowe









