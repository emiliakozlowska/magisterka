import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import random as rn
import xgboost as xgb
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



risk_df = pd.read_csv(r"D:\Studia\0. SGH\MAGISTERKA\dane\Maternal Health Risk Data Set.csv", sep=';', decimal=',')
risk_df.shape
risk_df['RiskLevel'].unique()
risk_df.isnull().sum()
risk_df.info()

mapping = {'low risk': 0, 'mid risk': 1, 'high risk': 2}
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
sns.set_theme(style="whitegrid")
for col in X_col_names:
    plt.figure(figsize=(8, 4))
    sns.kdeplot(risk_df[col], shade=True)
    plt.title(f'Wykres gęstości dla {col}')
    plt.xlabel(f'{col}')
    plt.ylabel('Gęstość')
    plt.show()
    
# Wykres czestosci dla RiskLevel
sns.set_theme(style="whitegrid")
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

# DRZEWA DECYZYJNE

X = risk_df_filtered[X_col_names]
y = risk_df_filtered['RiskLevel']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Definiowanie siatki parametrów do przeszukiwania
param_grid = {
    'max_depth': [5, 10, 15,],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [2, 4, 5],
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random']
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
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Dokładność modelu: {accuracy:.2f}')
print(classification_report(y_test, y_pred))
print(conf_matrix)

# Wykres macieży pomyłek
labels = ['Low Risk', 'Medium Risk', 'High Risk']
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Macierz pomyłek')
plt.xlabel('Przewidywane etykiety')
plt.ylabel('Prawdziwe etykiety')
plt.show()
 
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

######################
# ROC? można dla każdej klasy osobno zrobić


###############################################################################

# LASY LOSOWE

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Podstawowy model z 100 drzewami
random_forest_100 = RandomForestClassifier(n_estimators = 100, 
                                           criterion = 'gini',
                                           max_depth = None,
                                           min_samples_leaf = 1,
                                           min_weight_fraction_leaf = 0.0,
                                           max_features = 'sqrt',
                                           max_leaf_nodes = None,
                                           bootstrap = True,
                                           oob_score = False,
                                           min_impurity_decrease = 0.0,                                           
                                           random_state = 1)
random_forest_100.fit(X_train, y_train)

# Ocena modelu na zbiorze treningowym i testowym
y_pred_train = random_forest_100.predict(X_train)
accuracy_train = accuracy_score(y_train, y_pred_train)
y_pred_test = random_forest_100.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
print("Dokładność predykcji na zbiorze treningowym:", accuracy_train)
print("Dokładność predykcji na zbiorze testowym:", accuracy_test)
print(classification_report(y_test,random_forest_100.predict(X_test)))
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Szukanie hiperparametrów
random_forest = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 20, 30],
    'min_samples_leaf': [1, 2, 3],
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [2, 3, 4]
}
grid_search = GridSearchCV(estimator=random_forest, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
grid_search.fit(X_train, y_train)
print("Najlepsze parametry:", grid_search.best_params_)
print("Najlepsza dokładność:", grid_search.best_score_)
best_model = grid_search.best_estimator_

# Ocena modelu na zbiorze treningowym i testowym
y_pred_train = best_model.predict(X_train)
accuracy_train = accuracy_score(y_train, y_pred_train)
y_pred_test = best_model.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
print("Dokładność predykcji na zbiorze treningowym:", accuracy_train)
print("Dokładność predykcji na zbiorze testowym:", accuracy_test)
print(classification_report(y_test, best_model.predict(X_test)))
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Ważnoć cech
feature_importance = best_model.feature_importances_
feature_score = pd.Series(feature_importance, index=X_train.columns)
score = feature_score.sort_values()
plt.figure(figsize=(12,5))
plt.barh(y=score.index, width=score.values, color='blue')
plt.grid(alpha=0.4)
plt.title('Feature Importance of Best Model')
plt.show()

###############################################################################

# XGBoost

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=3, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# Ocena modelu
accuracy = accuracy_score(y_test, y_pred_xgb)
print("Dokładność predykcji: %.2f%%" % (accuracy * 100.0))
y_pred_train = xgb_model.predict(X_train)
accuracy_train = accuracy_score(y_train, y_pred_train)
y_pred_test = xgb_model.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
print("Dokładność predykcji na zbiorze treningowym:", accuracy_train)
print("Dokładność predykcji na zbiorze testowym:", accuracy_test)
print(classification_report(y_test, y_pred_test))
conf_matrix = confusion_matrix(y_test, y_pred_test)
print(conf_matrix)

# Poszukiwanie hiperparametrów
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.1, 0.2, 0.3],
    'max_depth': [10, 20, 50],
    'min_child_weight': [2, 3, 4],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.001, 0.01],
    'reg_lambda': [0, 0.001, 0.01]
}
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)
print("Najlepsze parametry:", grid_search.best_params_)
print("Najlepsza dokładność:", grid_search.best_score_)
best_model = grid_search.best_estimator_
y_pred_test = best_model.predict(X_test)

# Ocena modelu
accuracy = accuracy_score(y_test, y_pred_test)
print("Dokładność predykcji: %.2f%%" % (accuracy * 100.0))
y_pred_train = best_model.predict(X_train)
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)
print("Dokładność predykcji na zbiorze treningowym:", accuracy_train)
print("Dokładność predykcji na zbiorze testowym:", accuracy_test)
print(classification_report(y_test, y_pred_test))
conf_matrix = confusion_matrix(y_test, y_pred_test)
print(conf_matrix)

# Ważnoć cech
feature_importance = best_model.feature_importances_
feature_score = pd.Series(feature_importance, index=X_train.columns)
score = feature_score.sort_values()
plt.figure(figsize=(12,5))
plt.barh(y=score.index, width=score.values, color='blue')
plt.grid(alpha=0.4)
plt.title('Feature Importance of Best Model')
plt.show()

###############################################################################

# SVM (do wywalenia)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

# Ocena modelu
accuracy = accuracy_score(y_test, y_pred)
print("Dokładność predykcji: %.2f%%" % (accuracy * 100.0))
print(classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Szukanie hiperparametów
param_grid = {
    'C': [0.1, 1, 10],  # Parametr regularyzacji
    'gamma': ['scale', 0.001, 0.01, 0.1],
}
grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)
print("Najlepsze parametry:", grid_search.best_params_)
print("Najlepsza dokładność:", grid_search.best_score_)

###############################################################################

# Sieci neuronowe

risk_df_sieci.head()
X = risk_df_sieci[X_col_names]
y = risk_df_sieci['RiskLevel']
y_one_hot = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.3, random_state=42)

# Przekształcenie zmiennych
num_transformer = StandardScaler()
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, X_col_names)
    ]) 
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

np.random.seed(42)
rn.seed(42)
tf.random.set_seed(42)

# Sieć 1
siec1 = Sequential()
siec1.add(Dense(6, input_dim=X_train.shape[1], activation='relu'))
siec1.add(Dense(3, activation='softmax'))
siec1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
siec1.fit(X_train, y_train, epochs=50, batch_size=10)

accuracy_train = siec1.evaluate(X_train, y_train)
print('Accuracy: %.2f' % (accuracy_train[1]*100))
accuracy_test = siec1.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy_test[1]*100))

# Sieć 2 - wybór liczby epok
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
siec2t = Sequential()
siec2t.add(Dense(6, input_dim=X_train.shape[1], activation='relu'))
siec2t.add(Dense(3, activation='softmax'))
siec2t.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
siec2t.fit(X_train, y_train, epochs=200, batch_size=10, 
           validation_data=(X_test, y_test), 
           callbacks=[early_stopping])  # Epoki = 115

siec2 = Sequential()
siec2.add(Dense(6, input_dim=X_train.shape[1], activation='relu'))
siec2.add(Dense(3, activation='softmax'))
siec2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
siec2.fit(X_train, y_train, epochs=150, batch_size=10)

accuracy_train = siec2.evaluate(X_train, y_train)
print('Accuracy: %.2f' % (accuracy_train[1]*100))
accuracy_test = siec2.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy_test[1]*100))

# Sieć 3 - callback
checkpoint = ModelCheckpoint(filepath='best_model.keras', monitor='val_accuracy', 
                             save_best_only=True, mode='max', verbose=1)

siec3 = Sequential()
siec3.add(Dense(6, input_dim=X_train.shape[1], activation='relu'))
siec3.add(Dense(3, activation='softmax'))
siec3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
siec3.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=150, batch_size=10, callbacks=[checkpoint])

accuracy_train = siec3.evaluate(X_train, y_train)
print('Accuracy: %.2f' % (accuracy_train[1]*100))
accuracy_test = siec3.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy_test[1]*100))

# Sieć 4

# Funkcja tworząca model
def create_model(neurons=1, hidden_layers=1):
    model = Sequential()
    model.add(Dense(neurons, input_shape=(X_train.shape[1],), activation='relu'))
    for _ in range(hidden_layers - 1):
        model.add(Dense(neurons, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))  # Liczba neuronów odpowiada liczbie klas
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Zdefiniuj zakres hiperparametrów do przetestowania
neurons_range = [15, 20, 25]
hidden_layers_range = [2, 3, 4]

best_score = 0
best_params = {}

# Pętla do iteracji przez hiperparametry
for neurons in neurons_range:
    for hidden_layers in hidden_layers_range:
        # Tworzenie i trenowanie modelu
        model = create_model(neurons=neurons, hidden_layers=hidden_layers)
        model.fit(X_train, y_train, epochs=115, batch_size=10, verbose=0, validation_split=0.2)
        
        # Ocena modelu
        _, accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        # Aktualizacja najlepszych parametrów, jeśli model jest lepszy
        if accuracy > best_score:
            best_score = accuracy
            best_params = {'neurons': neurons, 'hidden_layers': hidden_layers}

# Wyświetlenie najlepszych hiperparametrów
print("Najlepsze hiperparametry:", best_params)
print("Dokładność:", best_score)


siec4 = Sequential()
siec4.add(Dense(15, input_dim=X_train.shape[1], activation='relu'))
siec4.add(Dense(15, activation='relu'))
siec4.add(Dense(15, activation='relu'))
siec4.add(Dense(3, activation='softmax'))
siec4.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
siec4.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=115, batch_size=10)

accuracy_train = siec4.evaluate(X_train, y_train)
print('Accuracy: %.2f' % (accuracy_train[1]*100))
accuracy_test = siec4.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy_test[1]*100))