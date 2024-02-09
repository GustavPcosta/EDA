

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle


data = {
    'id': [1, 2, 3, 4, 5],
    'nome': ['Apartamento 1', 'Apartamento 2', 'Apartamento 3', 'Apartamento 4', 'Apartamento 5'],
    'bairro_group': ['Manhattan', 'Brooklyn', 'Manhattan', 'Queens', 'Brooklyn'],
    'bairro': ['Midtown', 'Williamsburg', 'Upper East Side', 'Astoria', 'Park Slope'],
    'latitude': [40.75362, 40.71427, 40.77339, 40.7681, 40.67283],
    'longitude': [-73.98377, -73.96151, -73.96625, -73.91514, -73.97083],
    'room_type': ['Entire home/apt', 'Private room', 'Entire home/apt', 'Private room', 'Entire home/apt'],
    'price': [225, 100, 300, 80, 150],
    'minimo_noites': [1, 2, 3, 1, 2],
    'numero_de_reviews': [45, 30, 20, 10, 50],
    'ultima_review': ['2019-05-21', '2019-06-15', '2019-07-10', '2019-08-05', '2019-09-20'],
    'reviews_por_mes': [0.38, 0.5, 0.3, 0.2, 0.6],
    'calculado_host_listings_count': [2, 1, 3, 1, 2],
    'disponibilidade_365': [355, 180, 240, 120, 300]
}


df = pd.DataFrame(data)


df['ultima_review'] = pd.to_datetime(df['ultima_review'])


print(df.head())


print(df.describe())

df.hist(figsize=(12, 10))
plt.tight_layout()
plt.show()


sns.pairplot(df)
plt.show()


correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


plt.figure(figsize=(12, 6))
sns.boxplot(x='bairro', y='price', data=df)
plt.title('Preço por Bairro')
plt.xticks(rotation=90)
plt.show()


plt.figure(figsize=(12, 6))
sns.boxplot(x='room_type', y='price', data=df)
plt.title('Preço por Tipo de Quarto')
plt.xticks(rotation=90)
plt.show()


df = pd.get_dummies(df, drop_first=True)

X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


modelo = LinearRegression()
modelo.fit(X_train, y_train)

train_score = modelo.score(X_train, y_train)
test_score = modelo.score(X_test, y_test)
print(f'Train Score: {train_score}')
print(f'Test Score: {test_score}')

with open('modelo.pkl', 'wb') as f:
    pickle.dump(modelo, f)
