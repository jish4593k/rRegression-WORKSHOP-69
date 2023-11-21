
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


datasets = pd.read_csv('Position_Salaries.csv')
X = datasets.iloc[:, 1:2].values
Y = datasets.iloc[:, 2].values



sc_X = StandardScaler()
X_scaled = sc_X.fit_transform(X)
sc_Y = StandardScaler()
Y_scaled = np.squeeze(sc_Y.fit_transform(Y.reshape(-1, 1)))

model = Sequential()
model.add(Dense(units=6, activation='relu', input_dim=1))
model.add(Dense(units=6, activation='relu'))
model.add(Dense(units=1, activation='linear'))

# Compiling the Neural Network
model.compile(optimizer='adam', loss='mean_squared_error')


model.fit(X_scaled, Y_scaled, epochs=100, batch_size=5)


X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, Y, color='red', label='Actual Data')
plt.plot(X_grid, sc_Y.inverse_transform(model.predict(sc_X.transform(X_grid))), color='blue', label='Neural Network Regression')
plt.title('Neural Network Regression results')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.legend()
plt.show()


new_position_level = 6.5
scaled_prediction = model.predict(sc_X.transform([[new_position_level]]))[0][0]
predicted_salary = sc_Y.inverse_transform([[scaled_prediction]])[0][0]

print(f'Predicted Salary for Position Level {new_position_level}: ${predicted_salary:.2f}')
