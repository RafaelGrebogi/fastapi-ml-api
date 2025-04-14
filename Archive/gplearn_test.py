from gplearn.genetic import SymbolicRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

random_state = 41

# Load data
data = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

# Define Genetic Programming model
gp = SymbolicRegressor(population_size=1000, generations=100,  metric='mse', verbose=1, random_state = random_state,  parsimony_coefficient = 0.01)
gp.fit(X_train, y_train)

# Predict and print best symbolic expression
y_pred = gp.predict(X_test)
print(gp._program)  # Best evolved mathematical formula