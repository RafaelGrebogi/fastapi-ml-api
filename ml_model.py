import pandas as pd
import numpy as np

# This first version is using just regression datasets from Sklearn
# Datasets: diabetes, california_housing
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.linear_model import LinearRegression  # Implements Linear Regression model
from sklearn.metrics import mean_squared_error  # Measures model accuracy


####################
# DEFINITIONS
####################
def get_random_state(random_state=None):
    """
    If random_state is set, return it.
    If None, generate a random integer seed.
    """
    if random_state is None:
        random_state = np.random.randint(0, 10000)  # Generate a random seed
        print(f"Generated random_state: {random_state}")  # Optional: Show the seed

    return random_state

def get_manual_split_perc(split_perc=None):
    """
    If split_perc is set, return it.
    If None, set a default split percentage.
    """
    if split_perc is None:
        split_perc = 0.8
        print(f"Manual split percentage defined as: {split_perc * 100}")
    
    return split_perc


        

####################
# LOAD DATASETS
####################

# Load data and Create Dataframe -> Diabetes
data_diabetes = load_diabetes()

df_diabetes = pd.DataFrame(data_diabetes.data, columns=data_diabetes.feature_names)
df_diabetes["target"] = data_diabetes.target


# Load data and Create Dataframe -> California Housing
data_calihousing = fetch_california_housing()

df_calihousing = pd.DataFrame(data_calihousing.data,  columns=data_calihousing.feature_names)
df_calihousing["target"] = data_calihousing.target


####################
# DATA PREPROCESSING
####################

# Verify missing values in the Diabetes dataset
if df_diabetes.isnull().sum().sum() > 0:
    print("Warning: There are missing values in the Diabetes dataset!")
    df_diabetes.dropna(inplace=True)
else:
    print("No missing values detected in the Diabetes dataset.")

# Verify missing values in the Calihousing dataset
if df_diabetes.isnull().sum().sum() > 0:
    print("Warning: There are missing values in the California Housing dataset!")
    df_calihousing.dropna(inplace=True)
else:
    print("No missing values detected in the California Housing dataset.")

####################
# MACHINE LEARNING
####################

# (1) Split train-test subsets

# Diabetes dataset
X_diabetes = df_diabetes.drop(columns=["target"])  # Extract Features (independent variables)
y_diabetes = df_diabetes["target"]  # Extract Target variable (dependent variable)

# California Housing dataset
X_calihousing = df_calihousing.drop(columns=["target"])
y_calihousing = df_calihousing["target"]

split_type = "random"

if split_type == "random":
    from sklearn.model_selection import train_test_split  # Splits dataset into train & test sets

    user_defined_seed = None  # THIS WILL BE A FUNCTION INPUT PARAMETER

    random_state = get_random_state(user_defined_seed) # Set to an integer for a fixed seed, or None for random
    # Split data: 80% training, 20% testing
    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_diabetes, y_diabetes, test_size=0.2, random_state=random_state)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_calihousing, y_calihousing, test_size=0.2, random_state=random_state)
    print("Using random split")

elif split_type == "manual":
    split_manual_trainPerc = 0.8 # THIS WILL BE A FUNCTION INPUT PARAMETER

    split_manual_trainPerc = get_manual_split_perc(split_manual_trainPerc) # If not defined, set split to 80%-20%

    split_index = int(split_manual_trainPerc * len(X_diabetes))
    X_train_d, y_train_d = X_diabetes[:split_index], y_diabetes[:split_index]
    X_test_d, y_test_d = X_diabetes[split_index:], y_diabetes[split_index:]

    split_index = int(split_manual_trainPerc * len(X_calihousing))
    X_train_c, y_train_c = X_calihousing[:split_index], y_calihousing[:split_index]
    X_test_c, y_test_c = X_calihousing[split_index:], y_calihousing[split_index:]
    print("Using manual sequential split")

else:
    print("Invalid split type. Choose 'random' or 'manual'.")


# (2) Create and trani Linear Regression models
# Train Linear Regression model for Diabetes dataset
model_diabetes = LinearRegression()  # Create model
model_diabetes.fit(X_train_d, y_train_d)  # Train using training data

# Train Linear Regression model for California Housing dataset
model_calihousing = LinearRegression()
model_calihousing.fit(X_train_c, y_train_c)

# Predict target values on the test set
y_pred_d = model_diabetes.predict(X_test_d)
y_pred_c = model_calihousing.predict(X_test_c)

mse_d = mean_squared_error(y_test_d, y_pred_d)
mse_c = mean_squared_error(y_test_c, y_pred_c)

print(f"Diabetes Dataset - Mean Squared Error: {mse_d:.2f}")
print(f"California Housing Dataset - Mean Squared Error: {mse_c:.2f}")


# Print Diabetes equation
features = X_train_d.columns  # Get feature names
coefs = model_diabetes.coef_  # Get coefficients

# Format the equation
equation = "y = " + " + ".join(f"{coef:.3f}*{feature}" for coef, feature in zip(coefs, features))
equation += f" + {model_diabetes.intercept_:.3f}"

print("Diabetes Model Equation:")
print(equation)

# Print California Housing equation
features = X_train_c.columns  # Get feature names
coefs = model_calihousing.coef_  # Get coefficients

# Format the equation
equation = "y = " + " + ".join(f"{coef:.3f}*{feature}" for coef, feature in zip(coefs, features))
equation += f" + {model_calihousing.intercept_:.3f}"

print("California Housing Model Equation:")
print(equation)