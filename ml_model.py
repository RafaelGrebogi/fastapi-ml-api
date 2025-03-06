import pandas as pd
import numpy as np

# This first version is using just regression datasets from Sklearn
# Datasets: diabetes, california_housing
from sklearn.datasets import load_diabetes, fetch_california_housing


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

