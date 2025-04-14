import requests

url = "http://127.0.0.1:8000/predict_diabetes"
data = {
    "age": 0.05,
    "sex": -0.02,
    "bmi": 0.04,
    "bp": 0.02,
    "s1": -0.01,
    "s2": 0.03,
    "s3": -0.02,
    "s4": 0.01,
    "s5": -0.04,
    "s6": 0.02
}

response = requests.post(url, json=data)
print("Prediction:", response.json())