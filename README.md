# Machine Learning API for Regression

This project provides a FastAPI-based machine learning API that allows users to make predictions using trained models.

## Features
- **Diabetes Prediction:** Predicts diabetes progression using a linear regression model.
- **California Housing Prediction:** Estimates housing prices based on features.
- **FastAPI Integration:** Provides an API for real-time predictions.

## 🛠 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ml-api.git
   cd ml-api
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Running the API
Start the FastAPI server:
```bash
uvicorn main:app --reload
```

## 💼 API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST   | `/predict_diabetes` | Predicts diabetes progression. |
| POST   | `/predict_calihousing` | Predicts house prices in California. |

## 📝 Example API Request (Diabetes)
Send a **POST request** with JSON input:
```json
{
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
```

## Model Equations
### **Diabetes Model**
```
y = 19.570*age + -240.127*sex + 521.213*bmi + 298.487*bp + -580.913*s1 +
    255.229*s2 + 4.186*s3 + 142.841*s4 + 730.115*s5 + 68.946*s6 + 152.478
```

### **California Housing Model**
```
y = 0.442*MedInc + 0.010*HouseAge + -0.115*AveRooms + 0.761*AveBedrms +
    -0.000*Population + -0.008*AveOccup + -0.429*Latitude + -0.443*Longitude + -37.812
```

## 📝 Notes
- The models are trained using **scikit-learn** and stored as `.pkl` files.
- **gplearn** and **TensorFlow Keras** may be integrated for future improvements.

