# Motion Data ML API

Python backend for the Motion Data System, providing data processing, feature extraction and machine-learning functionality for motion data collected by an ESP32-based embedded device.

The API is implemented using **FastAPI** and integrates with **Firebase** and **Supabase** for device communication, data storage and service management.

## Main Functions

The backend supports three main operating modes:

* **Training** – processes labelled motion data and trains a machine-learning model
* **Testing** – evaluates a trained model against labelled test data
* **Production** – processes new sensor data and generates predictions

The API exposes endpoints that allow the embedded system and associated applications to trigger these workflows.

## Data Processing

Motion data from the accelerometer and gyroscope is processed using fixed-size windows.

For each window, features are extracted from the three accelerometer and gyroscope axes, including:

* Time-domain statistical features
* Frequency-domain features
* FFT-based signal information

The extracted features are then used as inputs to the machine-learning models.

## Machine Learning

The current implementation supports:

* **Random Forest classification using scikit-learn**
* **TensorFlow/Keras models**

The system includes model training, testing and prediction workflows, together with model archiving and storage of test results.

## Backend and Database

The FastAPI application integrates with:

* **Firebase** for receiving and processing sensor data
* **Supabase/PostgreSQL** for users, devices, services and application data
* REST API endpoints for communication between the embedded device and backend services

Environment variables are used for database and service credentials.

## Main Technologies

* Python
* FastAPI
* NumPy
* Pandas
* SciPy
* scikit-learn
* TensorFlow/Keras
* Firebase
* Supabase/PostgreSQL

## Related Project

The embedded firmware and data acquisition system are available here:

[MotionDataSystem](https://github.com/RafaelGrebogi/MotionDataSystem)

The embedded system uses an **ESP32 and MPU6050 IMU** to acquire motion data and communicate with this backend for processing and machine-learning analysis.

## Running the API

Create and activate a Python virtual environment, install the required dependencies, and start the FastAPI server with:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Required service credentials and configuration should be provided through environment variables.

## Development Status

This repository is part of an ongoing engineering project exploring embedded motion sensing, data processing and machine-learning-based movement analysis.
