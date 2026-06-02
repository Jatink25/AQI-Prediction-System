# AQI Prediction System

An Arduino-based Air Quality Monitoring and Prediction System that measures environmental conditions and estimates Air Quality Index (AQI) in real time. The project combines embedded systems with machine learning to predict future AQI trends using historical sensor data.

## Features

- Real-time AQI monitoring
- Temperature and humidity sensing
- Gas sensor data processing
- AQI calculation
- Machine Learning based AQI prediction
- Data cleaning and preprocessing pipeline
- Model analysis and visualization tools
```
## Project Structure

AQI-Prediction-System
│
├── Arduino
│   └── phase3code_final.ino
│
├── Dataset
│   └── cleaned_data.csv
│
├── ML model
│   ├── data_cleaner.py
│   ├── model4_0.py
│   └── model_graph.py
│
├── License
│
└── README.md
```
## Technologies Used

- Arduino IDE
- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib

## Machine Learning Workflow

1. Collect sensor data.
2. Clean and preprocess dataset.
3. Generate lag-based AQI features.
4. Train Linear Regression model.
5. Evaluate model performance.
6. Export coefficients for Arduino deployment.

## Author

Jatin Kumar
