# Diabetes Prediction - MLOps SP26 Assignment 1

A complete machine learning pipeline for diabetes prediction.

## Project Structure
- `data_model.ipynb` - EDA, data cleaning & model training
- `app.py` - FastAPI application
- `diabetes_model.pkl` - Saved best model
- `training_columns.pkl` - Saved training column names
- `requirements.txt` - Python dependencies
- `screenshots/` - API test screenshots

## Setup Instructions
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run FastAPI Server
```bash
uvicorn app:app --reload
```

## API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | Health check |
| POST | /predict | Predict diabetes |

## Model Performance
| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Decision Tree | 98.35% | 0.9832 |
| Random Forest | 98.35% | 0.9830 |
| Logistic Regression | 93.07% | 0.9251 |
| KNN | 87.79% | 0.8782 |
| SVM | 84.49% | 0.7738 |

**Best Model: Decision Tree Classifier**
