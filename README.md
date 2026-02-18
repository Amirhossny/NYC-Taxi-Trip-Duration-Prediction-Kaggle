# NYC Taxi Trip Duration Prediction

## Overview
This project is a machine learning pipeline designed to predict the duration of taxi trips in New York City based on trip metadata (pickup & dropoff coordinates, timestamps, passenger count, and vendor information).  
The project demonstrates a complete workflow including **data preprocessing, feature engineering, model training, evaluation, and CLI-based usage**, making it suitable as a portfolio project.

---

## Project Goal
Predict the duration of taxi trips using structured trip data while showcasing best practices in pipeline-based architecture, feature engineering, and model evaluation.  
- **Primary model:** Ridge Regression  
- **Secondary model:** Lasso Regression (optional comparison)

---

## Repository Structure
```
NYC-Taxi-Trip-Duration/
├── Data/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── Kmeans_Helper/
│   ├── __init__.py
│   ├── cluster_pair_avg.pkl
│   ├── kmeans.pkl
│   └── Train_Cluster.py
├── Models/
│   ├── ridge_model.pkl
│   └── lasso_model.pkl
├── NoteBook/
│   └── EDA.ipynb
├── src/
│   ├── __init__.py
│   ├── eval.py
│   ├── Feature_Engineering.py
│   ├── Helper_Fun.py
│   ├── logger.py
│   ├── Preprocessing.py
│   └── training.py
├── .gitignore
├── config.yaml
├── Debug.py
├── main.py
├── README.md
├── nyc_taxi.log
└── requirements.txt

```


## Data Access

The raw dataset used in this project is **not included** in the repository due to file size.  

You can download the dataset from Kaggle:

[NYC Taxi Trip Duration Dataset on Kaggle](https://www.kaggle.com/c/nyc-taxi-trip-duration/data)

After downloading, place the files in the `Data/` folder with the following structure:



## Data & Features

### Raw Data Fields
- `pickup_datetime`, `dropoff_datetime` — timestamps for trip start and end  
- `pickup_latitude`, `pickup_longitude`, `dropoff_latitude`, `dropoff_longitude` — geospatial coordinates  
- `passenger_count`, `vendor_id`, `store_and_fwd_flag` — categorical and trip metadata  
- `trip_duration` — target variable (duration in seconds)

### Feature Engineering
- **Distance-based features:** Haversine and Manhattan distances between pickup & dropoff  
- **Temporal features:** hour of day, day of week, month, rush-hour flag, is_weekend flag  
- **Cluster-based features:**  
  - KMeans clustering on pickup/dropoff coordinates  
  - Average trip duration between cluster pairs, integrated as a fully sklearn-compatible transformer  
- **Target transformation:** log-transform applied to `trip_duration` to reduce skewness  
- **Other engineered features:** `store_and_fwd_flag` and passenger count treated carefully based on data quality  

> **Note:** Outliers in `trip_duration` were removed from training only, improving distance correlation and model stability. Trips stored in vehicle memory (`store_and_fwd_flag = Y`) take slightly longer, which is kept as a feature.

---

## Model Training & Evaluation Pipeline

1. **Data Preprocessing & Feature Engineering:**  
   - Run `preprocess_data` for train/val/test sets  
   - Apply engineered features including KMeans cluster-based features  

2. **Model Training:**  
   - Ridge Regression as the primary model  
   - Lasso Regression as a secondary model for comparison  
   - Pipeline-based architecture: `Feature Engineering → Preprocessing → Model`  

3. **Evaluation:**  
   - Metrics used: **R² Score** and **RMSE**  
   - Log-transformed target handled during both training and evaluation  

**Final Performance (Test Set):**

| Model | R² Score | RMSE (seconds) |
|-------|-----------|----------------|
| Ridge | 0.6171   | 5067.93        |
| Lasso | 0.6140   | 5068.04        |

---

## CLI Usage

**Train and evaluate Ridge or Lasso:**

```bash
# Train Ridge (default)
python main.py --mode train --model ridge

# Train Lasso
python main.py --mode train --model lasso

# Test a trained model
python main.py --mode test --model ridge

# Train and Test together
python main.py --mode all --model ridge

# Optional: Train and Test Lasso together
python main.py --mode all --model lasso
```

## Installation & Setup

Clone the repository and navigate into it:

```bash
git clone https://github.com/yourusername/NYC-taxi-trip-duration.git
cd NYC-taxi-trip-duration
pip install -r requirements.txt
```

## Notes & Design Decisions

- **Ridge Regression** chosen as the primary model due to its stability with multicollinear features.
- **Lasso Regression** used optionally to evaluate feature selection effects.
- **Log transformation** applied on the target (`trip_duration`) to handle skewness.
- **KMeans cluster features** added to capture spatial trip patterns.
- **CLI-based `main.py`** enables reproducible training and testing with configurable model selection.







