# Bus Delay Prediction System

A complete, working Machine Learning laboratory mini-project demonstrating the end-to-end Machine Learning workflow for predicting bus transit delays using Classification.

---

## 1. Problem Statement
Public bus transit systems frequently suffer from delays caused by dynamic factors such as route distance, varying traffic density, adverse weather conditions, peak travel hours, and cascading delays from earlier stops. Predicting bus arrival status in advance helps transit operators manage schedules and passengers plan commutes efficiently.

---

## 2. Objectives
- Demonstrate a clear, reproducible end-to-end Machine Learning classification pipeline.
- Preprocess real-world bus-trip features including categorical encodings and standard scaling.
- Perform Exploratory Data Analysis (EDA) with informative dark-themed visualizations.
- Train a Random Forest Classifier using an 80/20 stratified split.
- Evaluate the model on unseen test data using Accuracy, Precision, Recall, F1-Score, and a Confusion Matrix.
- Provide an interactive, single-page, dark-themed web interface for real-time predictions.
- Persist inference logs in a local SQLite database.

---

## 3. Machine Learning Type
- **Type**: Supervised Machine Learning — **Binary Classification**
- **Target Variable**: `Delay_Status`
  - `0 = On-Time`
  - `1 = Delayed`

---

## 4. Features & Attributes
The dataset uses 5 input features representing transit and environmental conditions:

| Feature Name | Data Type | Description / Categories |
|---|---|---|
| `Distance_km` | Numerical (Float) | Total trip distance in kilometers (e.g., 3.0 to 35.0 km) |
| `Traffic_Level` | Categorical | Real-time traffic condition (`Low`, `Medium`, `High`) |
| `Weather` | Categorical | Atmospheric weather condition (`Clear`, `Cloudy`, `Rainy`, `Foggy`) |
| `Time_of_Day` | Categorical | Trip schedule window (`Morning`, `Afternoon`, `Evening`, `Night`) |
| `Previous_Delay_Minutes` | Numerical (Float) | Delay accumulated at previous stops (minutes) |
| **`Delay_Status`** | **Target (Categorical)** | **`On-Time`** or **`Delayed`** |

---

## 5. Technologies Used
- **Programming Language**: Python 3.x
- **Web Framework**: Flask 3.x
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn (RandomForestClassifier, ColumnTransformer, OneHotEncoder, StandardScaler, Pipeline)
- **Data Visualization**: Matplotlib (Custom dark theme)
- **Model Persistence**: Joblib
- **Database**: SQLite3
- **Frontend**: HTML5, CSS3 (Modern Dark Theme), JavaScript (Fetch API)
- **IDE**: Visual Studio Code (VS Code)

---

## 6. Project Structure
```text
BusDelayPrediction/
│
├── app.py                     # Flask web backend and routing
├── ml_pipeline.py             # Data generator, training, evaluation, plotting, DB
├── requirements.txt           # Python dependencies
├── README.md                  # Complete project documentation
│
├── data/
│   └── bus_delay_data.csv     # Local bus trip dataset (220+ records)
│
├── models/
│   └── bus_delay_model.pkl    # Serialized scikit-learn pipeline artifact
│
├── static/
│   ├── style.css              # Dark theme styling for entire lab interface
│   └── plots/                 # Locally generated dark-themed EDA & CM plots
│       ├── traffic_vs_delay.png
│       ├── weather_vs_delay.png
│       ├── time_vs_delay.png
│       ├── distance_vs_delay.png
│       ├── delay_distribution.png
│       └── confusion_matrix.png
│
├── templates/
│   └── index.html             # Single-page screenshot-friendly dark UI
│
└── database/
    └── predictions.db         # SQLite3 database for prediction history
```

---

## 7. Machine Learning Workflow

```
+--------------------------+
|  data/bus_delay_data.csv |
+------------+-------------+
             |
             v
+--------------------------+
|   Data Preprocessing     |  --> StandardScaler (Numerical)
| (ColumnTransformer/Pipe) |  --> OneHotEncoder (Categorical)
+------------+-------------+
             |
             v
+--------------------------+
|   80/20 Stratified Split |  --> Train: 176 samples / Test: 44 samples
+------------+-------------+
             |
             v
+--------------------------+
|  RandomForestClassifier  |  --> models/bus_delay_model.pkl
+------------+-------------+
             |
             v
+--------------------------+
|     Model Evaluation     |  --> Accuracy, Precision, Recall, F1, Confusion Matrix
+------------+-------------+
             |
             v
+--------------------------+
|   Web UI Prediction      |  --> BUS STATUS: ON-TIME / DELAYED (Confidence %)
+------------+-------------+
             |
             v
+--------------------------+
| SQLite History Logging   |  --> database/predictions.db
+--------------------------+
```

---

## 8. Steps & Application Sections

### Step 1 — Dataset Overview
Displays the total number of records (220), feature count (5), target variable (`Delay_Status`), class distribution, and a clean dark table displaying the first 10 rows of the dataset.

### Step 2 — Data Preprocessing
Demonstrates missing value checks (0 before and after), separation of numerical (`Distance_km`, `Previous_Delay_Minutes`) and categorical (`Traffic_Level`, `Weather`, `Time_of_Day`) features, and Scikit-Learn `ColumnTransformer` encoding.

### Step 3 — Exploratory Data Analysis (EDA)
Generates and displays dark-themed charts:
1. Traffic Level vs Delay Status (Primary bar chart)
2. Weather Condition vs Delay Status
3. Time of Day vs Delay Status
4. Distance vs Delay Status (Boxplot)
5. Overall Class Distribution (Target Count)

### Step 4 — Model Training
Trains a `RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)` on an 80/20 stratified split and persists the trained pipeline to `models/bus_delay_model.pkl`.

### Step 5 — Bus Delay Prediction
Provides an intuitive input form with validation (non-negative numbers, valid dropdown selections). Generates predictions and actual model probabilities:
```
BUS STATUS: ON-TIME   (with Confidence: XX.X%)
or
BUS STATUS: DELAYED   (with Confidence: XX.X%)
```

### Step 6 — Model Evaluation
Calculates live performance metrics strictly on the unseen 20% test partition:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix** with visual plot and True Positive / False Positive breakdown.

### Step 7 — Prediction History
Logs all user predictions with timestamp, confidence, and input parameters into `database/predictions.db` using SQLite3. Persists across server restarts.

---

## 9. Screenshot Guide for Laboratory Record

| Figure Label | Content to Capture | Section in Application |
|---|---|---|
| **Fig 1** | Project directory structure in VS Code file explorer | VS Code Explorer |
| **Fig 2** | Step 1: Dataset Overview & Sample Table | `#dataset` |
| **Fig 3** | Step 2: Preprocessing Summary & Architecture | `#preprocessing` |
| **Fig 4** | Step 3: EDA Plots (Traffic vs Delay, Weather, etc.) | `#eda` |
| **Fig 5** | Step 4: Model Training Details & Status Badge | `#training` |
| **Fig 6** | Step 5: Bus Delay Prediction Input Form | `#prediction` |
| **Fig 7** | Step 5: Prediction Result (`ON-TIME` / `DELAYED`) | `#prediction` (Result Card) |
| **Fig 8** | Step 6: Evaluation Metrics (Accuracy, Precision, Recall, F1) | `#evaluation` |
| **Fig 9** | Step 6: Confusion Matrix Heatmap & Breakdown | `#evaluation` (Matrix Plot) |
| **Fig 10** | Step 7: SQLite Prediction History Table | `#history` |

---

## 10. Installation & How to Run

### Step 1: Open Terminal in VS Code
Open the project folder in VS Code.

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Application
```bash
python app.py
```

### Step 4: Open in Web Browser
Navigate to:
```
http://127.0.0.1:5000
```

---

## 11. Final Laboratory Result
> “The Bus Delay Prediction system was successfully implemented using Machine Learning. The system preprocesses bus-trip data, performs exploratory data analysis, trains a Random Forest classification model, evaluates its performance using Accuracy, Precision, Recall and F1-Score, predicts whether a bus will be On-Time or Delayed, and stores prediction history using SQLite.”

---

## 12. Future Enhancements
- Integration with real-time GPS transit feeds (GTFS-RT).
- Incorporating seasonal holiday and special event transit data.
- Comparison with Gradient Boosting (XGBoost/LightGBM) models.
