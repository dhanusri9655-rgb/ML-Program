import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
PLOTS_DIR = os.path.join(BASE_DIR, 'static', 'plots')
DB_DIR = os.path.join(BASE_DIR, 'database')

CSV_PATH = os.path.join(DATA_DIR, 'bus_delay_data.csv')
MODEL_PATH = os.path.join(MODEL_DIR, 'bus_delay_model.pkl')
DB_PATH = os.path.join(DB_DIR, 'predictions.db')


def generate_dataset(num_records=220, seed=42):
    """Generate realistic bus-trip dataset with at least 150 records."""
    np.random.seed(seed)
    os.makedirs(DATA_DIR, exist_ok=True)

    traffic_choices = ['Low', 'Medium', 'High']
    weather_choices = ['Clear', 'Cloudy', 'Rainy', 'Foggy']
    time_choices = ['Morning', 'Afternoon', 'Evening', 'Night']

    traffic_probs = [0.35, 0.40, 0.25]
    weather_probs = [0.45, 0.25, 0.20, 0.10]
    time_probs = [0.30, 0.25, 0.35, 0.10]

    traffic = np.random.choice(traffic_choices, size=num_records, p=traffic_probs)
    weather = np.random.choice(weather_choices, size=num_records, p=weather_probs)
    time_of_day = np.random.choice(time_choices, size=num_records, p=time_probs)

    distance = np.round(np.random.uniform(3.0, 35.0, size=num_records), 1)

    # Previous delay: mostly 0-10, some higher
    prev_delay = np.random.exponential(scale=6.0, size=num_records)
    prev_delay = np.round(np.clip(prev_delay, 0, 45), 0).astype(int)

    # Realistic logic for Delay Status
    delay_scores = []
    for i in range(num_records):
        score = 0.0

        # Traffic impact
        if traffic[i] == 'High':
            score += 2.8
        elif traffic[i] == 'Medium':
            score += 1.2
        else:
            score += 0.0

        # Weather impact
        if weather[i] == 'Rainy':
            score += 2.2
        elif weather[i] == 'Foggy':
            score += 1.8
        elif weather[i] == 'Cloudy':
            score += 0.5
        else:
            score += 0.0

        # Time of day impact
        if time_of_day[i] == 'Evening':
            score += 2.0
        elif time_of_day[i] == 'Morning':
            score += 1.6
        elif time_of_day[i] == 'Afternoon':
            score += 0.6
        else:
            score += 0.0

        # Distance impact (longer routes have higher risk of delays)
        score += (distance[i] / 10.0) * 0.9

        # Previous delay impact
        score += (prev_delay[i] / 8.0) * 1.4

        # Add slight natural randomness
        score += np.random.normal(0, 0.9)
        delay_scores.append(score)

    delay_scores = np.array(delay_scores)
    # Threshold chosen to balance classes nicely (~45% Delayed, ~55% On-Time)
    threshold = np.percentile(delay_scores, 55)
    delay_status = np.where(delay_scores >= threshold, 'Delayed', 'On-Time')

    df = pd.DataFrame({
        'Distance_km': distance,
        'Traffic_Level': traffic,
        'Weather': weather,
        'Time_of_Day': time_of_day,
        'Previous_Delay_Minutes': prev_delay,
        'Delay_Status': delay_status
    })

    df.to_csv(CSV_PATH, index=False)
    print(f"[OK] Dataset successfully created with {len(df)} records at: {CSV_PATH}")
    return df


def style_dark_axis(ax, title="", xlabel="", ylabel=""):
    """Helper to apply consistent dark theme styling to Matplotlib axes."""
    ax.set_facecolor('#1e293b')
    ax.tick_params(colors='#e2e8f0', labelsize=10)
    for spine in ax.spines.values():
        spine.set_color('#475569')
        spine.set_linewidth(1.2)
    ax.grid(True, linestyle='--', alpha=0.25, color='#94a3b8')
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold', color='#f8fafc', pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11, fontweight='600', color='#cbd5e1', labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, fontweight='600', color='#cbd5e1', labelpad=8)


def generate_eda_plots(df):
    """Generate dark-themed EDA plots for Step 3."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

    # Color palette
    colors = {'On-Time': '#38bdf8', 'Delayed': '#f87171'}

    # 1. Traffic Level vs Delay Status (Primary Important Graph)
    fig, ax = plt.subplots(figsize=(6, 4.2), facecolor='#0f172a')
    order = ['Low', 'Medium', 'High']
    cross_tab = pd.crosstab(df['Traffic_Level'], df['Delay_Status']).reindex(order)
    cross_tab.plot(kind='bar', stacked=False, color=[colors.get(c, '#94a3b8') for c in cross_tab.columns],
                   edgecolor='#0f172a', linewidth=1.5, ax=ax, width=0.6)
    style_dark_axis(ax, title='Traffic Level vs Bus Delay Status', xlabel='Traffic Level', ylabel='Number of Trips')
    ax.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='#f8fafc', title='Delay Status', title_fontsize=9)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'traffic_vs_delay.png'), dpi=160, facecolor=fig.get_facecolor())
    plt.close()

    # 2. Weather Condition vs Delay Status
    fig, ax = plt.subplots(figsize=(6, 4.2), facecolor='#0f172a')
    weather_order = ['Clear', 'Cloudy', 'Rainy', 'Foggy']
    cross_weather = pd.crosstab(df['Weather'], df['Delay_Status']).reindex(weather_order)
    cross_weather.plot(kind='bar', stacked=False, color=[colors.get(c, '#94a3b8') for c in cross_weather.columns],
                       edgecolor='#0f172a', linewidth=1.5, ax=ax, width=0.6)
    style_dark_axis(ax, title='Weather Condition vs Bus Delay Status', xlabel='Weather Condition', ylabel='Number of Trips')
    ax.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='#f8fafc', title='Delay Status', title_fontsize=9)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'weather_vs_delay.png'), dpi=160, facecolor=fig.get_facecolor())
    plt.close()

    # 3. Time of Day vs Delay Status
    fig, ax = plt.subplots(figsize=(6, 4.2), facecolor='#0f172a')
    time_order = ['Morning', 'Afternoon', 'Evening', 'Night']
    cross_time = pd.crosstab(df['Time_of_Day'], df['Delay_Status']).reindex(time_order)
    cross_time.plot(kind='bar', stacked=False, color=[colors.get(c, '#94a3b8') for c in cross_time.columns],
                    edgecolor='#0f172a', linewidth=1.5, ax=ax, width=0.6)
    style_dark_axis(ax, title='Time of Day vs Bus Delay Status', xlabel='Time of Day', ylabel='Number of Trips')
    ax.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='#f8fafc', title='Delay Status', title_fontsize=9)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'time_vs_delay.png'), dpi=160, facecolor=fig.get_facecolor())
    plt.close()

    # 4. Distance vs Delay Status (Grouped Boxplot / Distribution)
    fig, ax = plt.subplots(figsize=(6, 4.2), facecolor='#0f172a')
    data_ontime = df[df['Delay_Status'] == 'On-Time']['Distance_km']
    data_delayed = df[df['Delay_Status'] == 'Delayed']['Distance_km']
    box = ax.boxplot([data_ontime, data_delayed], tick_labels=['On-Time', 'Delayed'], patch_artist=True,
                     widths=0.45, medianprops=dict(color='#ffffff', linewidth=2))
    box['boxes'][0].set(facecolor='#0284c7', edgecolor='#38bdf8', linewidth=1.5)
    box['boxes'][1].set(facecolor='#dc2626', edgecolor='#f87171', linewidth=1.5)
    for whisker in box['whiskers']:
        whisker.set(color='#94a3b8', linewidth=1.2)
    for cap in box['caps']:
        cap.set(color='#94a3b8', linewidth=1.2)
    style_dark_axis(ax, title='Trip Distance (km) vs Delay Status', xlabel='Bus Delay Status', ylabel='Distance (km)')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'distance_vs_delay.png'), dpi=160, facecolor=fig.get_facecolor())
    plt.close()

    # 5. Overall Class Distribution (Target variable count)
    fig, ax = plt.subplots(figsize=(6, 4.2), facecolor='#0f172a')
    status_counts = df['Delay_Status'].value_counts()
    bar_colors = [colors.get(k, '#38bdf8') for k in status_counts.index]
    bars = ax.bar(status_counts.index, status_counts.values, color=bar_colors, edgecolor='#0f172a', width=0.45)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1.5, f'{height}',
                ha='center', va='bottom', color='#f8fafc', fontweight='bold', fontsize=11)
    style_dark_axis(ax, title='Target Class Distribution (Delay_Status)', xlabel='Class Label', ylabel='Record Count')
    ax.set_ylim(0, max(status_counts.values) * 1.18)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'delay_distribution.png'), dpi=160, facecolor=fig.get_facecolor())
    plt.close()

    print("[OK] All 5 dark-themed EDA plots generated successfully.")


def generate_confusion_matrix_plot(cm, labels=['On-Time', 'Delayed']):
    """Generate a clean dark-themed confusion matrix plot."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.2), facecolor='#0f172a')
    ax.set_facecolor('#1e293b')

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'dark_blue_cyan', ['#0f172a', '#1e3a8a', '#0284c7', '#38bdf8']
    )
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='#94a3b8')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#e2e8f0')

    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(labels, color='#e2e8f0', fontsize=11, fontweight='600')
    ax.set_yticklabels(labels, color='#e2e8f0', fontsize=11, fontweight='600')

    # Values in matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = '#ffffff' if val > thresh else '#e2e8f0'
            ax.text(j, i, format(val, 'd'),
                    ha="center", va="center",
                    color=color, fontsize=14, fontweight='bold')

    style_dark_axis(ax, title='Test Set Confusion Matrix', xlabel='Predicted Label', ylabel='Actual True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'), dpi=160, facecolor=fig.get_facecolor())
    plt.close()
    print("[OK] Dark-themed confusion matrix plot saved.")


def train_and_evaluate_model():
    """Execute complete ML pipeline: data loading, preprocessing, RF training, evaluation, model persistence."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if not os.path.exists(CSV_PATH):
        df = generate_dataset()
    else:
        df = pd.read_csv(CSV_PATH)

    generate_eda_plots(df)

    # Features and Target
    feature_cols = ['Distance_km', 'Traffic_Level', 'Weather', 'Time_of_Day', 'Previous_Delay_Minutes']
    X = df[feature_cols]
    y = df['Delay_Status']

    # Preprocessing pipelines
    num_cols = ['Distance_km', 'Previous_Delay_Minutes']
    cat_cols = ['Traffic_Level', 'Weather', 'Time_of_Day']

    num_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ]
    )

    # Full Model Pipeline with Random Forest Classifier
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', rf_classifier)
    ])

    # 80% Training, 20% Testing with Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Model
    pipeline.fit(X_train, y_train)

    # Save Pipeline
    joblib.dump(pipeline, MODEL_PATH)
    print(f"[OK] Model pipeline saved to {MODEL_PATH}")

    # Evaluate on Test Set
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label='Delayed')
    rec = recall_score(y_test, y_pred, pos_label='Delayed')
    f1 = f1_score(y_test, y_pred, pos_label='Delayed')

    labels = ['On-Time', 'Delayed']
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    generate_confusion_matrix_plot(cm, labels)

    # Compile evaluation and pipeline metadata
    metrics = {
        'total_records': len(df),
        'total_features': len(feature_cols),
        'feature_names': feature_cols,
        'target_variable': 'Delay_Status',
        'classes': labels,
        'class_distribution': df['Delay_Status'].value_counts().to_dict(),
        'missing_before': int(df.isnull().sum().sum()),
        'missing_after': 0,
        'numerical_features': num_cols,
        'categorical_features': cat_cols,
        'encoding_method': 'One-Hot Encoding (OneHotEncoder) & Standardization (StandardScaler)',
        'model_name': 'Random Forest Classifier',
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'train_ratio': '80%',
        'test_ratio': '20%',
        'accuracy': round(acc * 100, 2),
        'precision': round(prec * 100, 2),
        'recall': round(rec * 100, 2),
        'f1_score': round(f1 * 100, 2),
        'confusion_matrix': {
            'tn': int(cm[0][0]),  # True On-Time
            'fp': int(cm[0][1]),  # False Delayed
            'fn': int(cm[1][0]),  # False On-Time
            'tp': int(cm[1][1])   # True Delayed
        }
    }

    return pipeline, metrics


def init_database():
    """Initialize SQLite database for prediction history."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            distance REAL NOT NULL,
            traffic_level TEXT NOT NULL,
            weather TEXT NOT NULL,
            time_of_day TEXT NOT NULL,
            previous_delay REAL NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    print(f"[OK] Database initialized at {DB_PATH}")


def add_prediction_to_db(distance, traffic, weather, time_of_day, prev_delay, prediction, confidence):
    """Insert a prediction record into the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO prediction_history (distance, traffic_level, weather, time_of_day, previous_delay, prediction, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (float(distance), traffic, weather, time_of_day, float(prev_delay), prediction, float(confidence)))
    conn.commit()
    record_id = cursor.lastrowid
    conn.close()
    return record_id


def get_prediction_history(limit=50):
    """Fetch prediction history ordered by newest first."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM prediction_history ORDER BY id DESC LIMIT ?', (limit,))
    rows = cursor.fetchall()
    history = [dict(row) for row in rows]
    conn.close()
    return history


def clear_prediction_history():
    """Clear all records from prediction history."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM prediction_history')
    conn.commit()
    conn.close()


if __name__ == '__main__':
    print("Executing ML Pipeline...")
    init_database()
    pipeline, metrics = train_and_evaluate_model()
    print("ML Pipeline Execution Finished Successfully.")
    print(f"Accuracy: {metrics['accuracy']}%")
    print(f"Precision: {metrics['precision']}%")
    print(f"Recall: {metrics['recall']}%")
    print(f"F1-Score: {metrics['f1_score']}%")
