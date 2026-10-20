import os
import sqlite3
import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash

from ml_pipeline import (
    BASE_DIR, DATA_DIR, MODEL_DIR, PLOTS_DIR, DB_DIR,
    CSV_PATH, MODEL_PATH, DB_PATH,
    generate_dataset, train_and_evaluate_model,
    init_database, add_prediction_to_db, get_prediction_history, clear_prediction_history
)

app = Flask(__name__)
app.secret_key = 'bus_delay_ml_laboratory_secret_key_2026'

# Global cache for pipeline and metrics
pipeline = None
metrics = None


def startup_ml_system():
    global pipeline, metrics
    init_database()
    if not os.path.exists(CSV_PATH) or not os.path.exists(MODEL_PATH):
        print("[*] Generating dataset and training Random Forest model...")
        pipeline, metrics = train_and_evaluate_model()
    else:
        print("[*] Loading trained model and evaluating metrics...")
        pipeline, metrics = train_and_evaluate_model()


# Initialize upon module load
startup_ml_system()


@app.route('/')
def index():
    # Load dataset sample
    df = pd.read_csv(CSV_PATH)
    dataset_sample = df.head(10).to_dict(orient='records')
    dataset_columns = list(df.columns)

    # Fetch history
    history = get_prediction_history(limit=25)

    return render_template(
        'index.html',
        dataset_sample=dataset_sample,
        dataset_columns=dataset_columns,
        metrics=metrics,
        history=history,
        prediction_result=None
    )


@app.route('/predict', methods=['POST'])
def predict():
    global pipeline, metrics
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.is_json

    try:
        # Extract inputs (support both form data and json)
        if request.is_json:
            data = request.get_json()
            distance_raw = data.get('distance')
            traffic = data.get('traffic_level')
            weather = data.get('weather')
            time_of_day = data.get('time_of_day')
            prev_delay_raw = data.get('previous_delay')
        else:
            distance_raw = request.form.get('distance')
            traffic = request.form.get('traffic_level')
            weather = request.form.get('weather')
            time_of_day = request.form.get('time_of_day')
            prev_delay_raw = request.form.get('previous_delay')

        # Input Validation
        if distance_raw is None or prev_delay_raw is None or not traffic or not weather or not time_of_day:
            raise ValueError("All input fields are required.")

        try:
            distance = float(distance_raw)
            if distance < 0:
                raise ValueError("Distance must be a non-negative number (>= 0 km).")
        except ValueError as e:
            if "non-negative" in str(e):
                raise e
            raise ValueError("Distance must be a valid numerical value.")

        try:
            prev_delay = float(prev_delay_raw)
            if prev_delay < 0:
                raise ValueError("Previous delay must be a non-negative number (>= 0 minutes).")
        except ValueError as e:
            if "non-negative" in str(e):
                raise e
            raise ValueError("Previous delay must be a valid numerical value.")

        valid_traffic = ['Low', 'Medium', 'High']
        valid_weather = ['Clear', 'Cloudy', 'Rainy', 'Foggy']
        valid_time = ['Morning', 'Afternoon', 'Evening', 'Night']

        if traffic not in valid_traffic:
            raise ValueError(f"Invalid Traffic Level: '{traffic}'. Must be one of {valid_traffic}.")
        if weather not in valid_weather:
            raise ValueError(f"Invalid Weather Condition: '{weather}'. Must be one of {valid_weather}.")
        if time_of_day not in valid_time:
            raise ValueError(f"Invalid Time of Day: '{time_of_day}'. Must be one of {valid_time}.")

        # Ensure model is ready
        if pipeline is None:
            pipeline = joblib.load(MODEL_PATH)

        # Prepare DataFrame for scikit-learn pipeline
        input_data = pd.DataFrame([{
            'Distance_km': distance,
            'Traffic_Level': traffic,
            'Weather': weather,
            'Time_of_Day': time_of_day,
            'Previous_Delay_Minutes': prev_delay
        }])

        # Perform actual prediction & confidence estimation
        pred_label = pipeline.predict(input_data)[0]
        pred_probas = pipeline.predict_proba(input_data)[0]
        classes = list(pipeline.classes_)

        # Extract probability for the predicted class
        class_idx = classes.index(pred_label)
        confidence = round(float(pred_probas[class_idx]) * 100, 1)

        # Store in SQLite database
        record_id = add_prediction_to_db(
            distance=distance,
            traffic=traffic,
            weather=weather,
            time_of_day=time_of_day,
            prev_delay=prev_delay,
            prediction=pred_label,
            confidence=confidence
        )

        result_payload = {
            'success': True,
            'id': record_id,
            'prediction': pred_label,
            'confidence': confidence,
            'inputs': {
                'distance': distance,
                'traffic_level': traffic,
                'weather': weather,
                'time_of_day': time_of_day,
                'previous_delay': prev_delay
            }
        }

        if is_ajax:
            return jsonify(result_payload)

        # Form submission fallback
        df = pd.read_csv(CSV_PATH)
        dataset_sample = df.head(10).to_dict(orient='records')
        dataset_columns = list(df.columns)
        history = get_prediction_history(limit=25)

        return render_template(
            'index.html',
            dataset_sample=dataset_sample,
            dataset_columns=dataset_columns,
            metrics=metrics,
            history=history,
            prediction_result=result_payload
        )

    except Exception as e:
        error_msg = str(e)
        if is_ajax:
            return jsonify({'success': False, 'error': error_msg}), 400

        flash(error_msg, 'danger')
        return redirect(url_for('index', _anchor='prediction'))


@app.route('/api/history', methods=['GET'])
def api_history():
    history = get_prediction_history(limit=50)
    return jsonify({'success': True, 'history': history})


@app.route('/api/clear-history', methods=['POST'])
def api_clear_history():
    clear_prediction_history()
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.is_json
    if is_ajax:
        return jsonify({'success': True, 'message': 'Prediction history cleared.'})
    flash('Prediction history cleared successfully.', 'info')
    return redirect(url_for('index', _anchor='history'))


if __name__ == '__main__':
    print("[*] Starting Bus Delay Prediction Flask Application on http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False)
