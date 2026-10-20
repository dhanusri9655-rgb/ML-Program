import urllib.request
import urllib.parse
import json
import time

def run_tests():
    # Test Flask app client
    from app import app
    client = app.test_client()

    print("--- [TEST 1] Testing GET / ---")
    res = client.get('/')
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    html = res.get_data(as_text=True)
    assert "Bus Delay Prediction" in html
    assert "STEP 1" in html
    assert "STEP 2" in html
    assert "STEP 3" in html
    assert "STEP 4" in html
    assert "STEP 5" in html
    assert "STEP 6" in html
    assert "STEP 7" in html
    assert "Accuracy" in html
    print("[PASS] GET / returned 200 with all steps present.")

    print("\n--- [TEST 2] Testing POST /predict (High Delay Scenario) ---")
    payload_delayed = {
        'distance': '25.0',
        'traffic_level': 'High',
        'weather': 'Rainy',
        'time_of_day': 'Evening',
        'previous_delay': '20'
    }
    res_delayed = client.post('/predict',
                              data=json.dumps(payload_delayed),
                              content_type='application/json',
                              headers={'X-Requested-With': 'XMLHttpRequest'})
    assert res_delayed.status_code == 200
    data_delayed = res_delayed.get_json()
    print(f"Prediction result: {data_delayed['prediction']}, Confidence: {data_delayed['confidence']}%")
    assert data_delayed['success'] is True
    assert data_delayed['prediction'] in ['On-Time', 'Delayed']
    print("[PASS] Delayed test case passed.")

    print("\n--- [TEST 3] Testing POST /predict (Low Delay / On-Time Scenario) ---")
    payload_ontime = {
        'distance': '4.0',
        'traffic_level': 'Low',
        'weather': 'Clear',
        'time_of_day': 'Morning',
        'previous_delay': '0'
    }
    res_ontime = client.post('/predict',
                             data=json.dumps(payload_ontime),
                             content_type='application/json',
                             headers={'X-Requested-With': 'XMLHttpRequest'})
    assert res_ontime.status_code == 200
    data_ontime = res_ontime.get_json()
    print(f"Prediction result: {data_ontime['prediction']}, Confidence: {data_ontime['confidence']}%")
    assert data_ontime['success'] is True
    assert data_ontime['prediction'] in ['On-Time', 'Delayed']
    print("[PASS] On-Time test case passed.")

    print("\n--- [TEST 4] Testing Input Validation & Error Handling ---")
    invalid_payload = {
        'distance': '-5.0',  # Negative distance
        'traffic_level': 'High',
        'weather': 'Rainy',
        'time_of_day': 'Evening',
        'previous_delay': '10'
    }
    res_invalid = client.post('/predict',
                              data=json.dumps(invalid_payload),
                              content_type='application/json',
                              headers={'X-Requested-With': 'XMLHttpRequest'})
    assert res_invalid.status_code == 400
    data_invalid = res_invalid.get_json()
    print(f"Validation error caught properly: {data_invalid['error']}")
    assert data_invalid['success'] is False
    print("[PASS] Input validation test passed.")

    print("\n--- [TEST 5] Testing SQLite Prediction History API ---")
    res_history = client.get('/api/history')
    assert res_history.status_code == 200
    data_history = res_history.get_json()
    assert data_history['success'] is True
    assert len(data_history['history']) >= 2
    print(f"[PASS] Retrieved {len(data_history['history'])} persistent records from SQLite.")

    print("\n==========================================")
    print("ALL TESTS PASSED SUCCESSFULLY! (5/5)")
    print("==========================================")

if __name__ == '__main__':
    run_tests()
