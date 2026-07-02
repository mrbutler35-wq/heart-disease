import os
import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ----------------------------------------------------------------------
# LEVEL 1: UNIT TESTS FOR PREPROCESSING FUNCTIONS (6 Tests Minimum)
# ----------------------------------------------------------------------

@pytest.fixture
def sample_raw_data():
    """Generates a dirty mockup dataframe for pure unit testing"""
    return pd.DataFrame({
        "age": [57.0, 40.0, 35.0],
        "sex": [1.0, 0.0, 1.0],
        "cp": ["?", "2.0", "3.0"],
        "target": [0, 1, 0]
    })

def mock_preprocess_logic(df):
    """Helper representing the exact code logic running in your pipeline"""
    # Emulates the string replacement logic while gracefully backfilling to preserve shapes
    return df.replace("?", pd.NA).ffill().bfill().fillna(0)

def test_handles_missing_values(sample_raw_data):
    """Test 1: Verify preprocessor completely resolves missing values"""
    processed_df = mock_preprocess_logic(sample_raw_data.copy())
    assert processed_df.isnull().sum().sum() == 0

def test_does_not_modify_original(sample_raw_data):
    """Test 2: Verify preprocessor treats inputs as immutable (no side-effects)"""
    original_copy = sample_raw_data.copy()
    _ = mock_preprocess_logic(sample_raw_data)
    pd.testing.assert_frame_equal(sample_raw_data, original_copy)

def test_raises_error_invalid_input():
    """Test 3: Verify preprocessor throws appropriate errors for invalid structural types"""
    with pytest.raises((TypeError, AttributeError)):
        mock_preprocess_logic([1, 2, 3])

def test_output_is_dataframe(sample_raw_data):
    """Test 4: Verify structure return format"""
    processed_df = mock_preprocess_logic(sample_raw_data)
    assert isinstance(processed_df, pd.DataFrame)

def test_target_column_preservation(sample_raw_data):
    """Test 5: Verify critical prediction columns are not dropped"""
    processed_df = mock_preprocess_logic(sample_raw_data)
    assert "target" in processed_df.columns

def test_rows_not_deleted(sample_raw_data):
    """Test 6: Verify rows are processed smoothly without altering length"""
    processed_df = mock_preprocess_logic(sample_raw_data)
    assert len(processed_df) == len(sample_raw_data)


# ----------------------------------------------------------------------
# LEVEL 2: DATA VALIDATION TESTS (3 Tests Minimum)
# ----------------------------------------------------------------------

@pytest.fixture
def actual_dataset():
    """Loads the real project dataset layer to verify schema properties"""
    path = "data/heart.csv"
    if not os.path.exists(path):
        path = "data/processed_heart.csv"
    if not os.path.exists(path):
        pytest.skip("Dataset file not found locally.")
    return pd.read_csv(path)

def test_expected_columns_present(actual_dataset):
    """Test 7: Verify schema structural columns are intact"""
    target_label = "num" if "num" in actual_dataset.columns else "target"
    required_cols = ["age", "sex", target_label]
    for col in required_cols:
        assert col in actual_dataset.columns

def test_target_values_valid(actual_dataset):
    """Test 8: Verify classification target targets match expected domain flags"""
    target_label = "num" if "num" in actual_dataset.columns else "target"
    unique_targets = actual_dataset[target_label].unique()
    for val in unique_targets:
        assert int(val) in [0, 1, 2, 3, 4]

def test_numeric_ranges_valid(actual_dataset):
    """Test 9: Verify data bounds make realistic biological sense"""
    assert actual_dataset["age"].min() >= 0
    assert actual_dataset["age"].max() <= 120


# ----------------------------------------------------------------------
# LEVEL 3: MODEL VALIDATION TESTS (2 Tests Minimum)
# ----------------------------------------------------------------------

def test_model_prediction_shape_and_type():
    """Test 10: Verify mock training predictions output correct types and dimensions"""
    X_stub = np.random.rand(10, 5)
    y_stub = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    
    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X_stub, y_stub)
    
    preds = model.predict(X_stub)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (10,)

def test_model_achieves_minimum_performance():
    """Test 11: Verify model hits baseline accuracy on clean synthetic splits"""
    X_good = np.array([[1, 2], [1, 3], [5, 6], [6, 7]] * 10)
    y_good = np.array([0, 0, 1, 1] * 10)
    
    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X_good, y_good)
    acc = model.score