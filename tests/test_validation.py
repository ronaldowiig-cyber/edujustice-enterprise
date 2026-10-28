import pytest
from umtej.engine import calculate_justice_index
from umtej.smart_umtej import SmartUMTEJEngine

def test_calculate_justice_index():
    assert calculate_justice_index([1, 1]) == 2

def test_smart_umtej_pipeline():
    engine = SmartUMTEJEngine(random_state=42)
    df = engine.generate_synthetic_cohort(n_samples=100)
    assert len(df) == 100
    assert "educational_risk" in df.columns
    
    results = engine.train_and_mitigate(df)
    assert "baseline_fairness" in results
    assert "mitigated_fairness" in results
