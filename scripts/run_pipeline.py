import sys
from umtej.engine import calculate_justice_index
from umtej.smart_umtej import SmartUMTEJEngine

def main():
    print("=== [1/2] Running Classic UMTEJ Index ===")
    mock_weights = [1.0, 1.0]
    score = calculate_justice_index(mock_weights)
    print(f"Classic Index Score: {score}")

    print("\n=== [2/2] Running SMART UMTEJ Fairness & AI Pipeline ===")
    engine = SmartUMTEJEngine(random_state=42)
    cohort = engine.generate_synthetic_cohort(n_samples=300)
    print(f"Generated Cohort Shape: {cohort.shape}")

    results = engine.train_and_mitigate(cohort)
    print(f"Baseline Demographic Parity Diff: {results['baseline_fairness']['demographic_parity_difference']:.4f}")
    print(f"Mitigated Demographic Parity Diff: {results['mitigated_fairness']['demographic_parity_difference']:.4f}")
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
