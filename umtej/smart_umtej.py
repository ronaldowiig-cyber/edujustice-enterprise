import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

class SmartUMTEJEngine:
    """
    SMART UMTEJ Early Warning & Algorithmic Fairness Engine.
    Integrates Synthetic Data Generation, ML Risk Prediction, and Fairlearn Mitigation.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.base_model = RandomForestClassifier(n_estimators=50, random_state=random_state)
        self.mitigated_model = None

    def generate_synthetic_cohort(self, n_samples: int = 300) -> pd.DataFrame:
        """
        Simulate student educational justice profile (N=300 Tehran pilot benchmark).
        Sensitive attribute: socio_economic_tier (0: Under-resourced, 1: Privileged)
        Features: access_score, digital_readiness, institutional_support
        Target: educational_risk (1: High Risk, 0: Low Risk)
        """
        np.random.seed(self.random_state)
        sensitive = np.random.binomial(n=1, p=0.45, size=n_samples)
        
        # Access and digital scores correlated with socio-economic context
        access_score = np.clip(np.random.normal(loc=50 + 25 * sensitive, scale=12, size=n_samples), 10, 100)
        digital_readiness = np.clip(np.random.normal(loc=55 + 20 * sensitive, scale=15, size=n_samples), 10, 100)
        institutional_support = np.clip(np.random.normal(loc=60, scale=10, size=n_samples), 20, 100)
        
        # Ground truth risk probability influenced by structural disparity
        latent_risk = (100 - access_score) * 0.4 + (100 - digital_readiness) * 0.4 - (institutional_support * 0.2)
        risk = (latent_risk > 35).astype(int)

        df = pd.DataFrame({
            "socio_economic_tier": sensitive,
            "access_score": access_score,
            "digital_readiness": digital_readiness,
            "institutional_support": institutional_support,
            "educational_risk": risk
        })
        return df

    def evaluate_fairness(self, y_true, y_pred, sensitive_features) -> dict:
        """Computes fairness metrics via Fairlearn MetricFrame."""
        dp_diff = demographic_parity_difference(y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features)
        frame = MetricFrame(
            metrics={"selection_rate": selection_rate},
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features
        )
        return {
            "demographic_parity_difference": float(dp_diff),
            "by_group_selection_rate": frame.by_group.to_dict()["selection_rate"]
        }

    def train_and_mitigate(self, df: pd.DataFrame):
        """Trains baseline model and applies Fairlearn Exponentiated Gradient mitigation."""
        X = df[["access_score", "digital_readiness", "institutional_support"]]
        y = df["educational_risk"]
        A = df["socio_economic_tier"]

        X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
            X, y, A, test_size=0.3, random_state=self.random_state, stratify=y
        )

        # Baseline training
        self.base_model.fit(X_train, y_train)
        base_preds = self.base_model.predict(X_test)
        base_fairness = self.evaluate_fairness(y_test, base_preds, A_test)

        # Fairlearn mitigation for Demographic Parity
        mitigator = ExponentiatedGradient(
            estimator=RandomForestClassifier(n_estimators=50, random_state=self.random_state),
            constraints=DemographicParity()
        )
        mitigator.fit(X_train, y_train, sensitive_features=A_train)
        self.mitigated_model = mitigator

        mitigated_preds = self.mitigated_model.predict(X_test)
        mitigated_fairness = self.evaluate_fairness(y_test, mitigated_preds, A_test)

        return {
            "baseline_fairness": base_fairness,
            "mitigated_fairness": mitigated_fairness,
            "test_samples": len(X_test)
        }
