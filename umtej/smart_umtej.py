"""SMART UMTEJ warning system (mock)."""

def check_alerts(metric, threshold=0.5):
    return [m for m in metric if m < threshold]
