"""UMTEJ educational justice engine (mock)."""

def compute_equity(scores):
    return sum(scores)/len(scores) if scores else 0.0
