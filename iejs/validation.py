"""IEJS validation core (mock)."""

def validate_record(rec):
    assert isinstance(rec, dict)
    return all(k in rec for k in ("student_id", "score", "term"))
