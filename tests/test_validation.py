from iejs.validation import validate_record

def test_ok():
    assert validate_record({"student_id": 1, "score": 10, "term": 1})
