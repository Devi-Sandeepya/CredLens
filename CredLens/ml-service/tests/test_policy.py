from src.app import policy

def test_approve():
    assert policy(0.20, 80, "NORMAL") == "APPROVE"

def test_refer_on_unusual():
    assert policy(0.20, 90, "UNUSUAL") == "REFER"

def test_decline():
    assert policy(0.70, 80, "NORMAL") == "DECLINE"
