import pytest
from vecdb.store.idmap import IdMap

def test_add_assigns_sequential_internal_rows():
    m = IdMap()
    assert m.add("a") == 0
    assert m.add("b") == 1
    assert m.add("c") == 2
    assert len(m) == 3

def test_round_trip_external_to_internal_and_back():
    m = IdMap()
    m.add("x")
    m.add("y")
    assert m.to_internal("y") == 1
    assert m.to_external(1) == "y"

def test_duplicate_external_id_raises():
    m = IdMap()
    m.add("a")
    with pytest.raises(ValueError):
        m.add("a")

def test_unknown_external_id_raises_keyerror():
    m = IdMap()
    with pytest.raises(KeyError):
        m.to_internal("missing")
