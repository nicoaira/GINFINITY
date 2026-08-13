import pytest

from ginfinity import InputValidationError, RNA


def test_record_normalizes_t_and_whitespace():
    record = RNA(" id ", " acgt ", " (()) ")
    assert record.identifier == "id"
    assert record.sequence == "ACGU"
    assert record.structure == "(())"


@pytest.mark.parametrize(("sequence", "structure", "message"), [
    ("", "", "empty sequence"),
    ("ACGN", "....", "unsupported sequence"),
    ("ACGU", "...", "characters against"),
    ("ACGU", "[..]", "unsupported structure"),
    ("ACGU", "((.)", "unmatched"),
])
def test_bad_input_is_rejected(sequence, structure, message):
    with pytest.raises(InputValidationError, match=message):
        RNA("id", sequence, structure)


def test_bad_identifier_is_rejected():
    with pytest.raises(InputValidationError, match="identifier"):
        RNA("bad\tid", "ACGU", "....")
