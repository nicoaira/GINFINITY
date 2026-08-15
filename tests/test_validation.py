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


@pytest.mark.parametrize(("start", "end", "message"), [
    (1, None, "both be provided"),
    (None, 3, "both be provided"),
    (-1, 2, "invalid slice"),
    (2, 2, "invalid slice"),
    (3, 2, "invalid slice"),
    (0, 5, "invalid slice"),
])
def test_invalid_slice_is_rejected(start, end, message):
    with pytest.raises(InputValidationError, match=message):
        RNA("id", "ACGU", "....", start=start, end=end)


def test_valid_slice_is_stored():
    record = RNA("id", "ACGUACGU", "((....))", start=2, end=6)
    assert record.start == 2
    assert record.end == 6
    assert record.core_length == 4
    assert record.sliced
