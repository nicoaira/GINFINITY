"""Strict public input contract."""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Mapping, Sequence


class InputValidationError(ValueError):
    """An RNA record is outside the supported input contract."""


def _as_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise InputValidationError(f"{name} must be an integer")
    return int(value)


def parse_position_list(value: object, *, name: str) -> list[int]:
    """Parse a single integer or a comma-separated integer list."""
    if value is None:
        return []
    if isinstance(value, bool):
        raise InputValidationError(f"{name} must be an integer")
    if isinstance(value, Integral):
        return [int(value)]
    if isinstance(value, float) and value.is_integer():
        return [int(value)]
    if not isinstance(value, str):
        raise InputValidationError(f"{name} must be an integer or integer list")
    text = value.strip()
    if not text:
        return []
    parts = [part.strip() for part in text.split(",")]
    if any(not part for part in parts):
        raise InputValidationError(f"empty value in {name} list")
    positions: list[int] = []
    for part in parts:
        try:
            positions.append(int(part, 10))
        except ValueError as error:
            raise InputValidationError(
                f"invalid integer {part!r} in {name}") from error
    return positions


def parse_slice_bounds(
    start: object,
    end: object,
) -> list[tuple[int, int]]:
    """Return 0-based half-open windows from parallel start/end values."""
    starts = parse_position_list(start, name="start")
    ends = parse_position_list(end, name="end")
    if not starts and not ends:
        return []
    if len(starts) != len(ends):
        raise InputValidationError(
            f"start has {len(starts)} value(s) but end has {len(ends)}")
    return list(zip(starts, ends))


def sliced_identifier(identifier: str, start: int, end: int) -> str:
    """Return a unique identifier for one window of a source molecule."""
    return f"{identifier}:{start}-{end}"


def _required_string_columns(
    row: Mapping[str, object],
    columns: Sequence[str],
) -> list[str]:
    missing = [column for column in columns if column not in row]
    if missing:
        raise InputValidationError(
            "missing RNA column(s): " + ", ".join(missing))
    values = [row[column] for column in columns]
    if not all(isinstance(value, str) for value in values):
        raise InputValidationError(
            "RNA identifier, sequence, and structure must be strings")
    return [str(value) for value in values]


def _validate_column_names(*columns: str | None) -> tuple[str, ...]:
    present = tuple(column for column in columns if column)
    if any(not column for column in present):
        raise ValueError("column names cannot be empty")
    if len(set(present)) != len(present):
        raise ValueError("RNA column names must differ")
    return present


@dataclass(frozen=True, slots=True)
class RNA:
    """One RNA sequence, its secondary structure, and an optional window.

    ``start`` and ``end`` are 0-based half-open coordinates into the
    normalized sequence, the same convention as ``sequence[start:end]``.
    Both must be omitted for a full-molecule record.
    """

    identifier: str
    sequence: str
    structure: str
    start: int | None = None
    end: int | None = None

    def __post_init__(self) -> None:
        identifier = self.identifier.strip()
        sequence, structure = validate_and_normalize(
            self.sequence, self.structure)
        if not identifier:
            raise InputValidationError("empty identifier")
        if any(char in identifier for char in "\t\r\n"):
            raise InputValidationError(
                "identifier must not contain tabs or line breaks")
        start = self.start
        end = self.end
        if (start is None) != (end is None):
            raise InputValidationError("start and end must both be provided")
        if start is not None and end is not None:
            start = _as_int(start, name="start")
            end = _as_int(end, name="end")
            length = len(sequence)
            if start < 0 or end > length or start >= end:
                raise InputValidationError(
                    f"invalid slice [{start}, {end}) for a {length} nt sequence")
        object.__setattr__(self, "identifier", identifier)
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(self, "structure", structure)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    @property
    def length(self) -> int:
        return len(self.sequence)

    @property
    def sliced(self) -> bool:
        return self.start is not None

    @property
    def core_length(self) -> int:
        if self.start is None or self.end is None:
            return self.length
        return self.end - self.start

    @classmethod
    def from_mapping(
        cls,
        row: Mapping[str, object],
        *,
        identifier_column: str = "transcript_id",
        sequence_column: str = "sequence",
        structure_column: str = "secondary_structure",
        start_column: str | None = None,
        end_column: str | None = None,
    ) -> "RNA":
        """Create one RNA from a mapping. Multiple windows must use
        ``many_from_mapping``."""
        records = cls.many_from_mapping(
            row,
            identifier_column=identifier_column,
            sequence_column=sequence_column,
            structure_column=structure_column,
            start_column=start_column,
            end_column=end_column,
            suffix_identifier=False,
        )
        if len(records) != 1:
            raise InputValidationError(
                "mapping defines multiple slices; use RNA.many_from_mapping()")
        return records[0]

    @classmethod
    def many_from_mapping(
        cls,
        row: Mapping[str, object],
        *,
        identifier_column: str = "transcript_id",
        sequence_column: str = "sequence",
        structure_column: str = "secondary_structure",
        start_column: str | None = "start",
        end_column: str | None = "end",
        suffix_identifier: bool = True,
    ) -> list["RNA"]:
        """Create one RNA per window listed in a mapping.

        Absent or empty start/end columns yield a single full-molecule
        record. Parallel comma-separated lists yield one record per
        window. When more than one window is present, identifiers are
        always suffixed as ``{id}:{start}-{end}``.
        """
        if (start_column is None) != (end_column is None):
            raise ValueError("start and end columns must both be provided")
        _validate_column_names(
            identifier_column, sequence_column, structure_column,
            start_column, end_column)
        identifier, sequence, structure = _required_string_columns(
            row, (identifier_column, sequence_column, structure_column))
        windows: list[tuple[int, int]] = []
        if start_column is not None and end_column is not None:
            if start_column in row or end_column in row:
                if start_column not in row or end_column not in row:
                    missing = [column for column in (start_column, end_column)
                               if column not in row]
                    raise InputValidationError(
                        "missing RNA column(s): " + ", ".join(missing))
                windows = parse_slice_bounds(row[start_column], row[end_column])
        if not windows:
            return [cls(identifier, sequence, structure)]
        use_suffix = suffix_identifier or len(windows) > 1
        return [
            cls(
                sliced_identifier(identifier, start, end) if use_suffix
                else identifier,
                sequence,
                structure,
                start=start,
                end=end,
            )
            for start, end in windows
        ]


def validate_and_normalize(sequence: str, structure: str,
                           *, maximum_length: int = 4096) -> tuple[str, str]:
    sequence = sequence.strip().upper().replace("T", "U")
    structure = structure.strip()
    if not sequence:
        raise InputValidationError("empty sequence")
    if len(sequence) > maximum_length:
        raise InputValidationError(
            f"sequence length {len(sequence)} exceeds maximum {maximum_length}")
    if len(sequence) != len(structure):
        raise InputValidationError(
            f"structure is {len(structure)} characters against a "
            f"{len(sequence)} nt sequence")
    invalid_bases = sorted(set(sequence) - set("ACGU"))
    if invalid_bases:
        raise InputValidationError(
            "unsupported sequence character(s): " + " ".join(invalid_bases))
    invalid_structure = sorted(set(structure) - set(".()"))
    if invalid_structure:
        raise InputValidationError(
            "unsupported structure character(s): "
            + " ".join(invalid_structure))
    stack: list[int] = []
    for index, char in enumerate(structure):
        if char == "(":
            stack.append(index)
        elif char == ")":
            if not stack:
                raise InputValidationError(
                    f"unmatched ')' at 0-based position {index}")
            stack.pop()
    if stack:
        raise InputValidationError(
            f"unmatched '(' at 0-based position {stack[0]}")
    return sequence, structure
