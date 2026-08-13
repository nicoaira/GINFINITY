"""Strict public input contract."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


class InputValidationError(ValueError):
    """An RNA record is outside the supported input contract."""


@dataclass(frozen=True, slots=True)
class RNA:
    """One RNA sequence and its caller-supplied secondary structure."""

    identifier: str
    sequence: str
    structure: str

    def __post_init__(self) -> None:
        identifier = self.identifier.strip()
        sequence, structure = validate_and_normalize(
            self.sequence, self.structure)
        if not identifier:
            raise InputValidationError("empty identifier")
        if any(char in identifier for char in "\t\r\n"):
            raise InputValidationError(
                "identifier must not contain tabs or line breaks")
        object.__setattr__(self, "identifier", identifier)
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(self, "structure", structure)

    @property
    def length(self) -> int:
        return len(self.sequence)

    @classmethod
    def from_mapping(
        cls,
        row: Mapping[str, object],
        *,
        identifier_column: str = "transcript_id",
        sequence_column: str = "sequence",
        structure_column: str = "secondary_structure",
    ) -> "RNA":
        """Create an RNA from a row with configurable column names."""
        columns = (identifier_column, sequence_column, structure_column)
        if len(set(columns)) != 3:
            raise ValueError("identifier, sequence, and structure columns must differ")
        missing = [column for column in columns if column not in row]
        if missing:
            raise InputValidationError(
                "missing RNA column(s): " + ", ".join(missing))
        values = [row[column] for column in columns]
        if not all(isinstance(value, str) for value in values):
            raise InputValidationError("RNA identifier, sequence, and structure must be strings")
        return cls(values[0], values[1], values[2])


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
