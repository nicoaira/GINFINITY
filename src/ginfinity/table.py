"""Delimited RNA table input with configurable column names."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import TextIO

from ._validation import InputValidationError, RNA


def _validate_columns(
    identifier_column: str,
    sequence_column: str,
    structure_column: str,
) -> tuple[str, str, str]:
    columns = (identifier_column, sequence_column, structure_column)
    if any(not column for column in columns):
        raise ValueError("column names cannot be empty")
    if len(set(columns)) != 3:
        raise ValueError("identifier, sequence, and structure columns must differ")
    return columns


def _read_handle(
    handle: TextIO,
    *,
    source: str,
    identifier_column: str,
    sequence_column: str,
    structure_column: str,
    delimiter: str,
) -> list[RNA]:
    if len(delimiter) != 1:
        raise ValueError("delimiter must be exactly one character")
    columns = _validate_columns(
        identifier_column, sequence_column, structure_column)
    reader = csv.DictReader(handle, delimiter=delimiter)
    if reader.fieldnames is None:
        raise ValueError(f"empty RNA table: {source}")
    if len(set(reader.fieldnames)) != len(reader.fieldnames):
        raise ValueError(f"duplicate column name in RNA table: {source}")
    missing = [column for column in columns if column not in reader.fieldnames]
    if missing:
        raise ValueError(
            f"RNA table {source} is missing column(s): " + ", ".join(missing))
    records: list[RNA] = []
    identifiers: set[str] = set()
    for row in reader:
        if None in row:
            raise ValueError(
                f"RNA table {source} line {reader.line_num} has extra fields")
        try:
            record = RNA.from_mapping(
                row,
                identifier_column=identifier_column,
                sequence_column=sequence_column,
                structure_column=structure_column,
            )
        except InputValidationError as error:
            raise InputValidationError(
                f"RNA table {source} line {reader.line_num}: {error}") from error
        if record.identifier in identifiers:
            raise InputValidationError(
                f"RNA table {source} line {reader.line_num}: duplicate "
                f"identifier {record.identifier!r}")
        identifiers.add(record.identifier)
        records.append(record)
    if not records:
        raise ValueError(f"RNA table contains no records: {source}")
    return records


def read_rna_table(
    path: str | Path,
    *,
    identifier_column: str = "transcript_id",
    sequence_column: str = "sequence",
    structure_column: str = "secondary_structure",
    delimiter: str = "\t",
) -> list[RNA]:
    """Read validated RNAs from a delimited table in input order."""
    path = Path(path)
    try:
        with path.open(newline="") as handle:
            return _read_handle(
                handle,
                source=str(path),
                identifier_column=identifier_column,
                sequence_column=sequence_column,
                structure_column=structure_column,
                delimiter=delimiter,
            )
    except UnicodeDecodeError as error:
        raise ValueError(f"RNA table is not valid text: {path}") from error


__all__ = ["read_rna_table"]
