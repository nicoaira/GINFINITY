"""Delimited RNA table input with configurable column names."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import TextIO

from ._validation import InputValidationError, RNA, _validate_column_names


def _read_handle(
    handle: TextIO,
    *,
    source: str,
    identifier_column: str,
    sequence_column: str,
    structure_column: str,
    start_column: str | None,
    end_column: str | None,
    delimiter: str,
) -> list[RNA]:
    if len(delimiter) != 1:
        raise ValueError("delimiter must be exactly one character")
    if (start_column is None) != (end_column is None):
        raise ValueError("start and end columns must both be provided")
    required = _validate_column_names(
        identifier_column, sequence_column, structure_column)
    _validate_column_names(
        identifier_column, sequence_column, structure_column,
        start_column, end_column)
    reader = csv.DictReader(handle, delimiter=delimiter)
    if reader.fieldnames is None:
        raise ValueError(f"empty RNA table: {source}")
    if len(set(reader.fieldnames)) != len(reader.fieldnames):
        raise ValueError(f"duplicate column name in RNA table: {source}")
    missing = [column for column in required if column not in reader.fieldnames]
    if missing:
        raise ValueError(
            f"RNA table {source} is missing column(s): " + ", ".join(missing))
    slice_columns = (start_column, end_column)
    if start_column and end_column:
        present = [column in reader.fieldnames for column in slice_columns]
        if any(present) and not all(present):
            missing_slice = [
                column for column in slice_columns
                if column not in reader.fieldnames]
            raise ValueError(
                f"RNA table {source} is missing column(s): "
                + ", ".join(missing_slice))
        if not any(present):
            start_column = None
            end_column = None
    records: list[RNA] = []
    identifiers: set[str] = set()
    for row in reader:
        if None in row:
            raise ValueError(
                f"RNA table {source} line {reader.line_num} has extra fields")
        try:
            expanded = RNA.many_from_mapping(
                row,
                identifier_column=identifier_column,
                sequence_column=sequence_column,
                structure_column=structure_column,
                start_column=start_column,
                end_column=end_column,
                suffix_identifier=True,
            )
        except InputValidationError as error:
            raise InputValidationError(
                f"RNA table {source} line {reader.line_num}: {error}") from error
        for record in expanded:
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
    start_column: str | None = "start",
    end_column: str | None = "end",
    delimiter: str = "\t",
) -> list[RNA]:
    """Read validated RNAs from a delimited table in input order.

    ``start`` and ``end`` columns are optional. When present, each row
    may list one window or parallel comma-separated windows; each window
    becomes its own record with identifier ``{id}:{start}-{end}``.
    """
    path = Path(path)
    try:
        with path.open(newline="") as handle:
            return _read_handle(
                handle,
                source=str(path),
                identifier_column=identifier_column,
                sequence_column=sequence_column,
                structure_column=structure_column,
                start_column=start_column,
                end_column=end_column,
                delimiter=delimiter,
            )
    except UnicodeDecodeError as error:
        raise ValueError(f"RNA table is not valid text: {path}") from error


__all__ = ["read_rna_table"]
