import pytest

from ginfinity import InputValidationError, RNA, read_rna_table


def test_rna_from_mapping_accepts_configurable_columns():
    record = RNA.from_mapping(
        {"name": "rna-1", "bases": "ACGT", "dot_bracket": "(())"},
        identifier_column="name",
        sequence_column="bases",
        structure_column="dot_bracket",
    )
    assert record == RNA("rna-1", "ACGU", "(())")


def test_read_rna_table_allows_extra_reordered_and_custom_columns(tmp_path):
    table = tmp_path / "structures.csv"
    table.write_text(
        "dot_bracket,source,rna_name,bases\n"
        "(()),example,first,ACGU\n"
        "....,example,second,GGAA\n")
    records = read_rna_table(
        table,
        identifier_column="rna_name",
        sequence_column="bases",
        structure_column="dot_bracket",
        delimiter=",",
    )
    assert [record.identifier for record in records] == ["first", "second"]
    assert records[0].structure == "(())"


def test_read_rna_table_reports_missing_configured_column(tmp_path):
    table = tmp_path / "structures.tsv"
    table.write_text("name\tbases\nfirst\tACGU\n")
    with pytest.raises(ValueError, match="dot_bracket"):
        read_rna_table(
            table,
            identifier_column="name",
            sequence_column="bases",
            structure_column="dot_bracket",
        )


def test_read_rna_table_adds_line_context_to_invalid_rna(tmp_path):
    table = tmp_path / "structures.tsv"
    table.write_text(
        "transcript_id\tsequence\tsecondary_structure\n"
        "bad\tACGN\t....\n")
    with pytest.raises(InputValidationError, match="line 2"):
        read_rna_table(table)
