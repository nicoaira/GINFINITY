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


def test_rna_from_mapping_reads_a_single_window_without_renaming():
    record = RNA.from_mapping(
        {"transcript_id": "rna-1", "sequence": "ACGUACGU",
         "secondary_structure": "((....))", "start": "2", "end": "6"},
        start_column="start",
        end_column="end",
    )
    assert record.identifier == "rna-1"
    assert record.start == 2
    assert record.end == 6


def test_rna_from_mapping_rejects_multiple_windows():
    with pytest.raises(InputValidationError, match="multiple slices"):
        RNA.from_mapping(
            {"transcript_id": "rna-1", "sequence": "ACGUACGU",
             "secondary_structure": "((....))", "start": "2,4", "end": "6,8"},
            start_column="start",
            end_column="end",
        )


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


def test_read_rna_table_expands_comma_separated_windows(tmp_path):
    table = tmp_path / "structures.tsv"
    table.write_text(
        "transcript_id\tsequence\tsecondary_structure\tstart\tend\n"
        "rna-1\tACGUACGU\t((....))\t2,4\t6, 8\n")
    records = read_rna_table(table)
    assert [(record.identifier, record.start, record.end) for record in records] == [
        ("rna-1:2-6", 2, 6),
        ("rna-1:4-8", 4, 8),
    ]
    assert records[0].sequence == records[1].sequence == "ACGUACGU"


def test_read_rna_table_ignores_absent_start_end_columns(tmp_path):
    table = tmp_path / "structures.tsv"
    table.write_text(
        "transcript_id\tsequence\tsecondary_structure\n"
        "rna-1\tACGU\t(())\n")
    records = read_rna_table(table)
    assert records[0].identifier == "rna-1"
    assert records[0].start is None


def test_read_rna_table_rejects_mismatched_window_lists(tmp_path):
    table = tmp_path / "structures.tsv"
    table.write_text(
        "transcript_id\tsequence\tsecondary_structure\tstart\tend\n"
        "rna-1\tACGUACGU\t((....))\t2,4\t6\n")
    with pytest.raises(InputValidationError, match="start has 2"):
        read_rna_table(table)
