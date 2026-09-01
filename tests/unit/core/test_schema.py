import pytest

from matera.export.schema import ColumnDef, ExcelSchema


def test_column_def_valid():
    col = ColumnDef("Q1_A", "Q1", "A")
    assert col.column_name == "Q1_A"
    assert col.question_id == "Q1"
    assert col.option_id == "A"
    assert col.value_encoding == "binary"


def test_column_def_invalid_types():
    with pytest.raises(ValueError, match="column_name must be a string"):
        ColumnDef(1, "Q1", "A")  # type: ignore
    with pytest.raises(ValueError, match="question_id must be a string"):
        ColumnDef("Q1_A", 1, "A")  # type: ignore
    with pytest.raises(ValueError, match="option_id must be a string"):
        ColumnDef("Q1_A", "Q1", 1)  # type: ignore


def test_column_def_invalid_encoding():
    with pytest.raises(ValueError, match="Invalid value_encoding"):
        ColumnDef("Q1_A", "Q1", "A", value_encoding="other")  # type: ignore


def test_column_def_empty_identifiers():
    with pytest.raises(ValueError, match="column_name cannot be empty"):
        ColumnDef(" ", "Q1", "A")
    with pytest.raises(ValueError, match="question_id cannot be empty"):
        ColumnDef("Q1_A", "", "A")
    with pytest.raises(ValueError, match="option_id cannot be empty"):
        ColumnDef("Q1_A", "Q1", "\t")


def test_excel_schema_valid():
    col1 = ColumnDef("Q1_A", "Q1", "A")
    col2 = ColumnDef("Q1_B", "Q1", "B")
    schema = ExcelSchema("form1", "v1", (col1, col2))
    assert schema.form_id == "form1"
    assert len(schema.columns) == 2

    # Ordering regression test
    assert schema.columns == (col1, col2)
    assert schema.columns[0].column_name == "Q1_A"
    assert schema.columns[1].column_name == "Q1_B"


def test_excel_schema_invalid_column_types():
    col1 = ColumnDef("Q1_A", "Q1", "A")
    with pytest.raises(ValueError, match="All elements in columns must be instances of ColumnDef"):
        ExcelSchema("form1", "v1", (col1, "Not a ColumnDef"))  # type: ignore


def test_excel_schema_empty_identifiers():
    col1 = ColumnDef("Q1_A", "Q1", "A")
    with pytest.raises(ValueError, match="form_id cannot be empty"):
        ExcelSchema(" ", "v1", (col1,))
    with pytest.raises(ValueError, match="form_version cannot be empty"):
        ExcelSchema("form1", "", (col1,))


def test_excel_schema_invalid_types():
    col1 = ColumnDef("Q1_A", "Q1", "A")
    with pytest.raises(ValueError, match="form_id must be a string"):
        ExcelSchema(1, "v1", (col1,))  # type: ignore
    with pytest.raises(ValueError, match="form_version must be a string"):
        ExcelSchema("form1", 1, (col1,))  # type: ignore


def test_excel_schema_mutable_columns_rejected():
    col1 = ColumnDef("Q1_A", "Q1", "A")
    columns_list = [col1]
    with pytest.raises(ValueError, match="columns must be a tuple"):
        ExcelSchema("form1", "v1", columns_list)  # type: ignore


def test_excel_schema_empty_columns():
    with pytest.raises(ValueError, match="columns cannot be empty"):
        ExcelSchema("form1", "v1", ())


def test_excel_schema_duplicate_column_name():
    col1 = ColumnDef("Q1_A", "Q1", "A")
    col2 = ColumnDef("Q1_A", "Q2", "B")
    with pytest.raises(ValueError, match="Duplicate column_name found: Q1_A"):
        ExcelSchema("form1", "v1", (col1, col2))


def test_excel_schema_duplicate_mapping():
    col1 = ColumnDef("Q1_A", "Q1", "A")
    col2 = ColumnDef("Q1_A_ALT", "Q1", "A")
    with pytest.raises(ValueError, match="Duplicate mapping for question_id=Q1, option_id=A"):
        ExcelSchema("form1", "v1", (col1, col2))


def test_immutability():
    from dataclasses import FrozenInstanceError

    col = ColumnDef("Q1_A", "Q1", "A")
    with pytest.raises(FrozenInstanceError):
        col.column_name = "Q1_B"  # type: ignore
