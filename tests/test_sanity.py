from app.execute import Result
from app.memory import Item
from app.sanity import coverage, referenced_columns, warnings
from app.schema import Column, Table

TABLES = [
    Table("district", [Column("district_id", "INTEGER", True), Column("A11", "INTEGER", False),
                       Column("A3", "TEXT", False)], [], 0),
    Table("loan", [Column("loan_id", "INTEGER", True), Column("district_id", "INTEGER", False),
                   Column("amount", "INTEGER", False), Column("status", "TEXT", False)], [], 0),
]


def test_warnings():
    assert warnings(Result(error="no such column")) == []
    assert warnings(Result(rows=[], columns=["x"])) == ["empty result"]
    assert warnings(Result(rows=[(None, 1)], columns=["avg", "n"])) == ["column avg is entirely NULL"]
    fanned = Result(rows=[(1,)] * 8 + [(2,)] * 4, columns=["x"])
    assert warnings(fanned) == ["12 rows but only 2 distinct: a join may be fanning out"]
    assert warnings(Result(rows=[(1,)] * 9, columns=["x"])) == []  # too few rows to tell
    assert warnings(Result(rows=[(i,) for i in range(20)], columns=["x"])) == []


def test_referenced_columns_skips_keys_literals_and_comments():
    sql = """SELECT T1.A11, "status" FROM district AS T1 JOIN loan AS T2 ON T1.district_id = T2.district_id
             WHERE T1.A3 = 'amount' -- amount is not used
             AND `T2`.[amount] > 0"""
    assert referenced_columns(sql, TABLES) == {"a11", "status", "a3", "amount"}


def test_coverage_counts_names_in_the_items_shown():
    needed = {"a11", "a3"}
    item = Item(id="g", db_id="d", kind="glossary", title="Average salary", body="district.A11 is the average salary",
                provenance="docs")
    assert coverage(needed, [item]) == 0.5
    assert coverage(needed, []) == 0.0
    assert coverage(set(), [item]) is None
    with_sql = Item(id="m", db_id="d", kind="metric", title="Region", body="region", sql="T.A3 = 'Prague'",
                    provenance="docs")
    assert coverage(needed, [item, with_sql]) == 1.0
