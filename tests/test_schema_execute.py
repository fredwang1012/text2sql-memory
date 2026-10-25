import sqlite3

import pytest

from app.execute import Result, run, same_result
from app.schema import schema_text


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "shop.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE customer (id INTEGER PRIMARY KEY, name TEXT, bio TEXT, score REAL);
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY, customer_id INTEGER, status TEXT,
            FOREIGN KEY (customer_id) REFERENCES customer(id));
        INSERT INTO customer VALUES (1,'Ann','%s',1.5),(2,'Bob','short',2.5),(3,'Ann','x',NULL);
        INSERT INTO orders VALUES (1,1,'paid'),(2,2,'open');
        """
        % ("long " * 30)
    )
    conn.commit()
    conn.close()
    return path


def test_schema_text_has_fk_samples_counts(db):
    text = schema_text(db)
    assert 'CREATE TABLE "customer"' in text
    assert "-- 3 rows" in text and "-- 2 rows" in text
    assert '"id" INTEGER PRIMARY KEY' in text
    assert 'FOREIGN KEY ("customer_id") REFERENCES "customer"("id")' in text
    assert "e.g. 'Ann', 'Bob'" in text
    assert "'paid'" in text
    assert "long long" not in text  # long values are skipped
    score_line = next(line for line in text.splitlines() if '"score"' in line)
    assert "e.g." not in score_line


def test_run_select(db):
    r = run(db, "SELECT name FROM customer ORDER BY id; -- done")
    assert r.error is None
    assert r.columns == ["name"]
    assert r.rows == [("Ann",), ("Bob",), ("Ann",)]


def test_run_with_and_replace_function(db):
    r = run(db, "WITH c AS (SELECT name FROM customer) SELECT replace(name,'A','a') FROM c LIMIT 1")
    assert r.error is None and r.rows == [("ann",)]


@pytest.mark.parametrize(
    "sql",
    [
        "PRAGMA table_info(customer)",
        "ATTACH DATABASE 'x.db' AS x",
        "INSERT INTO customer VALUES (9,'z','z',0)",
        "UPDATE customer SET name='z'",
        "DELETE FROM customer",
        "DROP TABLE customer",
        "CREATE TABLE t (a)",
        "ALTER TABLE customer ADD COLUMN z",
        "REPLACE INTO customer VALUES (1,'z','z',0)",
        "SELECT 1; DELETE FROM customer",
        "SELECT 1; SELECT 2",
        "WITH c AS (SELECT 1) DELETE FROM customer",
        "/* SELECT */ DROP TABLE customer",
        "",
        "-- only a comment",
    ],
)
def test_rejected(db, sql):
    r = run(db, sql)
    assert r.error and r.rows == []
    assert run(db, "SELECT COUNT(*) FROM customer").rows == [(3,)]


def test_comment_markers_inside_literals_are_kept(db):
    r = run(db, "SELECT '--;' AS x")
    assert r.error is None and r.rows == [("--;",)]


def test_timeout(db):
    sql = "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM r) SELECT COUNT(*) FROM r"
    r = run(db, sql, timeout_s=0.2)
    assert r.error == "timeout"


def test_row_cap(db):
    r = run(db, "SELECT * FROM customer", max_rows=2)
    assert len(r.rows) == 2 and r.truncated
    assert not run(db, "SELECT * FROM customer", max_rows=3).truncated


def test_sql_error_reported(db):
    assert "no such table" in run(db, "SELECT * FROM nope").error


def test_same_result_ignores_order_duplicates_and_float_noise():
    a = Result(rows=[(1, "a"), (2, "b"), (2, "b")])
    b = Result(rows=[(2, "b"), (1.0, "a")])
    assert same_result(a, b)
    assert same_result(Result(rows=[(0.1 + 0.2,)]), Result(rows=[(0.3,)]))
    assert same_result(Result(rows=[(None,)]), Result(rows=[(None,)]))
    assert not same_result(Result(rows=[(1,)]), Result(rows=[(2,)]))
    assert not same_result(Result(rows=[(1,)]), Result(rows=[(1,)], error="x"))


def test_same_result_is_false_when_either_side_is_truncated():
    cut = Result(rows=[(1,)], truncated=True)
    assert not same_result(cut, Result(rows=[(1,)]))
    assert not same_result(Result(rows=[(1,)]), cut)
    assert not same_result(cut, cut)
