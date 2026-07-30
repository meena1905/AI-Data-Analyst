import pytest
from app.analysis.sql_engine import is_read_only_select, run_sql
from app.utils.errors import QueryExecutionError
def test_is_read_only_select_accepts_select():
    assert is_read_only_select("SELECT * FROM orders")
    assert is_read_only_select("  with t as (select 1) select * from t ")
@pytest.mark.parametrize("sql", [
    "DROP TABLE orders",
    "DELETE FROM orders",
    "INSERT INTO orders VALUES (1)",
    "UPDATE orders SET revenue = 0",
    "ATTACH 'x.db'",
    "",
    "not sql at all",
])
def test_is_read_only_select_rejects_mutations(sql):
    assert not is_read_only_select(sql)
def test_run_sql_executes_select(conn, sales_df):
    conn.register("orders_view", sales_df)
    conn.execute('CREATE TABLE orders AS SELECT * FROM orders_view')
    df = run_sql(conn, "SELECT region, SUM(revenue) AS total FROM orders GROUP BY region ORDER BY total DESC")
    assert df.iloc[0]["region"] in {"North", "West", "East"}
    assert len(df) <= 4
def test_run_sql_rejects_ddl(conn, sales_df):
    conn.register("orders_view", sales_df)
    conn.execute('CREATE TABLE orders AS SELECT * FROM orders_view')
    with pytest.raises(QueryExecutionError):
        run_sql(conn, "DROP TABLE orders")
def test_run_sql_raises_on_invalid_query(conn, sales_df):
    conn.register("orders_view", sales_df)
    conn.execute('CREATE TABLE orders AS SELECT * FROM orders_view')
    with pytest.raises(QueryExecutionError):
        run_sql(conn, "SELECT nonexistent_column FROM orders")
