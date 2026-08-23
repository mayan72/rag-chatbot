from pathlib import Path

import pandas as pd

from rag.query_planner import QueryPlanner
from rag.structured_executor import StructuredExecutor
from rag.table_store import TableStore


def _engine(tmp_path: Path, frames: dict):
    store = TableStore(root=tmp_path / "tables")
    for name, frame in frames.items():
        store.upsert_dataframe(
            df=frame,
            document_id=f"uploaded_{name}",
            document_name=f"{name}.xlsx",
            source_type="xlsx",
        )
    planner = QueryPlanner()
    executor = StructuredExecutor(store)
    return store, planner, executor


def test_count_filters_any_schema(tmp_path):
    sales = pd.DataFrame(
        {
            "region": ["EMEA", "APAC", "EMEA", "AMER"],
            "status": ["Open", "Closed", "Open", "Open"],
            "amount": [10, 20, 30, 40],
        }
    )
    store, planner, executor = _engine(tmp_path, {"sales": sales})
    schemas = store.list_schemas()

    plan = planner.plan(
        "Give me the count of rows where status is Open for EMEA",
        schemas,
        llm=None,
    )

    assert plan.mode == "aggregate"
    result = executor.execute(plan, schemas)
    assert result.answer == "2"


def test_sum_on_different_file(tmp_path):
    hr = pd.DataFrame(
        {
            "department": ["Finance", "Finance", "Engineering"],
            "level": ["Senior", "Junior", "Senior"],
            "salary": [120000, 80000, 150000],
        }
    )
    store, planner, executor = _engine(tmp_path, {"hr": hr})
    schemas = store.list_schemas()

    plan = planner.plan(
        "What is the sum of salary where department is Finance",
        schemas,
        llm=None,
    )

    assert plan.operation == "sum"
    result = executor.execute(plan, schemas)
    assert result.answer == "200000.0"


def test_typo_value_still_matches(tmp_path):
    metals = pd.DataFrame(
        {
            "commodity_name": [
                "Aluminium cash-settlement (LME)",
                "Copper cash-settlement (LME)",
                "Aluminium cash-settlement (LME)",
            ],
            "risk_rating": ["High", "Medium", "Medium"],
        }
    )
    store, planner, executor = _engine(tmp_path, {"metals": metals})
    schemas = store.list_schemas()

    plan = planner.plan(
        "Give me the count of rows where risk rating is medium for Almunium case settlement",
        schemas,
        llm=None,
    )

    result = executor.execute(plan, schemas)
    assert result.answer == "1"

    plan_high = planner.plan(
        "Give me the count of rows where risk rating is High for Almunium case settlement",
        schemas,
        llm=None,
    )
    result_high = executor.execute(plan_high, schemas)
    assert result_high.answer == "1"


def test_total_revenue_for_region(tmp_path):
    sales = pd.DataFrame(
        {
            "Region": ["North", "South", "North", "West"],
            "Revenue": [275000, 50000, 330000, 10000],
        }
    )
    store, planner, executor = _engine(tmp_path, {"sales": sales})
    schemas = store.list_schemas()

    plan = planner.plan(
        "What is the total revenue for the North region?",
        schemas,
        llm=None,
    )

    assert plan.mode == "aggregate"
    assert plan.operation == "sum"
    assert plan.target_column == "Revenue"
    assert any(
        item.column == "Region" and "north" in item.value.casefold()
        for item in plan.filters
    )
    result = executor.execute(plan, schemas)
    assert float(result.answer) == 605000.0


def test_non_aggregate_stays_semantic(tmp_path):
    sales = pd.DataFrame({"region": ["EMEA"], "note": ["Demand improved"]})
    store, planner, _ = _engine(tmp_path, {"sales": sales})
    plan = planner.plan(
        "Summarize the latest demand commentary",
        store.list_schemas(),
        llm=None,
    )
    assert plan.mode == "semantic"


def test_correlation_between_quantity_and_revenue(tmp_path):
    sales = pd.DataFrame(
        {
            "Order_ID": [1, 2, 3, 4],
            "Quantity": [2, 4, 6, 8],
            "Revenue": [100.0, 200.0, 300.0, 400.0],
            "Metric": [None, None, None, None],
            "Expected_Value": [None, None, None, None],
        }
    )
    expected_sheet = pd.DataFrame(
        {
            "Order_ID": [None],
            "Quantity": [None],
            "Revenue": [None],
            "Metric": ["Total Revenue"],
            "Expected_Value": [4028000.0],
        }
    )
    stacked = pd.concat([sales, expected_sheet], ignore_index=True)
    store, planner, executor = _engine(tmp_path, {"workbook": stacked})
    schemas = store.list_schemas()

    plan = planner.plan(
        "What is the correlation between quantity and revenue?",
        schemas,
        llm=None,
    )

    assert plan.mode == "aggregate"
    assert plan.operation == "correlation"
    assert {plan.target_column, plan.second_column} == {"Quantity", "Revenue"}
    result = executor.execute(plan, schemas)
    assert float(result.answer) == 1.0
    assert result.row_count == 4


def test_correlation_uses_pearson_on_numeric_pairs(tmp_path):
    sales = pd.DataFrame(
        {
            "Quantity": [1, 2, 3, 10],
            "Revenue": [10, 12, 14, 11],
        }
    )
    store, planner, executor = _engine(tmp_path, {"sales": sales})
    schemas = store.list_schemas()
    plan = planner.plan(
        "What is the correlation between Quantity and Revenue?",
        schemas,
        llm=None,
    )
    result = executor.execute(plan, schemas)
    expected = float(
        pd.to_numeric(sales["Quantity"]).corr(pd.to_numeric(sales["Revenue"]))
    )
    assert abs(float(result.answer) - expected) < 1e-6
