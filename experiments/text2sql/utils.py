"""Utility metrics."""

import asyncio
import json
from collections import defaultdict

import pandas as pd
import sqlglot
import textdistance
from sqlglot import parse_one

from meadow.client.client import Client
from meadow.database.connector.connector import Column, Table
from meadow.database.connector.sqlite import SQLiteConnector
from meadow.database.database import Database


def load_data(path: str) -> list[dict]:
    """Load data from path."""
    try:
        with open(path) as f:
            data = json.loads(f.read())
    except json.decoder.JSONDecodeError:
        # Try with jsonl
        data = [json.loads(line) for line in open(path)]
    return data


def read_tables_json(
    schema_file: str,
    database_path: str,
    lowercase: bool = False,
) -> dict[str, dict[str, Table]]:
    """Read tables json."""
    data = json.load(open(schema_file))
    db_to_tables = {}
    for db in data:
        db_name = db["db_id"]
        db_path = f"{database_path}/{db_name}/{db_name}.sqlite"
        connector = SQLiteConnector(db_path)
        connector.connect()

        table_names = db["table_names_original"]
        db["column_names_original"] = [
            [x[0], x[1]] for x in db["column_names_original"]
        ]
        db["column_types"] = db["column_types"]
        if lowercase:
            table_names = [tn.lower() for tn in table_names]

        # Stores the start index of each table to get relative column index
        # Input: [[-1, '*'], [0, 'Perpetrator_ID'], [0, 'People_ID'], [0, 'Date'], [1, 'People_ID'], [1, 'Name']]
        # Output: [-1, 1, 4]
        db_table_col_offset_start = []
        cur_ti = 0
        cur_idx = 1
        for idx, (ti, col_name) in enumerate(db["column_names_original"]):
            if ti == -1:
                db_table_col_offset_start.append(-1)
            else:
                if ti == cur_ti:
                    db_table_col_offset_start.append(cur_idx)
                else:
                    cur_ti = ti
                    cur_idx = idx
                    db_table_col_offset_start.append(cur_idx)

        data_sample_query = "SELECT DISTINCT * FROM {} LIMIT 5"
        dfs = [
            connector.run_sql_to_df(data_sample_query.format(tn))
            for tn in table_names
        ]

        pks = db["primary_keys"]
        fks = db["foreign_keys"]
        tables = defaultdict(list)
        for idx, ((ti, col_name), col_type) in enumerate(
            zip(db["column_names_original"], db["column_types"])
        ):
            if ti == -1:
                continue
            if lowercase:
                col_name = col_name.lower()
                col_type = col_type.lower()

            foreign_keys = []
            for fk in fks:
                if idx == fk[0]:
                    other_column = db["column_names_original"][fk[1]]
                    other_column_index = fk[1] - db_table_col_offset_start[fk[1]]
                    other_table = table_names[other_column[0]]
                    foreign_keys.append((other_table, other_column_index))
            tables[table_names[ti]].append(
                Column(
                    name=col_name,
                    data_type=col_type,
                    primary_key=(idx in pks),
                    foreign_keys=foreign_keys,
                )
            )
        db_to_tables[db_name] = {
            table_name: Table(
                name=table_name,
                columns=tables[table_name],
                data=dfs[i].to_dict(orient="records"),
            )
            for i, table_name in enumerate(tables)
        }
    del connector
    return db_to_tables


def correct_casing(sql: str) -> str:
    """Correct casing of SQL."""
    parse: sqlglot.expressions.Expression = parse_one(sql, read="sqlite")
    return parse.sql()


def prec_recall_f1(gold: set, pred: set) -> dict[str, float]:
    """Compute precision, recall and F1 score."""
    prec = len(gold.intersection(pred)) / len(pred) if pred else 0.0
    recall = len(gold.intersection(pred)) / len(gold) if gold else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall else 0.0
    return {"prec": prec, "recall": recall, "f1": f1}


def edit_distance(s1: str, s2: str) -> int:
    """Compute edit distance between two strings."""
    # Make sure s1 is the shorter string
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances: list[int] = list(range(len(s1) + 1))
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(
                    1 + min((distances[i1], distances[i1 + 1], distances_[-1]))
                )
        distances = distances_
    return distances[-1]


def compare_pred_ref(
    pred_df: pd.DataFrame, ref_df: pd.DataFrame, ref_to_pred_col_map: dict[str, str], row_order_matters: bool
) -> tuple[float, str]:
    """Compare the input dataframes and see if they have the same content.

    We assume column order doesn't matter.

    If row order doesn't matter, we sort the values for a table first.

    Args:
        pred_df: the dataframe return by the predicted sql
        ref_df: the dataframe return by the reference sql
    Returns:
        a float score measuring similarity f1 looking at the collection of rows
    """
    # For each column in the ref_df, find the matching column (if exists) in the
    # pred_df and compare the values via F1 (if order doesn't matter) or directly (if order matters)
    pred_cols_mapped = [ref_to_pred_col_map.get(ref_col) for ref_col in ref_df.columns if ref_to_pred_col_map.get(ref_col) is not None]
    pred_df_to_compare = pred_df[pred_cols_mapped]
    if not row_order_matters:
        pred_df_to_compare = pred_df_to_compare.sort_values(by=pred_cols_mapped)
        ref_df = ref_df.sort_values(by=ref_df.columns.tolist())
    total_scores = []
    for ref_col_name in ref_df.columns:
        ref_col = ref_df[ref_col_name].tolist()
        pred_col_name = ref_to_pred_col_map.get(ref_col_name)
        if pred_col_name is None:
            pred_col = [None] * len(ref_col)
        else:
            pred_col = pred_df_to_compare[pred_col_name].tolist()
        # Compare the column values
        # Have special case for when both tables are empty
        if ref_col == [] and pred_col == []:
            similarity = 1
        else:
            similarity = textdistance.levenshtein.similarity(ref_col, pred_col)
        score = (similarity / max(len(ref_col), len(pred_col), 1))
        total_scores.append(score)
    return sum(total_scores) / len(total_scores)


def execution_accuracy(
    gold: str,
    pred: str,
    database: Database,
    client: Client,
) -> tuple[float, float, float]:
    """Evaluate execution accuracy for one example."""
    final_score = 0
    assert gold, "Gold SQL is empty"
    if not pred:
        return 0, 0
    try:
        df_pred = database.run_sql_to_df(pred)
    except Exception:
        return 0, 0
    df_gold = database.run_sql_to_df(gold)
    # Try to find matching columns for each col in df
    prompt = [
        {
            "role": "system",
            "content": """Please output a JSON map of the reference columns to the predicted columns given the dataframes. Make you best guess as to which columns map to which. If there is a column in either table that doesn't match, please leave it out."""
        },
        {
            "role": "user",
            "content": f"""Reference DataFrames:
{df_gold.head(10).to_string()}

Predicted DataFrames:
{df_pred.head(10).to_string()}"""
        }
    ]
    result = asyncio.run(client.chat(messages=prompt, response_format={"type": "json_object"}))

    ref_to_pred_col_map = json.loads(result.choices[0].message.content)

    # Detect if row order matters if the gold SQL has an order by in the outermost SQL
    parsed = sqlglot.parse_one(gold, read="sqlite")
    # The outermost args will only show last SELECT arguments and will ignore order by
    # clauses in the subqueries
    row_order_matters = parsed.args.get("order") is not None

    # Compare the two dataframes
    final_score = compare_pred_ref(df_pred, df_gold, ref_to_pred_col_map, row_order_matters=row_order_matters)
    # Check if pred empty or None value
    if df_pred.empty or all(df_pred.isnull().all()):
        empty_res = 0
    else:
        empty_res = 1
    # Check if gold empty or None value
    if df_gold.empty or all(df_gold.isnull().all()):
        empty_gold_res = 0
    else:
        empty_gold_res = 1
    return final_score, empty_res, empty_gold_res
