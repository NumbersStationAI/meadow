"""Utility metrics."""

import json
from collections import defaultdict

import sqlglot
from sqlglot import parse_one

from meadow.database.connector.connector import Column, Table


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
    lowercase: bool = False,
) -> dict[str, dict[str, Table]]:
    """Read tables json."""
    data = json.load(open(schema_file))
    db_to_tables = {}
    for db in data:
        db_name = db["db_id"]
        table_names = db["table_names_original"]
        db["column_names_original"] = [
            [x[0], x[1]] for x in db["column_names_original"]
        ]
        db["column_types"] = db["column_types"]
        if lowercase:
            table_names = [tn.lower() for tn in table_names]
        pks = db["primary_keys"]
        fks = db["foreign_keys"]
        tables = defaultdict(list)
        tables_pks = defaultdict(list)
        tables_fks = defaultdict(list)
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
                    other_table = table_names[other_column[0]]
                    foreign_keys.append((other_table, other_column[1]))
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
                pks=tables_pks[table_name],
                fks=tables_fks[table_name],
                examples=None,
            )
            for table_name in tables
        }
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
