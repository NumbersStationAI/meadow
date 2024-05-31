"""Evaluate text2sql spider model predictions."""

import json
import os
import re
import signal
import sys
from pathlib import Path
from typing import Any

import click
import pandas as pd
import sqlglot
from tqdm.auto import tqdm

from meadow.cache.duckdb import DuckDBCache
from meadow.client.client import Client

sys.path.append(os.path.join(os.path.dirname(__file__), "."))
# from metrics.spider import evaluation as spider_evaluation  # type: ignore # noqa: E402
from metrics.test_suite_sql_eval import (  # type: ignore # noqa: E402
    evaluation as test_suite_evaluation,
)

from experiments.text2sql.utils import (  # type: ignore  # noqa: E402
    correct_casing,
    edit_distance,
    execution_accuracy,
)
from meadow.client.api.openai import OpenAIClient
from meadow.database.connector.sqlite import SQLiteConnector
from meadow.database.database import Database

LEVELS = ["easy", "medium", "hard", "duckdb", "ddl", "all"]
PARTIAL_TYPES = [
    "select",
    "select(no AGG)",
    "where",
    "where(no OP)",
    "group(no Having)",
    "group",
    "order",
    "and/or",
    "IUEN",
    "keywords",
]
TIMEOUT_SECONDS = 30


def timeout_handler(signum: int, frame: Any) -> None:
    raise TimeoutError("Function execution timed out.")


def print_scores(scores: dict, model_name: str, metric_type: str = "exec") -> None:
    """Print scores."""

    def print_formated_s(
        row_name: str, ll: list[str], element_format: str = "{}", sep: str = "\t"
    ) -> None:
        template = "{}" + sep + sep.join([element_format] * len(ll))
        print(template.format(row_name, *ll))

    # Add empty scores for each level if not present
    for level in LEVELS:
        if level not in scores:
            scores[level] = {}
            scores[level]["count"] = 0
            scores[level]["exec"] = 0
            scores[level]["exact"] = 0

    print_formated_s("", LEVELS)
    counts = [scores[level]["count"] for level in LEVELS]
    print_formated_s("count", counts)
    print(f">======================   {model_name}     =====================")
    if metric_type == "exec":
        print(">=====================   EXECUTION ACCURACY     =====================")
        exec_scores = [scores[level]["exec"] for level in LEVELS]
        print_formated_s("execution", exec_scores, element_format="{:.3f}")

    elif metric_type == "exact":
        print("\n>====================== EXACT MATCHING ACCURACY =====================")
        exact_scores = [scores[level]["exact"] for level in LEVELS]
        print_formated_s("exact match", exact_scores, element_format="{:.3f}")


def compute_exact_match_metric(
    predictions: list,
    references: list,
    gold_dbs: list,
    kmaps: dict,
    db_dir: str,
) -> dict:
    """Compute exact match metric."""
    exact_match = {}
    exact_match["all"] = {}
    exact_match["all"]["count"] = 0
    exact_match["all"]["exact"] = 0
    for prediction, reference, gold_db in tqdm(
        zip(predictions, references, gold_dbs), total=len(predictions)
    ):
        exact_match["all"]["count"] += 1
        try:
            match = int(prediction.trim() == reference.trim())
            exact_match["all"]["exact"] += match
        except Exception:
            pass
    return exact_match


def compute_test_suite_metric(
    predictions: list,
    references: list,
    gold_dbs: list,
    kmaps: dict,
    db_dir: str,
) -> tuple[Any, list[int | None]]:
    """Compute test suite execution metric."""
    evaluator = test_suite_evaluation.Evaluator(
        db_dir=db_dir,
        kmaps=kmaps,
        etype="exec",
        plug_value=False,
        keep_distinct=False,
        progress_bar_for_each_datapoint=False,
    )
    # Only used for Sparc/CoSQL
    turn_scores: dict[str, list] = {"exec": [], "exact": []}
    by_row_metrics: list[int | None] = []
    for prediction, reference, gold_db in tqdm(
        zip(predictions, references, gold_dbs),
        total=len(predictions),
    ):
        turn_idx = 0
        # skip final utterance-query pairs
        if turn_idx < 0:
            continue

        # Register the timeout handler function
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(TIMEOUT_SECONDS)

        try:
            ex_metrics = evaluator.evaluate_one(
                gold_db,
                reference,
                prediction,
                turn_scores,
                idx=turn_idx,
            )
            signal.alarm(0)

            by_row_metrics.append(int(ex_metrics["exec"]))
        except Exception as e:
            raise e
            by_row_metrics.append(None)
            pass
    evaluator.finalize()
    return evaluator.scores, by_row_metrics


def compute_execution_accuracy(
    gold_sqls: list[str],
    pred_sqls: list[str],
    gold_dbs: list[str],
    db_dir: str,
    client: Client,
) -> tuple[float, float, list[float], list[float]]:
    """Compute execution accuracy in utils."""
    scores = []
    empty_res_scores = []
    empty_res_gold_scores = []
    for gold_sql, pred_sql, gold_db in tqdm(
        zip(gold_sqls, pred_sqls, gold_dbs), total=len(gold_sqls), desc="Exec Acc Ours"
    ):
        connector = SQLiteConnector(db_path = str(Path(db_dir) / gold_db / f"{gold_db}.sqlite"))
        database = Database(connector=connector)
        try:
            exec_score, empty_res_score, empty_res_gold_score = execution_accuracy(gold_sql, pred_sql, database, client)
            scores.append(exec_score)
            empty_res_scores.append(empty_res_score)
            empty_res_gold_scores.append(empty_res_gold_score)
        except Exception:
            scores.append(0)
            empty_res_scores.append(0)
            empty_res_gold_scores.append(0)
            pass
    print("PERCENT EMPTY GOLD", (len(empty_res_gold_scores) - sum(empty_res_gold_scores)) / len(empty_res_gold_scores))
    return sum(scores) / len(scores), sum(empty_res_scores) / len(empty_res_scores), scores, empty_res_scores


def compute_metrics(
    gold_sqls: list[str],
    pred_sqls: list[str],
    gold_dbs: list[str],
    kmaps: dict,
    database_dir: str,
    model_name: str,
    client: Client,
) -> dict[str, str]:
    """Compute all metrics for data slice."""
    if len(gold_sqls) != len(pred_sqls):
        raise ValueError(
            f"Gold {len(gold_sqls)} and pred {len(pred_sqls)} have different number of lines!"
        )
    all_metrics: dict[str, Any] = {}

    # Execution Accuracy (Spider)
    metrics, by_row_metrics = compute_test_suite_metric(
        pred_sqls,
        gold_sqls,
        gold_dbs,
        kmaps,
        database_dir,
    )
    all_metrics["exec"] = metrics
    all_metrics["by_row_exec"] = by_row_metrics
    print_scores(metrics, model_name, "exec")

    # Exact Match Accuracy (Spider)
    metrics = compute_exact_match_metric(
        pred_sqls, gold_sqls, gold_dbs, kmaps, database_dir
    )
    all_metrics["exact"] = metrics
    print_scores(metrics, model_name, "exact")

    # Execution Accuracy (Ours)
    exec_sc, empty_res, exec_by_row, empty_res_by_row = compute_execution_accuracy(
        gold_sqls,
        pred_sqls,
        gold_dbs,
        database_dir,
        client,
    )
    all_metrics["exec_ours"] = {"exec_ours": exec_sc}
    all_metrics["nonempty_results"] = {"nonempty_results": empty_res}
    all_metrics["by_row_exec_ours"] = exec_by_row
    all_metrics["by_row_nonempty_results"] = empty_res_by_row
    print("\n>====================== EXECUTION OURS ACCURACY =====================")
    print(f"{exec_sc:.3f}")

    # Get number emp

    # Equality Accuracy
    per_row_match = [
        int(gold.lower() == pred.lower()) for gold, pred in zip(gold_sqls, pred_sqls)
    ]
    all_metrics["equality"] = {"equality": sum(per_row_match) / len(gold_sqls)}
    all_metrics["by_row_equality"] = per_row_match

    # Edit Distance
    per_row_edit_dist = [
        edit_distance(gold, pred) for gold, pred in zip(gold_sqls, pred_sqls)
    ]
    edit_dist = sum(per_row_edit_dist) / len(gold_sqls)
    all_metrics["edit_distance"] = {"edit_distance": edit_dist}
    all_metrics["by_row_edit_distance"] = per_row_edit_dist

    return all_metrics


def get_to_print(metrics: dict, key: str, model_name: str, num_rows: int) -> dict:
    """Get pretty print dictionary of metrics."""
    return {
        "slice": key,
        "model": model_name,
        "support": num_rows,
        "exec": f"{metrics[key]['exec']['all']['exec']:.3f}",
        "exact": f"{metrics[key]['exact']['all']['exact']:.3f}",
        "exec_ours": f"{metrics[key]['exec_ours']['exec_ours']:.3f}",
        "nonempty_results": f"{metrics[key]['nonempty_results']['nonempty_results']:.3f}",
        "equality": f"{metrics[key]['equality']['equality']:.3f}",
        "edit_distance": f"{metrics[key]['edit_distance']['edit_distance']:.3f}",
    }


@click.group()
def cli() -> None:
    """Entrypoint."""
    pass


@cli.command()
@click.option("--gold", type=str, required=True)
@click.option("--pred", type=str, required=True)
@click.option("--tables", type=str, required=True)
@click.option("--db", type=str, default="")
@click.option("--slice-attribute", type=str, default=None)
@click.option("--output-dir", type=str, default="")
@click.option("--output-filename", type=str, default="")
@click.option(
    "--correct-sql-casing", type=bool, is_flag=True, default=False, required=False
)
@click.option(
    "--correct-flight-issue", type=bool, is_flag=True, default=False, required=False
)
@click.option("--api-key", type=str, required=True)
@click.option("--model", type=str, default="gpt-4o")
@click.option(
    "--client-cache-path",
    type=str,
    default="/home/lorr1/projects/code/meadow/test_cache.duckdb",
)
def evaluate(
    gold: str,
    pred: str,
    tables: str,
    db: str,
    slice_attribute: str,
    output_dir: str,
    output_filename: str,
    correct_sql_casing: bool,
    correct_flight_issue: bool,
    api_key: str,
    model: str,
    client_cache_path: str,
) -> None:
    """Evaluate SQL.

    Args:
        gold: path to gold sql file.
        pred: path to predicted json lines file.
        tables: the json path of the table metadata.
        db: path to database dir.
        slice_attribute: json attribute in gold data to slice on.
        output_dir: the prediction output directory
        output_filename: the prediction output filename
        correct_sql_casing: whether to correct casing of SQL keywords
    """
    # Setup model
    cache = DuckDBCache(client_cache_path)
    api_provider = OpenAIClient(api_key=api_key)
    client = Client(api_client=api_provider, model=model, cache=cache)

    # Setup sqls
    gold_path = Path(gold)
    pred_path = Path(pred)
    model_name = pred_path.stem
    if not output_filename:
        output_filename = pred_path.stem + "_eval.json"
    print(f"Saving to {Path(output_dir) / output_filename}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    kmaps = test_suite_evaluation.build_foreign_key_map_from_json(tables)

    gold_sqls_dict = json.load(gold_path.open("r", encoding="utf-8"))
    pred_sqls_dict = [json.loads(ll) for ll in pred_path.open("r").readlines()]

    pred_db_id_question = set([(p["db_id"], p["question"]) for p in pred_sqls_dict])
    gold_sqls_dict = [g for g in gold_sqls_dict if (g["db_id"], g["question"]) in pred_db_id_question]
    # Data validation
    assert len(gold_sqls_dict) == len(
        pred_sqls_dict
    ), "Sample size doesn't match between pred and gold file"

    if correct_flight_issue:
        def _remap_where_sql(
            node: sqlglot.Expression
        ) -> sqlglot.Expression:
            """Remap tables."""
            if isinstance(node, sqlglot.exp.Where):
                new_lhs = sqlglot.parse_one(f"TRIM({node.this.this.sql()})")
                node.this.args["this"] = new_lhs
            elif isinstance(node, sqlglot.exp.Join):
                new_lhs = sqlglot.parse_one(f"TRIM({node.args['on'].this.sql()})")
                new_rhs = sqlglot.parse_one(f"TRIM({node.args['on'].expression.sql()})")
                node.args["on"].args["this"] = new_lhs
                node.args["on"].args["expression"] = new_rhs
            return node

        for gold_sql in gold_sqls_dict:
            if gold_sql["db_id"] == "flight_2":
                try:
                    parsed = sqlglot.parse_one(gold_sql["query"], dialect="sqlite")
                    parsed = parsed.transform(_remap_where_sql)
                    gold_sql["query"] = parsed.sql(dialect="sqlite")
                except Exception:
                    pass

    # Keep track of everything
    full_results = []
    for gold_sql, pred_sql in zip(gold_sqls_dict, pred_sqls_dict):
        merged_res = {**pred_sql, **gold_sql}
        full_results.append(merged_res)

    gold_sqls = [
        re.sub(r"[\s\t\n]+", " ", p.get("gold", p.get("query", p.get("sql", ""))))
        for p in gold_sqls_dict
    ]
    gold_dbs = [p.get("db_id", p.get("db", "")) for p in gold_sqls_dict]
    pred_sqls = [re.sub(r"[\s\t\n]+", " ", p["pred"]) for p in pred_sqls_dict]
    if correct_sql_casing:
        # One line to correct casing of SQL keywords using correct_casing(sql)
        gold_sqls = [correct_casing(sql) for sql in gold_sqls]
        pred_sqls = [correct_casing(sql) for sql in pred_sqls]

    final_metrics: dict[str, dict[str, Any]] = {}
    to_print = []
    final_metrics["all"] = compute_metrics(
        gold_sqls=gold_sqls,
        pred_sqls=pred_sqls,
        gold_dbs=gold_dbs,
        kmaps=kmaps,
        database_dir=db,
        model_name=model_name + "(all)",
        client=client,
    )

    for k, v in final_metrics["all"].items():
        if k.startswith("by_row"):
            assert len(v) == len(gold_sqls)
            for dct, val in zip(full_results, v):
                dct[k[len("by_row_") :]] = val
    to_print.append(get_to_print(final_metrics, "all", model_name, len(gold_sqls)))
    # TODO: could be way more efficient if we subsliced the results but...whatever
    if slice_attribute:
        for unq_value in sorted(set([g[slice_attribute] for g in gold_sqls_dict])):
            idx_set = [
                i
                for i, g in enumerate(gold_sqls_dict)
                if g[slice_attribute] == unq_value
            ]
            print(f"Processing {unq_value} with {len(idx_set)} samples")
            final_metrics[unq_value] = compute_metrics(
                gold_sqls=[gold_sqls[i] for i in idx_set],
                pred_sqls=[pred_sqls[i] for i in idx_set],
                gold_dbs=[gold_dbs[i] for i in idx_set],
                kmaps=kmaps,
                database_dir=db,
                model_name=model_name + f"({unq_value})",
                client=client,
            )
            to_print.append(
                get_to_print(final_metrics, unq_value, model_name, len(idx_set))
            )

    df = pd.DataFrame(to_print)
    print(df.to_csv(sep=",", index=False))
    print("******")
    print(f"Saved metrics to {Path(output_dir) / output_filename}")
    json.dump(final_metrics, open(Path(output_dir) / output_filename, "w"), indent=4)
    output_filename = str(output_filename).replace("_eval.json", "_fd.jsonl")
    print(f"Saved dump to {Path(output_dir) / output_filename}")
    with open(Path(output_dir) / output_filename, "w") as f:
        for dct in full_results:
            f.write(json.dumps(dct) + "\n")


if __name__ == "__main__":
    cli()
