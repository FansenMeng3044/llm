"""Export OR-Tools result CSVs to JSON and Markdown artifacts.

The raw RALTestSets directory is intentionally ignored by git. This script reads
selected CSV result files from that local data directory and writes portable
artifacts under results/ for sharing or publishing.
"""
import argparse
import csv
import json
import os
from pathlib import Path

RESULT_TYPES = ("ortools", "ortools_taco_replay")


def parse_value(value):
    if value == "":
        return None
    try:
        if value.lower() in {"nan", "none", "null"}:
            return None
    except AttributeError:
        return value
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return int(number)
    return number


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows_as_text = list(reader)
        fieldnames = reader.fieldnames or []
    rows_as_json = [
        {key: parse_value(value) for key, value in row.items()}
        for row in rows_as_text
    ]
    return fieldnames, rows_as_text, rows_as_json


def md_escape(value):
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_markdown(path, dataset, result_type, source_csv, fieldnames, rows):
    lines = [
        f"# {dataset} {result_type}",
        "",
        f"Source CSV: `{source_csv.as_posix()}`",
        f"Rows: {len(rows)}",
        "",
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join(["---"] * len(fieldnames)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(md_escape(row.get(name, "")) for name in fieldnames) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def export_result(dataset_dir, data_root, output_root, result_type):
    csv_path = dataset_dir / f"{result_type}.csv"
    if not csv_path.exists():
        return None

    dataset = dataset_dir.name
    out_dir = output_root / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    fieldnames, rows_as_text, rows_as_json = read_csv(csv_path)
    rel_source = csv_path.relative_to(data_root.parent)

    json_payload = {
        "dataset": dataset,
        "result_type": result_type,
        "source_csv": rel_source.as_posix(),
        "row_count": len(rows_as_json),
        "rows": rows_as_json,
    }
    json_path = out_dir / f"{result_type}.json"
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    md_path = out_dir / f"{result_type}.md"
    write_markdown(md_path, dataset, result_type, rel_source, fieldnames, rows_as_text)

    return json_path, md_path, len(rows_as_json)


def write_index(output_root, exports):
    lines = [
        "# Exported OR-Tools Results",
        "",
        "These files are exported from local `RALTestSets/*/ortools.csv` and `RALTestSets/*/ortools_taco_replay.csv` artifacts.",
        "The `ortools_taco_replay` files are OR-Tools routes replayed through the TACO evaluation wrapper, not original TACO solver outputs.",
        "",
        "| Dataset | Result Type | Rows | JSON | Markdown |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in exports:
        dataset = item["dataset"]
        result_type = item["result_type"]
        json_rel = item["json_path"].relative_to(output_root).as_posix()
        md_rel = item["md_path"].relative_to(output_root).as_posix()
        lines.append(f"| {dataset} | {result_type} | {item['rows']} | [{json_rel}]({json_rel}) | [{md_rel}]({md_rel}) |")
    lines.append("")
    (output_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Export OR-Tools result CSVs to JSON and Markdown.")
    parser.add_argument("--data-root", default="RALTestSets", help="Directory containing RALTestSet_* folders")
    parser.add_argument("--output-root", default="results", help="Directory to write JSON/Markdown artifacts")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    exports = []
    for dataset_dir in sorted(p for p in data_root.glob("RALTestSet_M*_*") if p.is_dir()):
        for result_type in RESULT_TYPES:
            result = export_result(dataset_dir, data_root, output_root, result_type)
            if result is None:
                print(f"missing {dataset_dir.name}/{result_type}.csv")
                continue
            json_path, md_path, rows = result
            exports.append({
                "dataset": dataset_dir.name,
                "result_type": result_type,
                "rows": rows,
                "json_path": json_path,
                "md_path": md_path,
            })
            print(f"exported {dataset_dir.name}/{result_type}: {rows} rows")

    write_index(output_root, exports)
    print(f"wrote {output_root / 'README.md'}")


if __name__ == "__main__":
    main()
