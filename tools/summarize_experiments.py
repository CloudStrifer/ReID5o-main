"""Aggregate only completed experiments, retaining missing runs as pending."""

import argparse
import csv
import json
import statistics
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    groups = {}
    pending = []
    for job in plan["jobs"]:
        for rate in (0.0, 0.2, 0.3, 0.5):
            source = (
                Path(job["output_dir"])
                / "evaluation"
                / f"missing_{rate:g}_seed_42"
                / "metrics.json"
            )
            if not source.exists():
                pending.append(
                    {
                        "variant": job["variant"],
                        "seed": job["seed"],
                        "rate": rate,
                        "missing": str(source),
                    }
                )
                continue
            results = json.loads(source.read_text(encoding="utf-8"))["results"][
                job["query_task"]
            ]
            groups.setdefault((job["variant"], rate), []).append(results)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "variant",
                "missing_rate",
                "n",
                "R1_mean",
                "R1_std",
                "mAP_mean",
                "mAP_std",
            ],
        )
        writer.writeheader()
        for (variant, rate), rows in groups.items():
            row = {"variant": variant, "missing_rate": rate, "n": len(rows)}
            for metric in ("R1", "mAP"):
                values = [x[metric] for x in rows if x[metric] is not None]
                row[metric + "_mean"] = statistics.mean(values) if values else None
                row[metric + "_std"] = (
                    statistics.stdev(values) if len(values) > 1 else None
                )
            writer.writerow(row)
    output.with_suffix(".pending.json").write_text(
        json.dumps(pending, indent=2), encoding="utf-8"
    )
    print("Completed groups:", len(groups), "pending runs/rates:", len(pending))


if __name__ == "__main__":
    main()
