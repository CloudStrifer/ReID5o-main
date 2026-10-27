"""Generate reviewable experiment jobs; --execute runs them sequentially."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.options import get_args


def variants(suite):
    full = {"full": {}}
    if suite == "ablation":
        return {
            **full,
            "MA_only": {
                "use_cross_modal_completion": False,
                "use_reliability_fusion": False,
            },
            "MA_CC": {"use_reliability_fusion": False},
            "MA_RF": {"use_cross_modal_completion": False},
            "no_recon": {"completion_recon_loss_weight": 0.0},
            "no_cycle": {"completion_cycle_loss_weight": 0.0},
            "no_completion_losses": {
                "completion_recon_loss_weight": 0.0,
                "completion_cycle_loss_weight": 0.0,
            },
            "no_SPA": {"add_lora": False},
            "no_MA": {"use_missing_aware": False},
        }
    if suite == "sensitivity":
        return {
            **full,
            "cons_low": {"consistency_loss_weight": 0.05},
            "cons_high": {"consistency_loss_weight": 0.2},
            "completion_low": {
                "completion_recon_loss_weight": 0.15,
                "completion_cycle_loss_weight": 0.05,
            },
            "completion_high": {
                "completion_recon_loss_weight": 0.6,
                "completion_cycle_loss_weight": 0.2,
            },
            "fusion_low": {
                "fusion_sparsity_weight": 0.005,
                "fusion_uncertainty_weight": 0.01,
            },
            "fusion_high": {
                "fusion_sparsity_weight": 0.02,
                "fusion_uncertainty_weight": 0.04,
            },
        }
    if suite == "spa":
        return {
            **{f"experts_{e}": {"spa_experts": e} for e in range(1, 6)},
            **{f"parts_{p}": {"spa_parts": p} for p in range(1, 7)},
        }
    if suite == "rgb":
        return {
            f"rgb_{rate:g}": {"keep_rgb_prob": rate} for rate in (0.3, 0.5, 0.7, 0.9)
        }
    return full


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        required=True,
        choices=["ORBench", "CUHK-PEDES", "ICFG-PEDES", "RSTPReid"],
    )
    p.add_argument("--root_dir", default=str(ROOT / "data"))
    p.add_argument("--split_file", default=None)
    p.add_argument(
        "--suite",
        choices=["full", "ablation", "sensitivity", "spa", "rgb"],
        default="full",
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--output_dir", default=str(ROOT / "experiments"))
    p.add_argument("--execute", action="store_true")
    cli = p.parse_args()
    base = vars(get_args([]))
    base.update(
        dataset_name=cli.dataset,
        root_dir=str(Path(cli.root_dir).resolve()),
        split_file=str(Path(cli.split_file).resolve()) if cli.split_file else None,
    )
    out = Path(cli.output_dir).resolve() / cli.dataset / cli.suite
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    task = "TEXT+NIR+CP+SK" if cli.dataset == "ORBench" else "TEXT+SK"
    for name, changes in variants(cli.suite).items():
        for seed in cli.seeds:
            config = {
                **base,
                **changes,
                "seed": seed,
                "output_dir": str(out / f"{name}_seed_{seed}"),
            }
            config_path = out / f"{name}_seed_{seed}.yaml"
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            train = [
                sys.executable,
                str(ROOT / "train.py"),
                "--config_file",
                str(config_path),
            ]
            saved = str(Path(config["output_dir"]) / "configs.yaml")
            evaluate = [
                sys.executable,
                str(ROOT / "test.py"),
                "--config_file",
                saved,
                "--query_tasks",
                task,
                "--missing_rates",
                "0",
                "0.2",
                "0.3",
                "0.5",
            ]
            jobs.append(
                {
                    "variant": name,
                    "seed": seed,
                    "config": str(config_path),
                    "output_dir": config["output_dir"],
                    "query_task": task,
                    "train_command": train,
                    "evaluation_command": evaluate,
                }
            )
    plan = out / "plan.json"
    plan.write_text(
        json.dumps(
            {"dataset": cli.dataset, "suite": cli.suite, "jobs": jobs}, indent=2
        ),
        encoding="utf-8",
    )
    print(f"Generated {len(jobs)} jobs: {plan}")
    if cli.execute:
        for job in jobs:
            subprocess.run(job["train_command"], cwd=ROOT, check=True)
            subprocess.run(job["evaluation_command"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
