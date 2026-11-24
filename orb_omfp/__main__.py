"""Command-line entry points for paper reproduction."""

import argparse
import subprocess
import sys
from pathlib import Path
from .config import ROOT, MEASUREMENTS, GROUPS


def main():
    parser = argparse.ArgumentParser(prog="orb-omfp")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="Prepare the four fixed financial tasks")
    run = commands.add_parser("run", help="Recompute a paper experiment")
    run.add_argument(
        "stage",
        choices=["kernel", "selection", "surfaces", "reliability", "budgets", "synthetic", "all"],
    )
    run.add_argument("--group", choices=list(GROUPS))
    run.add_argument("--draws", type=int, default=128)
    run.add_argument("--features", type=int, default=262144)
    figures = commands.add_parser("figures", help="Redraw paper figures from CSV")
    figures.add_argument("--source", type=Path, default=MEASUREMENTS)
    figures.add_argument("--out", type=Path, default=ROOT / "figures")
    figures.add_argument("--only", nargs="*", help="Paper figure numbers, such as 4.3 4.10")
    tables = commands.add_parser("tables", help="Export paper tables from recomputed results")
    commands.add_parser("verify", help="Run numerical and provenance checks")
    args = parser.parse_args()
    if args.command == "prepare":
        from .data import prepare

        prepare()
    elif args.command == "run":
        if not 2 <= args.draws <= 256:
            parser.error("Use between 2 and 256 draws")
        if args.features < 4096 or args.features % 4096:
            parser.error("Use a positive multiple of 4096 feature probes")
        stages = (
            ["kernel", "selection", "surfaces", "reliability", "budgets", "synthetic"]
            if args.stage == "all"
            else [args.stage]
        )
        for stage in stages:
            if stage == "kernel":
                from .kernel import main as compute

                compute()
            elif stage == "selection":
                from .validation import run as compute

                compute()
            elif stage == "surfaces":
                from .features import fit_surface
                from .reporting import surfaces

                for group in [args.group] if args.group else GROUPS:
                    fit_surface(group, args.draws)
                surfaces()
            elif stage == "reliability":
                from .reliability import run as compute

                compute(args.group, args.draws, args.features)
            elif stage == "budgets":
                from .budgets import run as compute

                compute(args.group, args.draws)
            elif stage == "synthetic":
                from .synthetic import run as compute

                compute(args.draws)
    elif args.command == "tables":
        from .reporting import tables

        tables()
    elif args.command == "figures":
        from .figures import draw_paper

        draw_paper(args.source, args.out, args.only)
    elif args.command == "verify":
        raise SystemExit(
            subprocess.call(
                [sys.executable, "-m", "unittest", "discover", "-s", str(ROOT / "tests"), "-v"],
                cwd=ROOT,
            )
        )


if __name__ == "__main__":
    main()
