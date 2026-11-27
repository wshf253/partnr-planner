import argparse
from contextlib import ExitStack
from pathlib import Path
import json
import os

from .qwen import MODEL, choose_qwen, choose_reference
from .scenario import Symbolic, make_day, run_day


def positive(s):
    n = int(s)
    if n < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return n


def main():
    parser = argparse.ArgumentParser(description="Repeated PARTNR table-setting experiment")
    commands = parser.add_subparsers(dest="command", required=True)
    dataset = commands.add_parser("generate", help="generate scored Habitat episodes")
    run = commands.add_parser("run", help="run the paired partner experiment")
    for cmd in (dataset, run):
        cmd.add_argument("--days", type=positive, default=7)
        cmd.add_argument("--seed", type=int, default=17)
        cmd.add_argument("--dataset", type=Path, default=Path("data/table_partner/episodes.json.gz"))
    run.add_argument("--backend", choices=["habitat", "symbolic"], default="habitat")
    run.add_argument("--robot", choices=["qwen", "reference"], default="qwen")
    run.add_argument("--memory", choices=["on", "off"], default="on")
    run.add_argument("--partner", choices=["all", "A", "B", "C"], default="all")
    run.add_argument("--deadline", type=positive, help="fixed preparation deadline in backend steps")
    run.add_argument("--output", type=Path, required=True, help="new JSONL file, one row per day and partner")
    args = parser.parse_args()
    if args.command == "generate":
        from .native import generate
        generate(args.dataset, args.days, args.seed)
        print(f"Generated {args.days} episodes in {args.dataset}")
        return
    decide = choose_qwen if args.robot == "qwen" else choose_reference
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.backend == "habitat" and not args.dataset.is_file():
        parser.error(f"{args.dataset} is missing. Run 'python -m table_partner generate' first.")
    failures = 0
    with args.output.open("x", encoding="utf-8") as out:
        for partner in ("ABC" if args.partner == "all" else args.partner):
            history = []
            with ExitStack() as stack:
                session = None
                if args.backend == "habitat":
                    from .native import HabitatSession
                    session = HabitatSession(args.dataset, args.seed)
                    stack.callback(session.close)
                    if args.days > len(session.episodes):
                        parser.error(f"Dataset contains only {len(session.episodes)} days")
                for i in range(args.days):
                    row = dict(partner=partner, day=i+1, backend=args.backend, robot=args.robot,
                               model=os.getenv("QWEN_MODEL", MODEL) if args.robot == "qwen" else None,
                               seed=args.seed, memory=args.memory)
                    try:
                        day = session.start(i) if session else make_day(i, args.seed)
                        backend = session or Symbolic()
                        budget = args.deadline or day.deadline * (100 if session else 1)
                        result = run_day(day, partner, decide, history, backend, budget)
                        if args.memory == "on":
                            history.extend(result["evidence"])
                        row.update(result)
                    except RuntimeError as exc:
                        row.update(error=str(exc), success=False)
                        failures += 1
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
                    print(partner, i+1, row.get("decision"), row.get("ready_at"), row.get("error", ""))
    if failures:
        raise SystemExit(f"{failures} failed executions, see {args.output}")


if __name__ == "__main__":
    main()
