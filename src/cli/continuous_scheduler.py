"""CLI scheduler for daily or one-shot continuous flood-risk evaluation."""

from __future__ import annotations

import argparse

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from src.continuous.service import build_service


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for scheduler command."""

    parser = argparse.ArgumentParser(description="Run scheduled continuous evaluations.")
    parser.add_argument("config", help="Path to project YAML config.")
    parser.add_argument(
        "--mode",
        choices=["once", "daily"],
        default="daily",
        help="Run once or keep running with a daily schedule.",
    )
    parser.add_argument("--hour", type=int, default=6, help="Daily execution hour (0-23).")
    parser.add_argument("--minute", type=int, default=0, help="Daily execution minute (0-59).")
    parser.add_argument(
        "--timezone",
        default="Europe/Warsaw",
        help="IANA timezone for daily scheduler trigger.",
    )
    return parser


def command(
    config_path: str,
    mode: str,
    hour: int,
    minute: int,
    timezone: str,
) -> None:
    """Execute one-shot or continuous scheduler mode."""

    service = build_service(config_path)

    if mode == "once":
        result = service.evaluate(persist=True)
        print(result.model_dump_json(indent=2))
        return

    scheduler = BlockingScheduler(timezone=timezone)

    def run_job() -> None:
        result = service.evaluate(persist=True)
        print(result.model_dump_json(indent=2))

    scheduler.add_job(
        run_job,
        trigger=CronTrigger(hour=hour, minute=minute, timezone=timezone),
        id="continuous-evaluation",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=600,
        replace_existing=True,
    )

    print(
        f"Scheduled continuous evaluation at {hour:02d}:{minute:02d} {timezone}. "
        "Press Ctrl+C to stop."
    )
    scheduler.start()


def main() -> None:
    """Entrypoint for scheduler CLI."""

    args = build_parser().parse_args()
    command(
        config_path=args.config,
        mode=args.mode,
        hour=args.hour,
        minute=args.minute,
        timezone=args.timezone,
    )


if __name__ == "__main__":
    main()
