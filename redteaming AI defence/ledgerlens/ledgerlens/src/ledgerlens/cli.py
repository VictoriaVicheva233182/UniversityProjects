"""Command line interface: ``ledgerlens --help``."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ledgerlens import __version__
from ledgerlens.config import AppConfig, load_config
from ledgerlens.logging_utils import setup_logging

app = typer.Typer(add_completion=False, no_args_is_help=True, help="AI-assisted journal entry testing.")
console = Console()

ConfigOpt = typer.Option(Path("configs/default.yaml"), "--config", "-c", help="Settings file.")
ProviderOpt = typer.Option(None, "--provider", "-p", help="ollama | openai | anthropic | simulated")
ModelOpt = typer.Option(None, "--model", "-m", help="Model name, for example llama3.1:8b")
ScaleOpt = typer.Option(None, "--scale", help="Ledger size: 1.0 is about 200,000 entries, 0.2 is a quick run.")


def _cfg(config: Path, provider: str | None = None, model: str | None = None, scale: float | None = None) -> AppConfig:
    cfg = load_config(config, provider=provider, model=model)
    if scale is not None:
        cfg.data.scale = scale
    cfg.out.mkdir(parents=True, exist_ok=True)
    return cfg


@app.callback()
def main(
    log_level: str = typer.Option("INFO", "--log-level"),
    version: bool = typer.Option(False, "--version", help="Show the version and exit."),
) -> None:
    setup_logging(log_level)
    if version:
        console.print(__version__)
        raise typer.Exit()


@app.command()
def generate(config: Path = ConfigOpt, scale: float | None = ScaleOpt, seed: int | None = typer.Option(None)) -> None:
    """Create the fictional company's ledger with hidden fraud schemes."""
    from ledgerlens.pipeline import generate as run

    cfg = _cfg(config, scale=scale)
    if seed is not None:
        cfg.data.seed = seed
    with console.status("Writing a year of bookkeeping..."):
        out = run(cfg)
    console.print(f"{out['entries']:,} journal entries written to {cfg.out / 'data'}")


def _print_metrics(m: dict) -> None:
    table = Table(
        title=f"How far down each list until the fraud shows up ({m['population']:,} entries, {m['schemes']} schemes)"
    )
    table.add_column("Method")
    for k in (50, 100, 200):
        table.add_column(f"Schemes in top {k}", justify="right")
    table.add_column("Entries to see all", justify="right")
    short = {
        "score_rules": "Classic rules",
        "score_isolation_forest": "Isolation forest",
        "score_autoencoder": "Autoencoder",
        "score_ledgerlens": "LedgerLens",
    }
    for key, x in m["methods"].items():
        at = {a["k"]: a for a in x["at_k"]}
        table.add_row(
            short.get(key, x["label"]),
            *(str(at[k]["schemes_found"]) for k in (50, 100, 200)),
            f"{x['entries_to_find_all_schemes'] or 'not all'}",
        )
    console.print(table)
    r = m["rules_any_hit"]
    console.print(
        f"Classic rules flagged [bold]{r['flagged_entries']:,}[/bold] entries ({r['flagged_share']:.1%} of the ledger)."
    )


@app.command()
def analyze(config: Path = ConfigOpt) -> None:
    """Run the classic rules and the anomaly models, and score them against the planted fraud."""
    from ledgerlens.pipeline import analyze as run

    cfg = _cfg(config)
    with console.status("Running rules and training the models..."):
        m = run(cfg)
    _print_metrics(m)


@app.command()
def investigate(
    config: Path = ConfigOpt,
    provider: str | None = ProviderOpt,
    model: str | None = ModelOpt,
    top: int | None = typer.Option(None, help="How many entries from the top of the list."),
) -> None:
    """Let the copilot investigate the top of the review list."""
    from ledgerlens.llm.base import LLMError
    from ledgerlens.llm.factory import create_llm
    from ledgerlens.pipeline import investigate as run

    cfg = _cfg(config, provider, model)
    if top is not None:
        cfg.copilot.top_n = top
    try:
        with console.status(f"The copilot is investigating {cfg.copilot.top_n} entries with {cfg.llm.model}..."):
            stats = run(cfg, create_llm(cfg.llm))
    except LLMError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    fraud_share = stats["suspicious_on_fraud"]
    fraud_text = "n/a" if fraud_share is None else f"{fraud_share:.0%}"
    console.print(
        f"Investigated {stats['investigated']} entries. Drafts with every number found in the ledger: "
        f"{(stats['grounded_share'] or 0):.0%}. Fraud entries called suspicious: {fraud_text}."
    )
    if stats["simulated"]:
        console.print("[yellow]Simulated model: these drafts test the plumbing, not an LLM's judgement.[/yellow]")


@app.command()
def report(config: Path = ConfigOpt) -> None:
    """Write the HTML report."""
    from ledgerlens.reporting.report import write_report

    path = write_report(_cfg(config))
    console.print(f"Report: {path}")


@app.command("all")
def run_all(
    config: Path = ConfigOpt,
    provider: str | None = ProviderOpt,
    model: str | None = ModelOpt,
    scale: float | None = ScaleOpt,
    top: int | None = typer.Option(None, help="How many entries the copilot investigates."),
) -> None:
    """Everything: generate, analyze, investigate, report."""
    from ledgerlens.llm.base import LLMError
    from ledgerlens.llm.factory import create_llm
    from ledgerlens.pipeline import analyze as run_analyze
    from ledgerlens.pipeline import generate as run_generate
    from ledgerlens.pipeline import investigate as run_investigate
    from ledgerlens.reporting.report import write_report

    cfg = _cfg(config, provider, model, scale)
    if top is not None:
        cfg.copilot.top_n = top
    console.rule("Step 1 of 4: write the ledger")
    out = run_generate(cfg)
    console.print(f"{out['entries']:,} journal entries")
    console.rule("Step 2 of 4: rules and models")
    with console.status("Training..."):
        _print_metrics(run_analyze(cfg))
    model_name = "simulated" if cfg.llm.provider == "simulated" else f"{cfg.llm.provider}:{cfg.llm.model}"
    console.rule(f"Step 3 of 4: copilot investigates the top {cfg.copilot.top_n} ({model_name})")
    try:
        with console.status("Investigating..."):
            stats = run_investigate(cfg, create_llm(cfg.llm))
        console.print(f"Drafts with every number found in the ledger: {(stats['grounded_share'] or 0):.0%}")
    except LLMError as exc:
        console.print(f"[red]{exc}[/red]\nSkipping the copilot. The report will be written without draft findings.")
    console.rule("Step 4 of 4: report")
    path = write_report(cfg)
    console.print(f"[bold]Done.[/bold] Open {path} or run: ledgerlens serve")


@app.command()
def explain(entry_id: str, config: Path = ConfigOpt) -> None:
    """Show why one entry is on the review list."""
    from ledgerlens.workspace import Workspace

    ws = Workspace.load(_cfg(config))
    e = ws.entry(entry_id)
    console.print(
        f"[bold]{entry_id}[/bold] {e['entry_type']} of €{float(e['amount']):,.2f} on {e['posting_date']} {e['posting_time']}, rank {ws.score_row(entry_id).get('rank')}"
    )
    for r in ws.reasons(entry_id, top=8):
        console.print(f"  - {r.text}")


@app.command()
def serve(host: str = typer.Option("127.0.0.1"), port: int = typer.Option(8000)) -> None:
    """Open the auditor workbench at http://HOST:PORT."""
    import uvicorn

    console.print(f"Workbench: http://{host}:{port}")
    uvicorn.run("ledgerlens.api.app:app", host=host, port=port)


@app.command()
def doctor(config: Path = ConfigOpt, provider: str | None = ProviderOpt, model: str | None = ModelOpt) -> None:
    """Check the model connection and the generated files."""
    from ledgerlens.llm.base import LLMError, Message
    from ledgerlens.llm.factory import create_llm

    cfg = _cfg(config, provider, model)
    ok = True
    try:
        reply = create_llm(cfg.llm).chat([Message("user", "Reply with the single word: ready")])
        console.print(f"[green]Model {cfg.llm.provider}:{cfg.llm.model} answered:[/green] {reply.strip()[:60]}")
    except LLMError as exc:
        ok = False
        console.print(f"[red]Model check failed:[/red] {exc}")
    for name in ("data/journal_entries.csv", "scores.csv", "metrics.json", "findings.jsonl", "report.html"):
        mark = "[green]found[/green]" if (cfg.out / name).is_file() else "[yellow]missing[/yellow]"
        console.print(f"{mark} {cfg.out / name}")
    if (cfg.out / "metrics.json").is_file():
        m = json.loads((cfg.out / "metrics.json").read_text(encoding="utf-8"))
        console.print(f"Ledger: {m['population']:,} entries, seed {m['seed']}")
    raise typer.Exit(0 if ok else 1)


if __name__ == "__main__":
    app()
