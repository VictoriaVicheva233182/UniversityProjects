"""Command line interface: ``assurance-lab --help``."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from assurance_lab import __version__
from assurance_lab.config import AppConfig, load_config, resolve_path
from assurance_lab.logging_utils import setup_logging

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Red team and harden a RAG assistant, then write an assurance report.",
)
console = Console()
logger = logging.getLogger("assurance_lab")

SUITE = Path("data/attacks/attack_suite.yaml")
BENIGN = Path("data/benign/benign_eval.yaml")
TRAIN_DATA = Path("data/generated/guardrail_train.jsonl")
MODEL = Path("models/guardrail.joblib")
REPORTS = Path("reports")
RUNS = REPORTS / "runs"

ProviderOpt = typer.Option(None, "--provider", "-p", help="ollama | openai | anthropic | simulated")
ModelOpt = typer.Option(None, "--model", "-m", help="Model name, for example llama3.2:3b")


@app.callback()
def main(
    log_level: str = typer.Option("INFO", "--log-level", help="DEBUG, INFO, WARNING"),
    version: bool = typer.Option(False, "--version", help="Show the version and exit."),
) -> None:
    setup_logging(log_level)
    if version:
        console.print(__version__)
        raise typer.Exit()


# --------------------------------------------------------------------------- data + model
@app.command("generate-data")
def generate_data(
    out: Path = typer.Option(TRAIN_DATA, help="Where to write the JSONL training file."),
    seed: int = typer.Option(42),
) -> None:
    """Generate the synthetic guardrail training set."""
    from assurance_lab.ml.dataset import generate_examples, save_jsonl

    examples = generate_examples(seed, resolve_path("data/knowledge_base"))
    save_jsonl(examples, resolve_path(out))
    n_bad = sum(e.label for e in examples)
    console.print(f"Wrote {len(examples)} examples ({n_bad} malicious) to {out}")


@app.command()
def train(
    data: Path = typer.Option(TRAIN_DATA, help="Training JSONL (generated if missing)."),
    extra_csv: Path | None = typer.Option(None, help="Extra labelled CSV with columns text,label."),
    out: Path = typer.Option(MODEL, help="Model output path."),
    target_fpr: float = typer.Option(0.02, help="Maximum false positive rate on validation."),
    seed: int = typer.Option(42),
) -> None:
    """Train the guardrail classifier and evaluate it on the held-out red team suite."""
    from assurance_lab.guardrails.input_classifier import InputClassifier
    from assurance_lab.ml.dataset import generate_examples, load_extra_csv, load_jsonl, save_jsonl
    from assurance_lab.ml.evaluate import evaluate_on_suite
    from assurance_lab.ml.model_card import write_model_card
    from assurance_lab.ml.train import train_guardrail
    from assurance_lab.rag.documents import load_corpus
    from assurance_lab.redteam.attacks import load_benign, load_suite

    data_path = resolve_path(data)
    if not data_path.is_file():
        save_jsonl(generate_examples(seed, resolve_path("data/knowledge_base")), data_path)
    examples = load_jsonl(data_path)
    if extra_csv:
        examples += load_extra_csv(resolve_path(extra_csv))

    training = train_guardrail(examples, resolve_path(out), target_fpr=target_fpr, seed=seed)
    classifier = InputClassifier.load(resolve_path(out))
    chunks = load_corpus(
        [resolve_path("data/knowledge_base"), resolve_path("data/untrusted_sources")]
    )
    held_out = evaluate_on_suite(
        classifier, load_suite(resolve_path(SUITE)), load_benign(resolve_path(BENIGN)), chunks
    )
    metrics = {"training": training, "held_out": held_out}
    reports = resolve_path(REPORTS)
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "guardrail_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_model_card(metrics, reports / "guardrail_model_card.md")

    m = held_out["metrics"]
    console.print(
        f"[bold]Held-out[/bold] recall {m['recall']:.2f}, precision {m['precision']:.2f}, "
        f"false positive rate {m['false_positive_rate']:.2f}. Model card: reports/guardrail_model_card.md"
    )


# --------------------------------------------------------------------------- red teaming
def _print_summary(summary: dict) -> None:
    table = Table(title=f"Red team results: {summary['profile']} ({summary['target_llm']})")
    table.add_column("Category")
    table.add_column("Runs", justify="right")
    table.add_column("Success", justify="right")
    table.add_column("ASR", justify="right")
    table.add_column("95% CI", justify="right")
    for cat in summary["categories"].values():
        lo, hi = cat["ci95"]
        table.add_row(
            cat["title"],
            str(cat["runs"]),
            str(cat["successes"]),
            f"{cat['asr']:.0%}",
            f"{lo:.0%} to {hi:.0%}",
        )
    a, b = summary["attacks"], summary["benign"]
    table.add_section()
    table.add_row(
        "[bold]All attacks[/bold]",
        str(a["runs"]),
        str(a["successes"]),
        f"[bold]{a['asr']:.0%}[/bold]",
        "",
    )
    console.print(table)
    console.print(
        f"Benign questions: {b['questions']}, wrongly blocked {b['false_block_rate']:.0%}, "
        f"expected fact found {b['keyword_hit_rate']:.0%}, mean latency {b['mean_latency_ms']:.0f} ms"
    )
    if summary.get("simulated"):
        console.print(
            "[yellow]Simulated provider: use these numbers to test the pipeline, not as findings.[/yellow]"
        )


def _run_redteam(
    cfg: AppConfig,
    converters: list[str],
    limit: int | None,
    categories: list[str] | None,
    benign: bool,
) -> Path:
    from assurance_lab.factory import build_pipeline
    from assurance_lab.llm.factory import create_llm
    from assurance_lab.redteam.attacks import load_benign, load_suite
    from assurance_lab.redteam.judge import LLMJudge
    from assurance_lab.redteam.runner import run_profile, save_run

    pipeline = build_pipeline(cfg)
    judge = LLMJudge(create_llm(cfg.judge)) if cfg.judge.enabled else None
    suite = load_suite(resolve_path(SUITE))
    benign_set = load_benign(resolve_path(BENIGN)) if benign else None

    with console.status(f"Running {cfg.profile}...") as status:
        out = run_profile(
            pipeline,
            suite,
            benign_set,
            converters=converters,
            judge=judge,
            limit=limit,
            categories=categories,
            progress=lambda rid: status.update(f"Running {cfg.profile}: {rid}"),
        )
    run_dir = save_run(out, resolve_path(RUNS))
    _print_summary(out.summary)
    console.print(f"Saved to {run_dir}")
    return run_dir


@app.command()
def redteam(
    config: Path = typer.Option(Path("configs/baseline.yaml"), "--config", "-c"),
    provider: str | None = ProviderOpt,
    model: str | None = ModelOpt,
    no_converters: bool = typer.Option(False, help="Only run each attack once (faster)."),
    limit: int | None = typer.Option(None, help="Run only the first N attacks (smoke test)."),
    category: list[str] | None = typer.Option(None, help="Only these categories (repeatable)."),
    skip_benign: bool = typer.Option(False, help="Skip the benign utility questions."),
) -> None:
    """Run the attack suite (and benign questions) against one configuration."""
    from assurance_lab.llm.base import LLMError
    from assurance_lab.redteam.converters import DEFAULT_CONVERTERS

    cfg = load_config(config, provider=provider, model=model)
    try:
        _run_redteam(
            cfg, [] if no_converters else DEFAULT_CONVERTERS, limit, category, not skip_benign
        )
    except LLMError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc


@app.command()
def report(
    baseline: Path | None = typer.Option(None, help="Baseline run folder (default: latest)."),
    hardened: Path | None = typer.Option(None, help="Hardened run folder (default: latest)."),
    out: Path = typer.Option(REPORTS, help="Output folder."),
) -> None:
    """Build the HTML and Markdown assurance report from two runs."""
    from assurance_lab.redteam.runner import latest_run
    from assurance_lab.reporting.report import write_report

    runs = resolve_path(RUNS)
    b = resolve_path(baseline) if baseline else latest_run(runs, "baseline")
    h = resolve_path(hardened) if hardened else latest_run(runs, "hardened")
    if not b or not h:
        console.print(
            "[red]Need a baseline and a hardened run. Run `assurance-lab redteam` for both first.[/red]"
        )
        raise typer.Exit(1)
    html, md = write_report(
        b, h, resolve_path(out), resolve_path(REPORTS) / "guardrail_metrics.json"
    )
    console.print(f"Report: {html}\nMarkdown: {md}")


@app.command("all")
def run_all(
    provider: str | None = ProviderOpt,
    model: str | None = ModelOpt,
    no_converters: bool = typer.Option(False, help="Only run each attack once (faster)."),
    retrain: bool = typer.Option(False, help="Retrain the guardrail even if a model exists."),
) -> None:
    """Full engagement: train guardrail, red team baseline and hardened, write the report."""
    from assurance_lab.llm.base import LLMError
    from assurance_lab.redteam.converters import DEFAULT_CONVERTERS
    from assurance_lab.reporting.report import write_report

    if retrain or not resolve_path(MODEL).is_file():
        console.rule("Step 1 of 4: train guardrail")
        train(data=TRAIN_DATA, extra_csv=None, out=MODEL, target_fpr=0.02, seed=42)
    converters = [] if no_converters else DEFAULT_CONVERTERS
    dirs = {}
    try:
        for step, profile in ((2, "baseline"), (3, "hardened")):
            console.rule(f"Step {step} of 4: red team {profile}")
            cfg = load_config(f"configs/{profile}.yaml", provider=provider, model=model)
            dirs[profile] = _run_redteam(cfg, converters, None, None, True)
    except LLMError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.rule("Step 4 of 4: report")
    html, _ = write_report(
        dirs["baseline"],
        dirs["hardened"],
        resolve_path(REPORTS),
        resolve_path(REPORTS) / "guardrail_metrics.json",
    )
    console.print(f"[bold green]Done.[/bold green] Open {html}")


# --------------------------------------------------------------------------- interactive
@app.command()
def chat(
    config: Path = typer.Option(Path("configs/baseline.yaml"), "--config", "-c"),
    provider: str | None = ProviderOpt,
    model: str | None = ModelOpt,
    customer: str = typer.Option("C1001", help="Logged in customer id."),
    trace: bool = typer.Option(True, help="Show flags and sources after each answer."),
) -> None:
    """Chat with the assistant in the terminal. Type 'exit' to stop."""
    from assurance_lab.factory import build_pipeline
    from assurance_lab.llm.base import LLMError

    pipeline = build_pipeline(load_config(config, provider=provider, model=model))
    console.print(
        f"Profile [bold]{pipeline.config.profile}[/bold], logged in as {customer}. Type 'exit' to stop."
    )
    while True:
        try:
            question = console.input("[bold cyan]you> [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        try:
            resp = pipeline.answer(question, customer)
        except LLMError as exc:
            console.print(f"[red]{exc}[/red]")
            continue
        style = "red" if resp.blocked else "green"
        console.print(f"[{style}]assistant>[/{style}] {resp.answer}")
        if trace:
            console.print(
                f"[dim]flags={resp.flags} sources={[s.id + ':' + s.action for s in resp.sources]} "
                f"accounts={resp.customer_ids_in_context} score={resp.input_score} {resp.latency_ms:.0f} ms[/dim]"
            )


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
    reload: bool = typer.Option(False, help="Auto reload on code changes (development)."),
) -> None:
    """Start the API and the test bench UI at http://HOST:PORT."""
    import uvicorn

    console.print(f"Test bench: http://{host}:{port}   API docs: http://{host}:{port}/docs")
    uvicorn.run("assurance_lab.api.app:app", host=host, port=port, reload=reload)


@app.command()
def doctor(
    config: Path = typer.Option(Path("configs/baseline.yaml"), "--config", "-c"),
    provider: str | None = ProviderOpt,
    model: str | None = ModelOpt,
) -> None:
    """Check that the model provider and the guardrail model are ready."""
    from assurance_lab.llm.base import LLMError, Message
    from assurance_lab.llm.factory import create_llm

    cfg = load_config(config, provider=provider, model=model)
    ok = True
    console.print(f"Provider: {cfg.llm.provider}, model: {cfg.llm.model}")
    try:
        reply = create_llm(cfg.llm).chat([Message("user", "Reply with the single word: ready")])
        console.print(f"[green]Model answered:[/green] {reply.strip()[:60]}")
    except LLMError as exc:
        ok = False
        console.print(f"[red]Model check failed:[/red] {exc}")
    if resolve_path(MODEL).is_file():
        console.print(f"[green]Guardrail model found:[/green] {MODEL}")
    else:
        console.print("[yellow]Guardrail model missing.[/yellow] Run: assurance-lab train")
    raise typer.Exit(0 if ok else 1)


if __name__ == "__main__":
    app()
