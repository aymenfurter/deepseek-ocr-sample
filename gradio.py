from __future__ import annotations

import sys
import importlib
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Allow running as a standalone script: ensure correct import resolution.
if __package__ in (None, ""):
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    # Avoid name collision with this module when importing the external gradio package.
    if str(SCRIPT_DIR) in sys.path:
        sys.path.remove(str(SCRIPT_DIR))

gr = importlib.import_module("gradio")

# Restore script directory so relative imports still resolve after grabbing external gradio.
if __package__ in (None, ""):
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))

from app import SCENARIO_LABELS, get_persisted_results, run_pipeline

INITIAL_RESULTS = get_persisted_results()
INITIAL_HTML = INITIAL_RESULTS.get("leaderboard_html") or "<p>No results yet.</p>"
INITIAL_CANDIDATES = INITIAL_RESULTS.get("candidate_details") or []
SCENARIO_CHOICES = ["all"] + SCENARIO_LABELS


def run_gradio_evaluation(selection: str | None = None):
    label = (selection or "").strip()
    try:
        if not label or label.lower() == "all":
            result = run_pipeline(force=True)
        else:
            result = run_pipeline(label, force=True)
    except ValueError as exc:
        snapshot = get_persisted_results()
        html = snapshot.get("leaderboard_html") or f"<p>{exc}</p>"
        return html, snapshot.get("candidate_details", [])
    except Exception as exc:  # pragma: no cover - defensive fallback
        snapshot = get_persisted_results()
        html = snapshot.get("leaderboard_html") or ""
        if not html:
            html = f"<p>Benchmark failed: {exc}</p>"
        return html, snapshot.get("candidate_details", [])

    html = result.get("leaderboard_html") or "<p>No results yet.</p>"
    details = result.get("candidate_details", [])
    return html, details


def _build_blocks_demo() -> gr.Blocks:
    with gr.Blocks(title="OCR Benchmark") as demo:
        gr.Markdown("## Benchmark\nChoose a scenario or run all to refresh the persisted results.")
        scenario_dropdown = gr.Dropdown(
            choices=SCENARIO_CHOICES,
            value="all",
            label="Scenario to run",
        )
        run_button = gr.Button("Run evaluation", variant="primary")
        leaderboard_output = gr.HTML(label="Leaderboard", value=INITIAL_HTML)
        candidate_output = gr.JSON(label="Candidate outputs", value=INITIAL_CANDIDATES)

        run_button.click(
            fn=run_gradio_evaluation,
            inputs=[scenario_dropdown],
            outputs=[leaderboard_output, candidate_output],
        )
    return demo


def _build_interface_demo() -> gr.Interface:
    description = "Enter a scenario label (or leave blank for all) to refresh the benchmark."

    if hasattr(gr, "Interface"):
        return gr.Interface(
            fn=run_gradio_evaluation,
            inputs=["text"],
            outputs=["html", "json"],
            title="OCR Benchmark",
            description=description,
        )

    raise RuntimeError("Gradio version does not support Blocks or Interface APIs.")


def build_demo():
    if hasattr(gr, "Blocks"):
        return _build_blocks_demo()
    return _build_interface_demo()


if __name__ == "__main__":
    build_demo().launch()
