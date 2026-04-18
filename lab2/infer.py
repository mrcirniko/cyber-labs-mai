from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


DEFAULT_MODEL = "qwen2.5:0.5b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_PROMPTS_PATH = Path("prompts.json")
DEFAULT_REPORT_PATH = Path("reports/inference_report.md")
DEFAULT_RESULTS_PATH = Path("reports/inference_results.json")
EXPECTED_PROMPT_COUNT = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 10 prompts through a local Ollama server and save an inference report."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model name. Default: {DEFAULT_MODEL}")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help=f"Ollama base URL. Default: {DEFAULT_OLLAMA_URL}")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS_PATH, help="Path to prompts.json.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH, help="Path to the Markdown report.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH, help="Path to the JSON results file.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Request timeout in seconds.")
    return parser.parse_args()


def load_prompts(path: Path) -> list[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Prompts file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in prompts file {path}: {exc}") from exc

    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise SystemExit("prompts.json must contain a JSON array of strings.")

    prompts = [item.strip() for item in data]
    if any(not prompt for prompt in prompts):
        raise SystemExit("All prompts must be non-empty strings.")
    if len(prompts) != EXPECTED_PROMPT_COUNT:
        raise SystemExit(f"Expected {EXPECTED_PROMPT_COUNT} prompts, got {len(prompts)}.")
    return prompts


def call_ollama(base_url: str, model: str, prompt: str, timeout: float) -> str:
    endpoint = f"{base_url.rstrip('/')}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}

    try:
        response = requests.post(endpoint, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not connect to Ollama at {endpoint}. Start the server with `ollama serve`."
        ) from exc

    if response.status_code != 200:
        details = response.text.strip()
        raise RuntimeError(
            f"Ollama returned HTTP {response.status_code}. "
            f"Check that model `{model}` is installed with `ollama pull {model}`. Details: {details}"
        )

    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Ollama returned a non-JSON response: {response.text[:500]}") from exc

    answer = data.get("response")
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError(f"Ollama response does not contain a non-empty `response` field: {data}")
    return answer.strip()


def run_inference(prompts: list[str], base_url: str, model: str, timeout: float) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for index, prompt in enumerate(prompts, start=1):
        print(f"[{index}/{len(prompts)}] sending prompt")
        answer = call_ollama(base_url=base_url, model=model, prompt=prompt, timeout=timeout)
        results.append({"prompt": prompt, "response": answer})
    return results


def write_json_results(path: Path, model: str, base_url: str, results: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model,
        "ollama_url": base_url,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def escape_markdown_table_cell(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def write_markdown_report(path: Path, model: str, results: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Отчет инференса",
        "",
        f"Модель: `{model}`",
        "",
        "| Запрос к LLM | Ответ LLM |",
        "| --- | --- |",
    ]
    for item in results:
        prompt = escape_markdown_table_cell(item["prompt"])
        response = escape_markdown_table_cell(item["response"])
        lines.append(f"| {prompt} | {response} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args.prompts)
    results = run_inference(prompts=prompts, base_url=args.ollama_url, model=args.model, timeout=args.timeout)
    write_json_results(path=args.results, model=args.model, base_url=args.ollama_url, results=results)
    write_markdown_report(path=args.report, model=args.model, results=results)
    print(f"Saved Markdown report: {args.report}")
    print(f"Saved JSON results: {args.results}")


if __name__ == "__main__":
    main()
