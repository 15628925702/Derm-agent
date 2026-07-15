from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.case_qa import (
    ask_qa_session,
    create_direct_qa_session,
    create_qa_session,
    export_qa_session,
    load_qa_session,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive DermAgent QA chat CLI.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--record", type=Path, help="Path to execution record JSON/JSONL.")
    source.add_argument("--session", type=Path, help="Path to an existing QA session JSON.")
    source.add_argument("--image", type=Path, help="Direct image path for Hulu-Med QA without DermAgent.")
    parser.add_argument("--case-id", type=str, default=None, help="Optional case id when the record file contains multiple cases.")
    parser.add_argument("--session-label", type=str, default="", help="Optional label for a new QA session.")
    parser.add_argument("--dataset-name", type=str, default="direct_hulumed", help="Dataset name label for direct image sessions.")
    parser.add_argument("--metadata-json", type=Path, default=None, help="Optional JSON file with clinical metadata for direct image sessions.")
    parser.add_argument("--workflow-json", type=Path, default=None, help="Optional JSON file with workflow context for direct image sessions.")
    parser.add_argument("--context-note", type=str, default="", help="Optional free-text context note for direct image sessions.")
    parser.add_argument("--model", type=str, default="Hulu-Med-4B")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8013/v1")
    parser.add_argument("--api-key", type=str, default="EMPTY")
    parser.add_argument("--audience-mode", choices=["patient", "doctor"], default="patient")
    parser.add_argument("--include-image", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--ask", type=str, default="", help="Ask one question and exit.")
    parser.add_argument("--export", type=Path, default=None, help="Export the final session to a file on exit.")
    parser.add_argument("--export-format", choices=["json", "jsonl", "txt", "md"], default=None)
    return parser.parse_args()


def _print_session_header(session_path: Path) -> None:
    session = load_qa_session(session_path)
    ctx = dict(session.context_snapshot or {})
    baseline = dict(ctx.get("baseline_diagnosis", {}) or {})
    agent_final = dict(ctx.get("agent_final_diagnosis", {}) or {})
    print(f"session_id: {session.session_id}")
    print(f"session_path: {session_path}")
    print(f"source_mode: {session.source_mode}")
    print(f"case_id: {ctx.get('case_id', '')}")
    print(f"dataset_name: {ctx.get('dataset_name', '')}")
    print(f"image_path: {ctx.get('image_path', '')}")
    print(f"baseline_final_diagnosis: {baseline.get('final_diagnosis', '')}")
    print(f"agent_final_diagnosis: {agent_final.get('final_diagnosis', '')}")
    print("")


def _load_optional_json(path: Path | None) -> dict:
    if not path:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _print_help() -> None:
    print("Commands:")
    print("  /help                     Show this help")
    print("  /show                     Show current session summary")
    print("  /suggest                  Show suggested follow-up questions")
    print("  /mode patient|doctor      Change audience mode")
    print("  /image on|off             Toggle include-image for future turns")
    print("  /export PATH [FORMAT]     Export session to json/jsonl/txt/md")
    print("  /exit                     Save and exit")
    print("")


def main() -> int:
    args = parse_args()

    if args.session:
        session_path = args.session.resolve()
        session = load_qa_session(session_path)
        audience_mode = str(session.audience_mode_default or args.audience_mode).strip().lower() or "patient"
        include_image = bool(session.include_image_default or args.include_image)
    elif args.image:
        clinical_metadata = _load_optional_json(args.metadata_json)
        workflow_context = _load_optional_json(args.workflow_json)
        session, session_path = create_direct_qa_session(
            image_path=args.image.resolve(),
            session_label=args.session_label,
            case_id=args.case_id,
            dataset_name=args.dataset_name,
            clinical_metadata=clinical_metadata,
            workflow_context=workflow_context,
            context_note=args.context_note,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            audience_mode=args.audience_mode,
            include_image=True,
        )
        audience_mode = args.audience_mode
        include_image = True
    else:
        session, session_path = create_qa_session(
            execution_record_path=args.record.resolve(),
            case_id=args.case_id,
            session_label=args.session_label,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            audience_mode=args.audience_mode,
            include_image=args.include_image,
        )
        audience_mode = args.audience_mode
        include_image = bool(args.include_image)

    _print_session_header(session_path)

    if args.ask:
        _, turn, _ = ask_qa_session(
            session_id_or_path=session_path,
            question=args.ask,
            audience_mode=audience_mode,
            include_image=include_image,
            max_tokens=args.max_tokens,
        )
        print(f"Q: {args.ask}")
        print(f"A: {turn['answer']}")
        if turn.get("follow_up_questions"):
            print("follow_up_questions:")
            for item in turn["follow_up_questions"]:
                print(f"  - {item}")
        if args.export:
            export_path = export_qa_session(session_path, args.export, export_format=args.export_format)
            print(f"exported: {export_path}")
        return 0

    _print_help()
    while True:
        try:
            raw = input("qa> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            raw = "/exit"

        if not raw:
            continue

        if raw == "/help":
            _print_help()
            continue
        if raw == "/show":
            _print_session_header(session_path)
            continue
        if raw == "/suggest":
            session = load_qa_session(session_path)
            for item in session.suggested_questions:
                print(f"- {item}")
            continue
        if raw.startswith("/mode "):
            mode = raw.split(" ", 1)[1].strip().lower()
            if mode in {"patient", "doctor"}:
                audience_mode = mode
                print(f"audience_mode={audience_mode}")
            else:
                print("Invalid mode. Use patient or doctor.")
            continue
        if raw.startswith("/image "):
            value = raw.split(" ", 1)[1].strip().lower()
            if value in {"on", "true", "1"}:
                include_image = True
            elif value in {"off", "false", "0"}:
                include_image = False
            else:
                print("Invalid image flag. Use on or off.")
                continue
            print(f"include_image={include_image}")
            continue
        if raw.startswith("/export "):
            tail = raw.split(" ", 1)[1].strip()
            parts = tail.split()
            export_target = Path(parts[0]).expanduser()
            export_format = parts[1].strip().lower() if len(parts) > 1 else args.export_format
            export_path = export_qa_session(session_path, export_target, export_format=export_format)
            print(f"exported: {export_path}")
            continue
        if raw == "/exit":
            if args.export:
                export_path = export_qa_session(session_path, args.export, export_format=args.export_format)
                print(f"exported: {export_path}")
            print("bye")
            return 0

        _, turn, _ = ask_qa_session(
            session_id_or_path=session_path,
            question=raw,
            audience_mode=audience_mode,
            include_image=include_image,
            max_tokens=args.max_tokens,
        )
        print(f"A: {turn['answer']}")
        refs = list(turn.get("evidence_refs", []) or [])
        if refs:
            print(f"refs: {', '.join(str(item) for item in refs)}")
        follow_ups = list(turn.get("follow_up_questions", []) or [])
        if follow_ups:
            print("follow_up_questions:")
            for item in follow_ups:
                print(f"  - {item}")


if __name__ == "__main__":
    raise SystemExit(main())
