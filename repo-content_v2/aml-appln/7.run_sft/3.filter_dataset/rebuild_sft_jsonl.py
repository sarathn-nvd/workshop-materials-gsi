#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from pathlib import Path
from multiprocessing import Pool


CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
LINE_SEPARATOR_CHARS = ("\u2028", "\u2029")


def sanitize_text(text: str) -> str:
    # Remove control characters that can break JSON parsing.
    return CONTROL_CHARS_RE.sub(" ", text)


def try_parse_json(line: str):
    try:
        return json.loads(line), None
    except json.JSONDecodeError as e:
        return None, e


def escape_control_chars_in_strings(text: str) -> str:
    # Escape control chars that appear inside JSON strings, including raw newlines.
    out = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                out.append(ch)
                escaped = False
                continue
            if ch == "\\":
                out.append(ch)
                escaped = True
                continue
            if ch == '"':
                out.append(ch)
                in_string = False
                continue
            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                out.append("\\r")
                continue
            if ch == "\t":
                out.append("\\t")
                continue
            if ch == "\u2028":
                out.append("\\u2028")
                continue
            if ch == "\u2029":
                out.append("\\u2029")
                continue
            if ord(ch) < 0x20:
                out.append(f"\\u{ord(ch):04x}")
                continue
            out.append(ch)
        else:
            if ch == '"':
                out.append(ch)
                in_string = True
            else:
                out.append(ch)
    return "".join(out)


_SPLITLINES_ESCAPES = (
    ("\x0b", "\\u000b"),
    ("\x0c", "\\u000c"),
    ("\x1c", "\\u001c"),
    ("\x1d", "\\u001d"),
    ("\x1e", "\\u001e"),
    ("\x85", "\\u0085"),
    ("\u2028", "\\u2028"),
    ("\u2029", "\\u2029"),
)


def safe_json_dumps(obj) -> str:
    text = json.dumps(obj, ensure_ascii=False)
    # Prevent ChatDataset's str.splitlines() reader from splitting a single
    # record at any character it recognizes as a line break. json.dumps with
    # ensure_ascii=False leaves raw \x0b, \x0c, \x1c-\x1e, \x85, U+2028, U+2029
    # alone -- splitlines() splits on all of them, which silently halves the
    # affected record and produces an "Unterminated string" parse error at
    # training time. Escape them here.
    for raw, esc in _SPLITLINES_ESCAPES:
        if raw in text:
            text = text.replace(raw, esc)
    return text


def ensure_last_assistant(obj):
    if not isinstance(obj, dict):
        return obj, False, "not_a_dict"
    messages = obj.get("messages")
    if not isinstance(messages, list):
        return obj, False, "missing_messages"

    # Trim trailing non-assistant messages
    changed = False
    while messages and isinstance(messages[-1], dict) and messages[-1].get("role") != "assistant":
        messages.pop()
        changed = True

    if not messages:
        return obj, changed, "no_assistant_turn"
    if not isinstance(messages[-1], dict) or messages[-1].get("role") != "assistant":
        return obj, changed, "no_assistant_turn"

    obj["messages"] = messages
    return obj, changed, None


def process_chunk(args):
    start_line, raw_lines, on_fail, enforce_last_assistant = args
    out_lines = []
    bad_lines = []
    stats = {
        "total": 0,
        "fixed": 0,
        "skipped": 0,
        "bad_decode": 0,
        "bad_json": 0,
    }

    for offset, raw in enumerate(raw_lines):
        line_no = start_line + offset
        stats["total"] += 1
        try:
            line = raw.decode("utf-8")
        except UnicodeDecodeError:
            stats["bad_decode"] += 1
            line = raw.decode("utf-8", errors="replace")

        line = line.rstrip("\n\r")
        obj, err = try_parse_json(line)
        if obj is None:
            stats["bad_json"] += 1
            sanitized = sanitize_text(line)
            obj, err = try_parse_json(sanitized)
            if obj is not None:
                stats["fixed"] += 1
            else:
                if on_fail == "skip":
                    stats["skipped"] += 1
                    bad_lines.append(f"{line_no}\tjson_decode_failed\t{err}\n")
                    continue
                if on_fail == "keep_raw":
                    bad_lines.append(f"{line_no}\tjson_decode_failed\t{err}\n")
                    out_lines.append(
                        json.dumps(
                            {"_raw": line, "_error": str(err)}, ensure_ascii=False
                        )
                        + "\n"
                    )
                    continue
                bad_lines.append(f"{line_no}\tjson_decode_failed\t{err}\n")
                stats["skipped"] += 1
                continue

        if enforce_last_assistant:
            obj, changed, err = ensure_last_assistant(obj)
            if err == "no_assistant_turn":
                stats["bad_json"] += 1
                if on_fail == "keep_raw":
                    out_lines.append(
                        json.dumps(
                            {"_raw": line, "_error": "no_assistant_turn"},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                else:
                    stats["skipped"] += 1
                bad_lines.append(f"{line_no}\tno_assistant_turn\n")
                continue
            if changed:
                stats["fixed"] += 1

        out_lines.append(safe_json_dumps(obj) + "\n")

    return out_lines, bad_lines, stats


def _stream_rebuild(
    fin,
    fout,
    fbad,
    *,
    on_fail: str,
    enforce_last_assistant: bool,
):
    stats = {
        "total": 0,
        "fixed": 0,
        "skipped": 0,
        "bad_decode": 0,
        "bad_json": 0,
    }
    buffer = []
    in_string = False
    escaped = False
    depth = 0
    started = False
    line_no = 1
    obj_start_line = 1
    obj_line_count = 0
    fixed_current = False

    while True:
        chunk = fin.read(1024 * 1024)
        if not chunk:
            break
        for ch in chunk:
            if ch == "\ufffd":
                stats["bad_decode"] += 1
                if started:
                    fixed_current = True
            if ch == "\n":
                line_no += 1

            if not started:
                if ch.isspace():
                    continue
                if ch != "{":
                    # Skip until the start of a JSON object.
                    continue
                started = True
                depth = 1
                obj_start_line = line_no
                obj_line_count = 1
                fixed_current = False
                in_string = False
                escaped = False
                buffer = ["{"]
                continue

            # Inside an object.
            if ch == "\n":
                obj_line_count += 1

            if in_string:
                if escaped:
                    buffer.append(ch)
                    escaped = False
                    continue
                if ch == "\\":
                    buffer.append(ch)
                    escaped = True
                    continue
                if ch == '"':
                    buffer.append(ch)
                    in_string = False
                    continue
                if ch == "\n":
                    buffer.append("\\n")
                    fixed_current = True
                    continue
                if ch == "\r":
                    buffer.append("\\r")
                    fixed_current = True
                    continue
                if ch == "\t":
                    buffer.append("\\t")
                    fixed_current = True
                    continue
                if ch == "\u2028":
                    buffer.append("\\u2028")
                    fixed_current = True
                    continue
                if ch == "\u2029":
                    buffer.append("\\u2029")
                    fixed_current = True
                    continue
                if ord(ch) < 0x20:
                    buffer.append(f"\\u{ord(ch):04x}")
                    fixed_current = True
                    continue
                buffer.append(ch)
                continue

            # Not in string.
            if ch == '"':
                buffer.append(ch)
                in_string = True
                continue
            if ch == "{":
                depth += 1
                buffer.append(ch)
                continue
            if ch == "}":
                depth -= 1
                buffer.append(ch)
                if depth == 0:
                    obj_text = "".join(buffer)
                    try:
                        obj = json.loads(obj_text)
                        if enforce_last_assistant:
                            obj, changed, err = ensure_last_assistant(obj)
                            if err == "no_assistant_turn":
                                stats["bad_json"] += 1
                                if on_fail == "keep_raw":
                                    fout.write(
                                        json.dumps(
                                            {"_raw": obj_text, "_error": "no_assistant_turn"},
                                            ensure_ascii=False,
                                        )
                                        + "\n"
                                    )
                                else:
                                    stats["skipped"] += 1
                                fbad.write(
                                    f"{obj_start_line}-{line_no}\tno_assistant_turn\n"
                                )
                                # Reset for next object
                                buffer = []
                                started = False
                                continue
                            if changed:
                                stats["fixed"] += 1
                        fout.write(safe_json_dumps(obj) + "\n")
                        stats["total"] += 1
                        if fixed_current or obj_line_count > 1:
                            stats["fixed"] += 1
                    except json.JSONDecodeError as e:
                        stats["bad_json"] += 1
                        if on_fail == "keep_raw":
                            fout.write(
                                json.dumps(
                                    {"_raw": obj_text, "_error": str(e)},
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                        else:
                            stats["skipped"] += 1
                        fbad.write(
                            f"{obj_start_line}-{line_no}\tjson_decode_failed\t{e}\n"
                        )
                    # Reset for next object
                    buffer = []
                    started = False
                continue
            buffer.append(ch)

    if started:
        stats["bad_json"] += 1
        obj_text = "".join(buffer)
        if on_fail == "keep_raw":
            fout.write(
                json.dumps({"_raw": obj_text, "_error": "unterminated_json"}, ensure_ascii=False)
                + "\n"
            )
        else:
            stats["skipped"] += 1
        fbad.write(f"{obj_start_line}-EOF\tjson_decode_failed\tunterminated_json\n")

    return stats


def reconstruct_jsonl(
    input_path: Path,
    output_path: Path,
    bad_path: Path,
    on_fail: str,
    workers: int,
    chunk_lines: int,
    multiline: bool,
    max_buffer_lines: int,
    enforce_last_assistant: bool,
):
    total = 0
    fixed = 0
    skipped = 0
    bad_decode = 0
    bad_json = 0

    with input_path.open("rb") as fin, output_path.open("w", encoding="utf-8") as fout, bad_path.open(
        "w", encoding="utf-8"
    ) as fbad:
        if multiline:
            text_stream = input_path.open("r", encoding="utf-8", errors="replace")
            stats = _stream_rebuild(
                text_stream,
                fout,
                fbad,
                on_fail=on_fail,
                enforce_last_assistant=enforce_last_assistant,
            )
            text_stream.close()
            total += stats["total"]
            fixed += stats["fixed"]
            skipped += stats["skipped"]
            bad_decode += stats["bad_decode"]
            bad_json += stats["bad_json"]
        elif workers <= 1:
            start_line = 1
            buffer = []
            for line_no, raw in enumerate(fin, 1):
                buffer.append(raw)
                if len(buffer) >= chunk_lines:
                    out_lines, bad_lines, stats = process_chunk(
                        (start_line, buffer, on_fail, enforce_last_assistant)
                    )
                    fout.writelines(out_lines)
                    fbad.writelines(bad_lines)
                    total += stats["total"]
                    fixed += stats["fixed"]
                    skipped += stats["skipped"]
                    bad_decode += stats["bad_decode"]
                    bad_json += stats["bad_json"]
                    buffer = []
                    start_line = line_no + 1
            if buffer:
                out_lines, bad_lines, stats = process_chunk(
                    (start_line, buffer, on_fail, enforce_last_assistant)
                )
                fout.writelines(out_lines)
                fbad.writelines(bad_lines)
                total += stats["total"]
                fixed += stats["fixed"]
                skipped += stats["skipped"]
                bad_decode += stats["bad_decode"]
                bad_json += stats["bad_json"]
        else:
            def task_iter():
                start_line = 1
                buffer = []
                for line_no, raw in enumerate(fin, 1):
                    buffer.append(raw)
                    if len(buffer) >= chunk_lines:
                        yield (start_line, buffer, on_fail, enforce_last_assistant)
                        buffer = []
                        start_line = line_no + 1
                if buffer:
                    yield (start_line, buffer, on_fail, enforce_last_assistant)

            with Pool(processes=workers) as pool:
                for out_lines, bad_lines, stats in pool.imap(process_chunk, task_iter(), chunksize=1):
                    fout.writelines(out_lines)
                    fbad.writelines(bad_lines)
                    total += stats["total"]
                    fixed += stats["fixed"]
                    skipped += stats["skipped"]
                    bad_decode += stats["bad_decode"]
                    bad_json += stats["bad_json"]

    return {
        "total": total,
        "fixed": fixed,
        "skipped": skipped,
        "bad_decode": bad_decode,
        "bad_json": bad_json,
    }


def rebuild_one(input_path: Path, output_path: Path, bad_path: Path, args) -> dict:
    return reconstruct_jsonl(
        input_path,
        output_path,
        bad_path,
        args.on_fail,
        args.workers,
        args.chunk_lines,
        args.multiline,
        args.max_buffer_lines,
        args.ensure_last_assistant,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild JSONL line-by-line with UTF-8 and JSON sanitation."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input JSONL file or directory containing final_data files",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output JSONL file or output directory for final_data files",
    )
    parser.add_argument(
        "--bad",
        type=Path,
        default=None,
        help="Path to write bad line report (default: <output>.bad.tsv)",
    )
    parser.add_argument(
        "--on-fail",
        choices=["skip", "keep_raw"],
        default="skip",
        help="What to do if a line still fails JSON parsing",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(32, os.cpu_count() or 1)),
        help="Number of worker processes (default: min(32, cpu_count))",
    )
    parser.add_argument(
        "--chunk-lines",
        type=int,
        default=20000,
        help="Lines per chunk sent to workers",
    )
    parser.add_argument(
        "--multiline",
        action="store_true",
        help="Attempt to assemble JSON objects that span multiple lines",
    )
    parser.add_argument(
        "--max-buffer-lines",
        type=int,
        default=5000,
        help="Max lines to buffer for a single JSON object in multiline mode",
    )
    parser.add_argument(
        "--ensure-last-assistant",
        action="store_true",
        help="Drop trailing non-assistant turns so the last message is assistant",
    )
    args = parser.parse_args()

    if args.multiline and args.workers > 1:
        sys.stderr.write("multiline mode forces --workers=1\n")
        args.workers = 1

    if args.input.is_dir():
        if not args.output.exists():
            args.output.mkdir(parents=True, exist_ok=True)
        elif not args.output.is_dir():
            raise ValueError("When input is a directory, output must be a directory.")

        targets = [
            "sft_mixed.chunk.00.jsonl",
            "sft_mixed.test.jsonl",
            "sft_mixed.val.jsonl",
        ]
        total_stats = {"total": 0, "fixed": 0, "skipped": 0, "bad_decode": 0, "bad_json": 0}
        for name in targets:
            in_path = args.input / name
            if not in_path.exists():
                sys.stderr.write(f"missing file: {in_path}\n")
                continue
            out_path = args.output / name
            bad_path = out_path.with_suffix(out_path.suffix + ".bad.tsv")
            stats = rebuild_one(in_path, out_path, bad_path, args)
            sys.stdout.write(
                f"file={name} "
                + " ".join(f"{k}={v}" for k, v in stats.items())
                + f" bad_report={bad_path}\n"
            )
            for k in total_stats:
                total_stats[k] += stats[k]
        sys.stdout.write(
            "TOTAL "
            + " ".join(f"{k}={v}" for k, v in total_stats.items())
            + "\n"
        )
    else:
        if args.bad is None:
            args.bad = args.output.with_suffix(args.output.suffix + ".bad.tsv")
        stats = rebuild_one(args.input, args.output, args.bad, args)
        sys.stdout.write(
            "done "
            + " ".join(f"{k}={v}" for k, v in stats.items())
            + f" bad_report={args.bad}\n"
        )


if __name__ == "__main__":
    main()

