"""Render a bound JSON model card; never compute or substitute benchmark results.

Markdown/HTML retain a complete JSON appendix. The compact PDF/PowerPoint have
three pages/slides and ship alongside the unchanged source JSON and its hashes.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import html
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import tempfile
import textwrap
import zipfile

BINDINGS = ("checkpoint_sha256", "recipe_sha256", "contract_hash")
FILENAMES = ("MODEL_CARD.md", "MODEL_CARD.html", "MODEL_CARD.pptx", "MODEL_CARD.pdf", "card.source.json")


def read_card(path):
    data = Path(path).read_bytes()

    def finite(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Nonfinite JSON evidence")
        return number

    def reject(value):
        raise ValueError("Nonfinite JSON evidence: " + value)

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON evidence field: " + key)
            result[key] = value
        return result

    return json.loads(data, parse_constant=reject, parse_float=finite,
                      object_pairs_hook=unique), data


def validate_card(card):
    if not isinstance(card, dict) or card.get("format") != "vcm-model-card-v2":
        raise ValueError("Expected vcm-model-card-v2 format")
    bindings = card.get("bindings")
    if not isinstance(bindings, dict):
        raise ValueError("Missing candidate bindings")
    for key in BINDINGS:
        if not isinstance(bindings.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", bindings[key]):
            raise ValueError("Missing actual SHA-256 candidate binding: " + key)

    def walk(value, path="card"):
        if isinstance(value, dict):
            for key, child in value.items():
                if (key == "sha256" or key.endswith("_sha256") or key.endswith("_hash")) and child is not None:
                    if not isinstance(child, str) or not re.fullmatch(r"[0-9a-f]{64}", child):
                        raise ValueError("Invalid SHA-256 hash: " + path + "." + key)
                walk(child, path + "." + key)
        elif isinstance(value, list):
            for child in value:
                walk(child, path)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Nonfinite card evidence: " + path)

    walk(card)
    pi = card.get("pi")
    if pi is not None and not isinstance(pi, dict):
        raise ValueError("Pi fields must be supplied as an object or null")
    if isinstance(pi, dict) and pi.get("status") == "deferred" and pi.get("results") is not None:
        raise ValueError("Deferred Pi cannot contain measured results")


def _value(value):
    if value is None:
        return "not measured"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _compact(value, length=150):
    text = _value(value)
    return text if len(text) <= length else text[:length] + "... [full value in card.source.json]"


def _rows(value, limit=6):
    result = []

    def visit(item, path=""):
        if isinstance(item, dict) and item:
            for key, child in item.items():
                visit(child, path + "." + key if path else key)
        else:
            result.append((path or "Value", _compact(item)))

    visit(value)
    return result[:limit] + ([("Additional fields", "See complete JSON appendix / card.source.json")]
                             if len(result) > limit else [])


def _failed(card):
    if card.get("failed_gates") is not None:
        return _value(card["failed_gates"])
    gates = card.get("gates")
    if isinstance(gates, dict):
        return "; ".join(key for key, value in gates.items() if value is False) or _value(None)
    return _value(None)


def _is_tts(name):
    return any(word in name.lower() for word in ("tts", "synthetic"))


def _voice_tables(value, prefix=""):
    tables = []
    if not isinstance(value, dict):
        return tables
    for key, child in value.items():
        label = prefix + "." + key if prefix else key
        if key in ("per_voice", "by_voice") and isinstance(child, dict):
            columns = ["Voice", "Exact accepted recall", "Strict accuracy", "Every supplied slot",
                       "Negatives", "False accepts", "Faults / missing"]
            data = []
            for voice, observed in child.items():
                observed = observed if isinstance(observed, dict) else {}
                metric = observed.get("all", observed)
                metric = metric if isinstance(metric, dict) else {}

                def first(*names):
                    return next((metric[name] for name in names if name in metric), None)

                joint = metric.get("joint_supplied_slots")
                slots = (joint.get("accuracy") if isinstance(joint, dict) else
                         first("every_slot_accuracy", "all_slots_accuracy", "every_slot_correctness"))
                data.append([voice, _value(first("complete_command_recall", "complete_command_accuracy", "exact_complete_command_accuracy", "recall")),
                             _value(metric.get("strict_accuracy")), _value(slots),
                             _value(first("negative_n", "negative")), _value(metric.get("false_accepts")),
                             _value(metric.get("faults")) + " / " + _value(metric.get("missing"))])
            tables.append((label, columns, data))
        elif isinstance(child, dict):
            tables.extend(_voice_tables(child, label))
    return tables


def _pi_rows(card):
    pi = card.get("pi")
    if not isinstance(pi, dict) or not isinstance(pi.get("results"), dict):
        return _rows(pi, 6)
    receipt = pi["results"]
    rows = [("status", _value(pi.get("status")))]
    for field in ("hardware", "environment", "metrics", "gates"):
        if field in receipt:
            rows += [(field + "." + key, value) for key, value in _rows(receipt[field], 5)]
    return rows


def _pages(card):
    benchmarks = card.get("benchmarks")
    benchmarks = benchmarks if isinstance(benchmarks, dict) else {}
    human = [(key, _rows(value, 7)) for key, value in benchmarks.items() if not _is_tts(key)]
    synthetic = [(key, _rows(value, 5)) for key, value in benchmarks.items() if _is_tts(key)]
    voice = [table for key, value in benchmarks.items() if _is_tts(key)
             for table in _voice_tables(value, key)]
    return [
        {"title": "Model, training and data", "sections": [
            ("Architecture", _rows(card.get("architecture"), 6)),
            ("Training", _rows(card.get("training"), 6)),
            ("Data", _rows(card.get("data", card.get("dataset")), 6)),
            ("Calibration and enrollment", _rows({key: card[key] for key in ("calibration", "enrollment") if key in card} or None, 5))],
         "voice_tables": []},
        {"title": "Human and original benchmark results", "sections": human or [("Benchmarks", _rows(None))],
         "voice_tables": []},
        {"title": "Synthetic results, gates and physical Pi", "sections": synthetic + [
            ("Physical Pi", _pi_rows(card)), ("Gates", _rows(card.get("gates"), 8)),
            ("Selected candidate and receipts", _rows({**card.get("bindings", {}),
                  **({"selected_candidate": card["selected_candidate"]} if "selected_candidate" in card else {})}, 6))],
         "voice_tables": voice},
    ]


def render_markdown(card, source_sha256=None):
    validate_card(card)
    lines = ["# " + _value(card.get("title", "VCM model card")), "",
             "**Status: " + _value(card.get("status")) + "**", "",
             "**Failed gates:** " + _failed(card), "",
             "Source JSON SHA-256: " + _value(source_sha256), ""]
    for page in _pages(card):
        lines += ["## " + page["title"], ""]
        for title, rows in page["sections"]:
            lines += ["### " + title, "", "| Field | Supplied evidence |", "| --- | --- |"]
            for key, value in rows:
                lines.append("| " + str(key).replace("|", "\\|") + " | " + value.replace("|", "\\|").replace("\n", "<br>") + " |")
            lines.append("")
        for title, columns, data in page["voice_tables"]:
            lines += ["### " + title, "", "| " + " | ".join(columns) + " |",
                      "| " + " | ".join("---" for _ in columns) + " |"]
            lines += ["| " + " | ".join(v.replace("|", "\\|") for v in row) + " |" for row in data]
            lines.append("")
    lines += ["## Complete source-card appendix", "",
              "All fields and missed verbatims are preserved below and in `card.source.json`.", "",
              "```json", json.dumps(card, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False), "```", ""]
    return "\n".join(lines)


def render_html(card, source_sha256=None):
    validate_card(card)
    esc = html.escape
    body = ["<header contenteditable=\"true\"><p class=\"eyebrow\">MODEL CARD</p><h1>" + esc(_value(card.get("title", "VCM model card"))) + "</h1>",
            "<p class=\"status\">" + esc(_value(card.get("status"))) + "</p>",
            "<p class=\"gates\"><strong>Failed gates:</strong> " + esc(_failed(card)) + "</p></header>"]
    for page in _pages(card):
        body.append("<article contenteditable=\"true\"><h2>" + esc(page["title"]) + "</h2>")
        for title, rows in page["sections"]:
            body.append("<section><h3>" + esc(title) + "</h3><table><tbody>")
            body += ["<tr><th>" + esc(str(key)) + "</th><td>" + esc(value) + "</td></tr>" for key, value in rows]
            body.append("</tbody></table></section>")
        for title, columns, data in page["voice_tables"]:
            body.append("<section><h3>" + esc(title) + "</h3><table><thead><tr>" + "".join("<th>" + esc(v) + "</th>" for v in columns) + "</tr></thead><tbody>")
            body += ["<tr>" + "".join("<td>" + esc(v) + "</td>" for v in row) + "</tr>" for row in data]
            body.append("</tbody></table></section>")
        body.append("</article>")
    body += ["<details open><summary>Complete source-card appendix</summary><pre>" + esc(json.dumps(card, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)) + "</pre></details>",
             "<footer>Source JSON SHA-256: " + esc(_value(source_sha256)) + "<br>Complete evidence and verbatims remain in card.source.json. Editing this document does not modify the machine card.</footer>"]
    return """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VCM model card</title><style>
:root{color-scheme:light}*{box-sizing:border-box}body{max-width:1040px;margin:36px auto;padding:0 24px;color:#172c3b;background:#fff;font:15px/1.5 Arial,Helvetica,sans-serif}
header{border-top:6px solid #315b70;padding-top:18px;margin-bottom:28px}.eyebrow{font-size:11px;letter-spacing:2px;color:#526c7a}h1{font-size:30px;line-height:1.15;margin:8px 0 18px}h2{font-size:21px;border-bottom:1px solid #c9d4da;padding-bottom:8px}h3{font-size:15px;margin-bottom:8px}.status{font-size:18px;font-weight:700;background:#eef2f5;padding:12px}.gates{border-left:4px solid #a9403b;padding:10px 14px;background:#fcf3f1}
section{margin:18px 0 26px}table{width:100%;border-collapse:collapse;font-size:13px;table-layout:fixed}td,th{padding:8px 10px;border:1px solid #d7dfe3;vertical-align:top;overflow-wrap:anywhere;text-align:left}tbody th{width:29%;background:#f2f5f7;font-weight:600}thead{background:#e9eff2}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:11px/1.45 monospace;background:#f3f5f6;padding:18px}footer{font-size:10px;color:#526c7a;overflow-wrap:anywhere;margin-top:24px}article{break-inside:avoid}button{font:inherit;padding:8px 14px;border:1px solid #acbcc5;background:#f2f5f7;cursor:pointer}
@media print{body{margin:0;padding:0}button{display:none}details{break-before:page}article{break-before:page}article:first-of-type{break-before:auto}}
</style></head><body><button id="save">Save edited HTML</button>""" + "".join(body) + """<script>document.getElementById('save').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob(['<!doctype html>'+document.documentElement.outerHTML],{type:'text/html'}));a.download='MODEL_CARD-edited.html';a.click();URL.revokeObjectURL(a.href);};</script></body></html>"""


def render_pdf(card, source_sha256):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepInFrame

    stream = BytesIO()
    width, height = A4
    styles = {
        "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=19, leading=23, textColor=colors.HexColor("#172c3b")),
        "status": ParagraphStyle("status", fontName="Helvetica-Bold", fontSize=10, leading=14, textColor=colors.HexColor("#a9403b")),
        "section": ParagraphStyle("section", fontName="Helvetica-Bold", fontSize=10, leading=13, spaceBefore=10, spaceAfter=5),
        "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=8, leading=10),
        "note": ParagraphStyle("note", fontName="Helvetica", fontSize=7, leading=9, textColor=colors.HexColor("#526c7a")),
    }

    def paragraph(text, style="cell"):
        return Paragraph(html.escape(str(text)).replace("\n", "<br/>"), styles[style])

    def table(data, widths, header=False):
        obj = Table([[paragraph(v) for v in row] for row in data], colWidths=widths, hAlign="LEFT")
        obj.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("GRID", (0, 0), (-1, -1), .3, colors.HexColor("#d7dfe3")),
                                ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                                ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                                ("BACKGROUND", (0, 0), (-1 if header else 0, 0 if header else -1), colors.HexColor("#eef2f5"))]))
        return obj

    story = []
    for n, page in enumerate(_pages(card)):
        if n:
            story.append(PageBreak())
        story += [paragraph(_value(card.get("title", "VCM model card")), "title"), Spacer(1, 8),
                  paragraph("Status: " + _value(card.get("status")), "status"),
                  paragraph("Failed gates: " + _compact(_failed(card), 270), "note"), Spacer(1, 12),
                  paragraph(page["title"], "section")]
        content = []
        for title, rows in page["sections"]:
            content += [paragraph(title, "section"), table(rows, [135, width - 207])]
        for title, columns, data in page["voice_tables"]:
            content += [paragraph(title, "section"), table([columns] + data, [(width - 72) / len(columns)] * len(columns), True)]
        story.append(KeepInFrame(width - 72, height - 190, content, mode="shrink", hAlign="LEFT"))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 6.5)
        canvas.setFillColor(colors.HexColor("#526c7a"))
        canvas.drawString(36, 25, "Source JSON SHA-256: " + source_sha256)
        canvas.drawString(36, 15, "Complete fields and verbatims: card.source.json | " + str(doc.page) + " / 3")
        canvas.restoreState()

    document = SimpleDocTemplate(stream, pagesize=A4, leftMargin=36, rightMargin=36,
                                 topMargin=30, bottomMargin=36, title=_value(card.get("title", "VCM model card")),
                                 invariant=1,
                                 author=card.get("author") if isinstance(card.get("author"), str) else "")
    document.build(story, onFirstPage=footer, onLaterPages=footer,
                   canvasmaker=Canvas)
    return stream.getvalue()


def render_pptx(card, source_sha256):
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.util import Inches, Pt

    presentation = Presentation()
    presentation.slide_width, presentation.slide_height = Inches(13.333), Inches(7.5)
    metadata = presentation.core_properties
    metadata.title = _value(card.get("title", "VCM model card"))
    metadata.author = card.get("author") if isinstance(card.get("author"), str) else ""
    metadata.last_modified_by = ""
    metadata.created = metadata.modified = datetime(2000, 1, 1)

    def textbox(slide, text, x, y, w, h, size=11, bold=False, color="172c3b"):
        shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        frame = shape.text_frame
        frame.word_wrap = True
        frame.margin_left = frame.margin_right = Inches(.03)
        frame.margin_top = frame.margin_bottom = 0
        for n, line in enumerate(str(text).split("\n")):
            p = frame.paragraphs[0] if n == 0 else frame.add_paragraph()
            p.text = line
            p.font.name, p.font.size, p.font.bold = "Arial", Pt(size), bold
            p.font.color.rgb = RGBColor.from_string(color)
        return shape

    for n, page in enumerate(_pages(card)):
        slide = presentation.slides.add_slide(presentation.slide_layouts[6])
        textbox(slide, _value(card.get("title", "VCM model card")), .35, .15, 12.6, .45, 23, True)
        textbox(slide, "Status: " + _value(card.get("status")), .35, .65, 12.6, .32, 13, True, "a9403b")
        textbox(slide, "Failed gates: " + _compact(_failed(card), 220), .35, 1.0, 12.6, .35, 9)
        textbox(slide, page["title"], .35, 1.4, 12.6, .3, 14, True)
        sections = page["sections"]
        has_voices = bool(page["voice_tables"])
        columns_count = 3 if len(sections) > 4 else 2
        column_width = 12.6 / columns_count
        row_limit, row_pitch = (4, .28) if has_voices else (6, .34)
        positions = [1.85] * columns_count
        for index, (title, rows) in enumerate(sections):
            column = min(range(columns_count), key=lambda c: positions[c])
            x, y = .35 + column * column_width, positions[column]
            textbox(slide, title + (" (summary)" if len(rows) > row_limit else ""), x, y, column_width - .2, .24, 10, True)
            # Receipt hashes remain prominent even when the selected-candidate record is large.
            if title == "Selected candidate and receipts":
                rows = sorted(rows, key=lambda row: (not ("receipt" in row[0] and "sha256" in row[0]),))
            visible = rows[:row_limit]
            key_width = 1.75 if columns_count == 2 else 1.2
            for r, (key, value) in enumerate(visible):
                row_y = y + .29 + r * row_pitch

                def two_lines(text, line_width):
                    lines = textwrap.wrap(str(text), width=line_width) or [""]
                    if len(lines) > 2:
                        lines = [lines[0], lines[1][:max(0, line_width - 7)] + " [JSON]"]
                    return "\n".join(lines)

                body_size = 8 if has_voices or columns_count == 3 else 9
                textbox(slide, two_lines(key, 27 if columns_count == 2 else 21), x, row_y, key_width, row_pitch - .015, body_size, True)
                textbox(slide, two_lines(value, 65 if columns_count == 2 else 41), x + key_width + .04, row_y,
                        column_width - key_width - .25, row_pitch - .015, body_size)
            positions[column] += .42 + len(visible) * row_pitch
        voice_tables = page["voice_tables"][:2]
        for table_index, (title, columns, data) in enumerate(voice_tables):
            table_y = 5.1
            table_width = 12.6 if len(voice_tables) == 1 else 6.15
            table_x = .35 + table_index * 6.45
            textbox(slide, title, table_x, table_y - .26, table_width, .24, 10, True)
            observed_rows = [columns] + data[:6]
            table = slide.shapes.add_table(len(observed_rows), len(columns), Inches(table_x), Inches(table_y),
                                           Inches(table_width), Inches(.28 * len(observed_rows))).table
            for row_index, row in enumerate(observed_rows):
                table.rows[row_index].height = Inches(.28)
                for column_index, value in enumerate(row):
                    cell = table.cell(row_index, column_index)
                    cell.text = value
                    cell.margin_top = cell.margin_bottom = 0
                    cell.margin_left = cell.margin_right = Inches(.04)
                    for p in cell.text_frame.paragraphs:
                        p.font.name, p.font.size, p.font.bold = "Arial", Pt(8 if len(voice_tables) == 1 else 7.5), row_index == 0
                        p.font.color.rgb = RGBColor.from_string("172c3b")
        textbox(slide, "Source JSON SHA-256: " + source_sha256 + " | " + str(n + 1) + " / 3", .35, 7.13, 12.6, .2, 7, color="526c7a")
    raw = BytesIO()
    presentation.save(raw)
    stable = BytesIO()
    with zipfile.ZipFile(raw) as incoming, zipfile.ZipFile(stable, "w", compression=zipfile.ZIP_DEFLATED) as outgoing:
        for name in sorted(incoming.namelist()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            outgoing.writestr(info, incoming.read(name))
    return stable.getvalue()


def _atomic(path, data):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def render(card_path, output_dir):
    source, output = Path(card_path).resolve(), Path(output_dir).resolve()
    if source in [output / name for name in FILENAMES + ("rendering-receipt.json",)]:
        raise ValueError("Rendering would overwrite the source card")
    card, original = read_card(source)
    validate_card(card)
    digest = hashlib.sha256(original).hexdigest()
    artifacts = {
        "MODEL_CARD.md": render_markdown(card, digest).encode("utf8"),
        "MODEL_CARD.html": render_html(card, digest).encode("utf8"),
        "MODEL_CARD.pptx": render_pptx(card, digest), "MODEL_CARD.pdf": render_pdf(card, digest),
        "card.source.json": original,
    }
    receipt = {"format": "vcm-card-rendering-v1", "source_card_sha256": digest,
               "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "artifacts": {name: hashlib.sha256(data).hexdigest() for name, data in artifacts.items()},
               "metrics_source": "supplied JSON only; no benchmark outcomes computed",
               "complete_fields_preserved": True}
    artifacts["rendering-receipt.json"] = (json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf8")
    output.mkdir(parents=True, exist_ok=True)
    for name, data in artifacts.items():
        _atomic(output / name, data)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = render(args.card, args.output)
    print(json.dumps({"source_card_sha256": receipt["source_card_sha256"], "output": str(args.output.resolve()),
                      "artifacts": list(receipt["artifacts"])}, allow_nan=False))


if __name__ == "__main__":
    main()
