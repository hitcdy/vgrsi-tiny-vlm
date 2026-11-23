"""Render paper/report.md into the submission .docx using the course template's styles.

    python -u paper/build_docx.py                 # needs paper/report.md
    python -u paper/build_docx.py --out ../Final-Report-VGRSI.docx

Markdown subset supported: '#'..'####', paragraphs, '- ' bullets, '> ' notes,
pipe tables, '{{TABLE1}}' / '{{TABLE4}}' (inlined from paper/*.md), '{{FIG2}}'
(image + caption from paper/figs, real file preferred over mock_*), and the
citable tokens <A> <B> <C> <D> <E> <DELTA> filled from paper/numbers.json.
Anything still unresolved is reported at the end instead of being guessed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(os.path.dirname(ROOT), "Final-Report-Template-arXiv-style.docx")

FIG_CAPTIONS = {
    "FIG1": "Fig. 1. The VGRSI loop. The analytic verifier decides which self-generated samples "
            "may enter the training pool; the evaluation split is never trained on.",
    "FIG2": "Fig. 2. Per-type strict accuracy across the five arms.",
    "FIG3": "Fig. 3. Training loss (answer tokens only) for SFT and the VGRSI rounds.",
    "FIG4": "Fig. 4. Verifier-gated pool growth over self-training rounds.",
    "FIG5": "Fig. 5. Accepted-cell ambiguity of the analytic verifier: a straddling object "
            "is accepted in several cells (grey), which is why strict judging matters.",
    "FIG6": "Fig. 6. Demo output: three real /predict responses with the verifier overlay.",
}
FIG_FILES = {
    "FIG1": ["fig1_loop.png"],
    "FIG2": ["g4_fig2_accuracy.png", "fig2_accuracy.png", "mock_fig2_accuracy.png"],
    "FIG3": ["g4_fig3_loss.png", "fig3_loss.png", "mock_fig3_loss.png"],
    "FIG4": ["g4_fig4_pool.png", "fig4_pool.png", "mock_fig4_pool.png"],
    "FIG5": ["g4_fig5_verifier.png", "fig5_verifier.png", "mock_fig5_verifier.png"],
    "FIG6": ["fig6_demo.png"],
}
TABLE_TOKENS = {"TABLE1": "table1_main_g4.md", "TABLE2": "table2_ablation.md",
                "TABLE3": "table3_demo.md", "TABLE4": "table4_verifier.md"}



def load_numbers():
    path = os.path.join(ROOT, "paper", "numbers.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        num = json.load(f)
    if num.get("backend") == "mock":
        print("[docx] numbers.json is from the MOCK backend - refusing to substitute "
              "simulated values; re-run with --backend real")
        return {}
    return num


def substitute(text, num):
    mapping = {
        "A": num.get("zero_shot_acc"), "B": num.get("sft_acc"), "C": num.get("arm_c_acc"),
        "D": num.get("seed_only_acc"), "E": num.get("vgrs_acc"),
        "DELTA": num.get("abs_gain_pts"), "REL": num.get("rel_gain_pct"),
        "VSSFT": num.get("vgrs_vs_sft_rel_pct"),
    }
    for k, v in mapping.items():
        if v is None:
            continue
        text = re.sub(r"⟨\s*" + k + r"\s*⟩", str(v), text)
    return text


def parse_md_table(lines):
    rows = []
    for ln in lines:
        ln = ln.strip()
        if not ln.startswith("|"):
            continue
        if re.fullmatch(r"\|[\s:\-\|]+\|", ln):
            continue
        cells = [c.strip().replace("**", "") for c in ln.strip("|").split("|")]
        rows.append(cells)
    return rows


def read_table_md(name):
    if not name:
        return None
    path = os.path.join(ROOT, "paper", name)
    if not os.path.exists(path):
        return None
    keep = []
    for ln in open(path, encoding="utf-8"):
        s = ln.rstrip("\n")
        if s.startswith("#") or s.startswith("> **MOCK"):
            continue
        keep.append(s)
    rows = parse_md_table(keep)
    return rows or None


def add_table(doc, rows, caption=None):
    if not rows:
        doc.add_paragraph("[table not generated yet]")
        return
    ncol = max(len(r) for r in rows)
    t = doc.add_table(rows=0, cols=ncol)
    t.style = "Table Grid"
    for i, r in enumerate(rows):
        cells = t.add_row().cells
        for j in range(ncol):
            cells[j].text = r[j] if j < len(r) else ""
            for p in cells[j].paragraphs:
                for run in p.runs:
                    run.font.size = Pt(8.5)
                    run.bold = (i == 0)
    if caption:
        p = doc.add_paragraph(caption)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.runs[0].font.size = Pt(9)


def add_figure(doc, token):
    for name in FIG_FILES.get(token, []):
        path = os.path.join(ROOT, "paper", "figs", name)
        if os.path.exists(path):
            doc.add_picture(path, width=Inches(6.1))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap = doc.add_paragraph(FIG_CAPTIONS.get(token, ""))
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap.runs[0].font.size = Pt(9)
            if name.startswith("mock_"):
                warn = doc.add_paragraph("NB: figure rendered by the mock runner - re-run "
                                         "with --backend real before submission.")
                warn.runs[0].font.size = Pt(8)
            return True
    doc.add_paragraph(f"[figure {token} missing]")
    return False



def add_page_number_footer(doc):
    """Centred PAGE field in the footer (the template's footer was emptied when the
    body was rebuilt)."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn as _qn

    footer = doc.sections[0].footer
    para = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = para.add_run()
    for el, attrs, text in (("w:fldChar", {"w:fldCharType": "begin"}, None),
                            ("w:instrText", {"xml:space": "preserve"}, "PAGE"),
                            ("w:fldChar", {"w:fldCharType": "end"}, None)):
        node = OxmlElement(el)
        for k, v in attrs.items():
            node.set(_qn(k), v)
        if text is not None:
            node.text = text
        run._r.append(node)
    for r in para.runs:
        r.font.size = Pt(9)


def build(md_path, out_path):
    num = load_numbers()
    text = substitute(open(md_path, encoding="utf-8").read(), num)
    lines = text.splitlines()
    doc = Document(TEMPLATE)
    body = doc.element.body
    # keep the final sectPr (page size / margins), otherwise python-docx cannot
    # compute a table width afterwards
    sect = body.find(qn("w:sectPr"))
    for child in list(body):
        if child is sect:
            continue
        body.remove(child)

    unresolved = set()
    i = 0
    while i < len(lines):
        ln = lines[i]
        stripped = ln.strip()
        tok = re.fullmatch(r"\{\{(TABLE\d|FIG\d)\}\}", stripped)
        if tok:
            key = tok.group(1)
            if key.startswith("TABLE"):
                add_table(doc, read_table_md(TABLE_TOKENS.get(key, "")),
                          caption={"TABLE1": "Table 1. Main results (mean over seeds).",
                                   "TABLE4": "Table 4. Analytic verifier reliability (model-free).",
                                   "TABLE2": "Table 2. Ablation (pool policy / rounds / correction path).",
                                   "TABLE3": "Table 3. Demonstration system architecture."}.get(key))
            else:
                add_figure(doc, key)
            i += 1
            continue
        if stripped.startswith("```"):
            i += 1
            block = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i]); i += 1
            i += 1
            for k, cl in enumerate(block):
                para = doc.add_paragraph()
                run = para.add_run(cl.rstrip())
                run.font.name = "Consolas"
                run.font.size = Pt(8.2)
                para.paragraph_format.space_after = Pt(0)
                para.paragraph_format.space_before = Pt(0)
                para.paragraph_format.line_spacing = 1.0
                if k < len(block) - 1:
                    para.paragraph_format.keep_with_next = True
            doc.add_paragraph()
            continue
        if stripped.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i]); i += 1
            add_table(doc, parse_md_table(block))
            continue
        for m in re.findall(r"⟨[^⟩]{1,12}⟩", ln):
            unresolved.add(m)
        if stripped.startswith("# "):
            doc.add_heading(stripped[2:], level=0)
        elif stripped.startswith("## "):
            doc.add_heading(stripped[3:], level=1)
        elif stripped.startswith("### "):
            doc.add_heading(stripped[4:], level=2)
        elif stripped.startswith("#### "):
            doc.add_heading(stripped[5:], level=3)
        elif stripped.startswith("- "):
            doc.add_paragraph(stripped[2:], style="List Bullet")
        elif stripped.startswith("> "):
            p = doc.add_paragraph(stripped[2:])
            for r in p.runs:
                r.italic = True
                r.font.size = Pt(9)
        elif stripped:
            doc.add_paragraph(stripped.replace("**", ""))
        i += 1

    add_page_number_footer(doc)
    doc.save(out_path)
    print(f"[docx] wrote {out_path}")
    if unresolved:
        print("[docx] UNRESOLVED tokens (fill paper/numbers.json or the text): "
              + ", ".join(sorted(unresolved)))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=os.path.join(ROOT, "paper", "report.md"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(ROOT), "Final-Report-VGRSI.docx"))
    a = ap.parse_args()
    if not os.path.exists(a.md):
        print(f"[docx] missing source {a.md}"); return 2
    return build(a.md, a.out)


if __name__ == "__main__":
    sys.exit(main())