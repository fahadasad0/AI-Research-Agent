"""Render a ``PaperDraft`` as an IEEE-style two-column A4 PDF (reportlab, pure Python).

Fonts: DejaVu Serif ships inside matplotlib, so full Unicode (author names with accents,
Greek letters, ...) renders correctly on Streamlit Cloud without extra font files.
"""
from __future__ import annotations

import io
import re
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, FrameBreak, Image, KeepTogether, NextPageTemplate,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)

from paper_builder import ROMAN, PaperDraft, ref_parts

FONT, BOLD, ITALIC, BOLDITALIC = "PaperSerif", "PaperSerif-Bold", "PaperSerif-Italic", "PaperSerif-BoldItalic"
_FONTS_READY = False


def register_fonts() -> None:
    global _FONTS_READY, FONT, BOLD, ITALIC, BOLDITALIC
    if _FONTS_READY:
        return
    base = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    files = {FONT: "DejaVuSerif.ttf", BOLD: "DejaVuSerif-Bold.ttf",
             ITALIC: "DejaVuSerif-Italic.ttf", BOLDITALIC: "DejaVuSerif-BoldItalic.ttf"}
    try:
        for name, fname in files.items():
            pdfmetrics.registerFont(TTFont(name, str(base / fname)))
        pdfmetrics.registerFontFamily(FONT, normal=FONT, bold=BOLD, italic=ITALIC, boldItalic=BOLDITALIC)
    except Exception:  # noqa: BLE001 - fall back to built-in Times (Latin-1 only)
        FONT, BOLD, ITALIC, BOLDITALIC = "Times-Roman", "Times-Bold", "Times-Italic", "Times-BoldItalic"
    _FONTS_READY = True


def inline(text: str) -> str:
    """Escape XML and convert light markdown (**bold**, *italic*, `code`) to reportlab markup."""
    t = escape(text or "")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\*\w])\*(?!\s)(.+?)(?<!\s)\*(?![\*\w])", r"<i>\1</i>", t)
    return t.replace("`", "")


def _hyphenation_lang(enabled: bool):
    """Return a pyphen language code if hyphenation is requested and pyphen is installed."""
    if not enabled:
        return None
    try:
        import pyphen  # noqa: F401
        return "en_GB"
    except ImportError:
        return None


def _styles(hyphenate: bool = True) -> dict:
    gray = colors.HexColor("#444444")
    lang = _hyphenation_lang(hyphenate)
    hyph = {"hyphenationLang": lang, "uriWasteReduce": 0.3} if lang else {}
    return {
        "title": ParagraphStyle("title", fontName=BOLD, fontSize=16.5, leading=20, alignment=TA_CENTER, spaceAfter=7),
        "author": ParagraphStyle("author", fontName=FONT, fontSize=9.5, leading=12, alignment=TA_CENTER, spaceAfter=2),
        "byline": ParagraphStyle("byline", fontName=ITALIC, fontSize=7.2, leading=9, alignment=TA_CENTER,
                                 textColor=gray, spaceAfter=9),
        "abstract": ParagraphStyle("abstract", fontName=FONT, fontSize=7.9, leading=10.2, alignment=TA_JUSTIFY,
                                   leftIndent=10, rightIndent=10, spaceAfter=4, **hyph),
        "body": ParagraphStyle("body", fontName=FONT, fontSize=7.9, leading=10.1, alignment=TA_JUSTIFY,
                               firstLineIndent=8, spaceAfter=3.5, allowWidows=0, **hyph),
        "h1": ParagraphStyle("h1", fontName=BOLD, fontSize=7.9, leading=10, alignment=TA_CENTER, spaceBefore=7,
                             spaceAfter=3.5, keepWithNext=1),
        "h2": ParagraphStyle("h2", fontName=ITALIC, fontSize=7.9, leading=10, alignment=TA_LEFT, spaceBefore=4,
                             spaceAfter=2, keepWithNext=1),
        "bullet": ParagraphStyle("bullet", fontName=FONT, fontSize=7.6, leading=9.8, alignment=TA_LEFT,
                                 leftIndent=10, firstLineIndent=-7, spaceAfter=1.5),
        "caption": ParagraphStyle("caption", fontName=FONT, fontSize=6.8, leading=8.6, alignment=TA_JUSTIFY,
                                  spaceBefore=2, spaceAfter=6),
        "tcap": ParagraphStyle("tcap", fontName=FONT, fontSize=6.6, leading=8.2, alignment=TA_CENTER,
                               spaceBefore=3, spaceAfter=2, keepWithNext=1),
        "cell": ParagraphStyle("cell", fontName=FONT, fontSize=6.3, leading=7.6, alignment=TA_LEFT),
        "cellb": ParagraphStyle("cellb", fontName=BOLD, fontSize=6.3, leading=7.6, alignment=TA_LEFT),
        "ref": ParagraphStyle("ref", fontName=FONT, fontSize=6.9, leading=8.7, alignment=TA_LEFT,
                              leftIndent=15, firstLineIndent=-15, spaceAfter=1.8, wordWrap="CJK"),
    }


def reference_html(p, n: int) -> str:
    r = ref_parts(p)
    out = f"[{n}]&nbsp; {escape(r['authors'])}, \u201c{escape(r['title'])},\u201d"
    if r["venue"]:
        out += f" <i>{escape(r['venue'])}</i>,"
    out += f" {r['date']}."
    if r["arxiv"]:
        out += f" {escape(r['arxiv'])}."
    if r["doi"]:
        out += f" doi: {escape(r['doi'])}."
    if r["url"]:
        out += f" [Online]. Available: {escape(r['url'])}"
    return out


def build_pdf(draft: PaperDraft, figures: dict) -> bytes:
    """Return the PDF as bytes. ``figures`` maps figure keys to PNG bytes."""
    try:
        return _build_pdf(draft, figures, hyphenate=True)
    except Exception:  # noqa: BLE001 - hyphenation is cosmetic; never let it break the download
        return _build_pdf(draft, figures, hyphenate=False)


def _build_pdf(draft: PaperDraft, figures: dict, hyphenate: bool) -> bytes:
    register_fonts()
    S = _styles(hyphenate)
    page_w, page_h = A4
    lm = rm = 15 * mm
    tm, bm, gutter = 17 * mm, 17 * mm, 6 * mm
    full_w = page_w - lm - rm
    col_w = (full_w - gutter) / 2

    # ---- title block (full width) -------------------------------------------------
    head = [
        Paragraph(inline(draft.title), S["title"]),
        Paragraph(inline(draft.author_line), S["author"]),
        Paragraph(inline(draft.byline), S["byline"]),
        Paragraph(f"<b><i>Abstract\u2014</i></b>{inline(draft.abstract)}", S["abstract"]),
        Paragraph(f"<b><i>Keywords\u2014</i></b>{inline(', '.join(draft.keywords))}", S["abstract"]),
        Spacer(1, 3),
    ]
    head_h = sum(f.wrap(full_w, 10_000)[1] + f.getSpaceBefore() + f.getSpaceAfter() for f in head) + 4

    # ---- body ------------------------------------------------------------------------
    body = []
    fig_no = table_no = 0
    for i, sec in enumerate(draft.sections):
        body.append(Paragraph(f"{ROMAN[i]}. {escape(sec.title.upper())}", S["h1"]))
        sub = 0
        for b in sec.blocks:
            t = b["type"]
            if t == "p":
                body.append(Paragraph(inline(b["text"]), S["body"]))
            elif t == "h":
                body.append(Paragraph(f"{chr(65 + sub)}. {inline(b['text'])}", S["h2"]))
                sub += 1
            elif t == "li":
                body.append(Paragraph(f"\u2022&nbsp; {inline(b['text'])}", S["bullet"]))
            elif t == "fig" and figures.get(b["key"]):
                fig_no += 1
                png = figures[b["key"]]
                w, h = ImageReader(io.BytesIO(png)).getSize()
                width = col_w * 0.98
                img = Image(io.BytesIO(png), width=width, height=width * h / w)
                body.append(KeepTogether([
                    img, Paragraph(f"<b>Fig. {fig_no}.</b> {inline(b['caption'])}", S["caption"])]))
            elif t == "table":
                table_no += 1
                body.append(Paragraph(f"TABLE {ROMAN[table_no - 1]}<br/>{escape(b['caption'])}", S["tcap"]))
                data = [[Paragraph(escape(c), S["cellb"]) for c in b["header"]]]
                data += [[Paragraph(inline(c), S["cell"]) for c in row] for row in b["rows"]]
                widths = [9 * mm, 11.5 * mm, col_w - 20.5 * mm]
                tbl = Table(data, colWidths=widths, repeatRows=1)
                tbl.setStyle(TableStyle([
                    ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.4, colors.black),
                    ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 1.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
                    ("LEFTPADDING", (0, 0), (-1, -1), 1.5), ("RIGHTPADDING", (0, 0), (-1, -1), 1.5),
                ]))
                body.append(tbl)
                body.append(Spacer(1, 6))

    body.append(Paragraph("REFERENCES", S["h1"]))
    for n, p in enumerate(draft.references, start=1):
        body.append(Paragraph(reference_html(p, n), S["ref"]))

    # ---- page templates ----------------------------------------------------------------
    gap = 3 * mm
    top_y = page_h - tm
    first_col_h = top_y - head_h - gap - bm
    frames_first = [
        Frame(lm, top_y - head_h, full_w, head_h, id="head", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
        Frame(lm, bm, col_w, first_col_h, id="c1", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
        Frame(lm + col_w + gutter, bm, col_w, first_col_h, id="c2", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
    ]
    later_h = page_h - tm - bm
    frames_later = [
        Frame(lm, bm, col_w, later_h, id="l1", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
        Frame(lm + col_w + gutter, bm, col_w, later_h, id="l2", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
    ]
    short_title = (draft.title[:88] + "...") if len(draft.title) > 90 else draft.title

    def footer(canvas, doc, first=False):
        canvas.saveState()
        canvas.setFont(FONT, 7)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawCentredString(page_w / 2, bm - 8 * mm * 0.8, str(doc.page))
        if first:
            canvas.setFont(ITALIC, 6.2)
            canvas.drawCentredString(page_w / 2, bm - 8 * mm * 0.8 - 8,
                                     "AI-generated draft. Verify all claims and references against the original sources.")
        else:
            canvas.setFont(ITALIC, 6.5)
            canvas.drawString(lm, page_h - tm + 6 * mm * 0.6, short_title)
            canvas.drawRightString(page_w - rm, page_h - tm + 6 * mm * 0.6, "AI-generated draft")
            canvas.setStrokeColor(colors.HexColor("#999999"))
            canvas.setLineWidth(0.3)
            canvas.line(lm, page_h - tm + 4 * mm * 0.6, page_w - rm, page_h - tm + 4 * mm * 0.6)
        canvas.restoreState()

    buf = io.BytesIO()
    doc = BaseDocTemplate(buf, pagesize=A4, title=draft.title, author=draft.author_line,
                          subject="AI-generated literature review", creator="AI Research Agent")
    doc.addPageTemplates([
        PageTemplate(id="first", frames=frames_first, onPage=lambda c, d: footer(c, d, first=True)),
        PageTemplate(id="later", frames=frames_later, onPage=lambda c, d: footer(c, d, first=False)),
    ])
    doc.build([NextPageTemplate("later")] + head + [FrameBreak()] + body)
    return buf.getvalue()
