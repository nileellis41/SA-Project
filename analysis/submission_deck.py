"""Insert the final-presentation slides into a copy of the existing deck (existing slides untouched).

    uv run --with python-pptx python -m analysis.submission_deck

Reads "Sleeve 1 PCA Curve Residuals.pptx", writes submission/Sleeve1_Final_Presentation.pptx.
Numbers are copied from submission/deck_numbers.md; figures from submission/figures/.
The existing slides are full-slide images, so their baked-in text is not edited; only two speaker notes are.
"""

from __future__ import annotations

from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Sleeve 1 PCA Curve Residuals.pptx"
OUT = ROOT / "submission" / "Sleeve1_Final_Presentation.pptx"
FIG = ROOT / "submission" / "figures"
INSERT_AFTER = 3  # the existing "Data and test design" slide

# Sampled from the existing slide images
BG, INK, MUTED, TEAL = "F5F7F6", "192026", "56616B", "0C6A5A"
HDR, HL, BORDER, CARD, CODE = "E4E6E3", "E3EFEB", "C2C4C3", "FDFDFD", "E9EDEC"
TITLE, BODY, MONO = "Georgia", "Segoe UI", "Consolas"
L, R = 0.89, 12.44
W = R - L


def rgb(h: str) -> RGBColor:
    return RGBColor.from_string(h)


def text(slide, x, y, w, h, paras, size=16, color=INK, font=BODY, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, spacing=4, italic=False):
    """paras: list of str or list of [(text, {bold, color, font, size, italic})] runs."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, 0)
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(spacing)
        runs = para if isinstance(para, list) else [(para, {})]
        for t, o in runs:
            r = p.add_run()
            r.text = t
            f = r.font
            f.name, f.size = o.get("font", font), Pt(o.get("size", size))
            f.bold, f.italic = o.get("bold", bold), o.get("italic", italic)
            f.color.rgb = rgb(o.get("color", color))
    return tb


def bullets(slide, x, y, w, h, items, size=16):
    tb = text(slide, x, y, w, h, items, size=size, spacing=9)
    for p in tb.text_frame.paragraphs:
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(Inches(0.25)))
        pPr.set("indent", str(-Inches(0.25)))
        bu = etree.SubElement(pPr, qn("a:buFont"))
        bu.set("typeface", "Arial")
        ch = etree.SubElement(pPr, qn("a:buChar"))
        ch.set("char", "•")
        clr = etree.Element(qn("a:buClr"))
        etree.SubElement(clr, qn("a:srgbClr")).set("val", TEAL)
        pPr.insert(0, clr)
    return tb


def box(slide, x, y, w, h, fill=CARD, line=None, radius=0.06, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(fill)
    if line:
        s.line.color.rgb = rgb(line)
        s.line.width = Pt(0.75)
    else:
        s.line.fill.background()
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    s.shadow.inherit = False
    return s


def card(slide, x, y, w, h, head, mono=None, body=None, size=14):
    box(slide, x, y, w, h, line="DCDFDD")
    paras = [[(head, {"bold": True, "color": TEAL, "size": size + 1})]]
    if mono:
        paras.append([(mono, {"font": MONO, "size": size + 1})])
    if body:
        paras.append([(body, {"size": size, "color": INK})])
    text(slide, x + 0.25, y + 0.2, w - 0.5, h - 0.35, paras, spacing=7)


def stat(slide, x, y, w, h, value, label, value_size=34):
    box(slide, x, y, w, h, line="DCDFDD")
    text(slide, x + 0.25, y + 0.18, w - 0.5, 0.75, [[(value, {"font": TITLE, "bold": True, "size": value_size, "color": TEAL})]])
    text(slide, x + 0.25, y + 0.95, w - 0.5, h - 1.05, [label], size=14, color=MUTED)


def code(slide, x, y, w, h, lines, size=15):
    box(slide, x, y, w, h, fill=CODE, radius=0.08)
    text(slide, x + 0.3, y + 0.15, w - 0.6, h - 0.25, lines, size=size, font=MONO, anchor=MSO_ANCHOR.MIDDLE, spacing=6)


def _border(cell, color=BORDER, width=9525):
    tcPr = cell._tc.get_or_add_tcPr()
    for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
        ln = etree.SubElement(tcPr, qn(tag), w=str(width))
        fill = etree.SubElement(ln, qn("a:solidFill"))
        etree.SubElement(fill, qn("a:srgbClr")).set("val", color)


def table(slide, x, y, w, cols, rows, size=14, row_h=0.36, hl=(), header=True, align=None, bold_col0=False):
    """cols: relative widths. rows: list of lists of str (row 0 is the header if header=True)."""
    shape = slide.shapes.add_table(len(rows), len(cols), Inches(x), Inches(y), Inches(w), Inches(row_h * len(rows)))
    tbl = shape.table
    tbl.first_row = False
    tbl.horz_banding = False
    total = sum(cols)
    for j, c in enumerate(cols):
        tbl.columns[j].width = Inches(w * c / total)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            _border(cell)
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(HDR if header and i == 0 else HL if i in hl else BG)
            cell.margin_left = cell.margin_right = Inches(0.1)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = (align[j] if align else (PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT))
            r = p.add_run()
            r.text = val
            r.font.name, r.font.size, r.font.color.rgb = BODY, Pt(size), rgb(INK)
            r.font.bold = (header and i == 0) or (bold_col0 and j == 0)
    return shape


def picture(slide, name, x, y, w=None, h=None):
    kw = {}
    if w:
        kw["width"] = Inches(w)
    if h:
        kw["height"] = Inches(h)
    return slide.shapes.add_picture(str(FIG / name), Inches(x), Inches(y), **kw)


def new_slide(prs, eyebrow, title, page, footer, notes):
    s = prs.slides.add_slide(prs.slide_layouts[0])
    for ph in list(s.placeholders):
        ph._element.getparent().remove(ph._element)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = rgb(BG)
    eb = text(s, L, 0.84, W, 0.35, [eyebrow.upper()], size=14, color=TEAL, bold=True)
    for r in eb.text_frame.paragraphs[0].runs:
        r.font._rPr.set("spc", "300")
    text(s, L, 1.12, W, 0.8, [title], size=38, font=TITLE, bold=True)
    text(s, L, 6.85, W - 1.0, 0.4, [footer], size=12, color=MUTED)
    text(s, R - 0.9, 6.85, 0.9, 0.4, [page], size=12, color=MUTED, align=PP_ALIGN.RIGHT)
    s.notes_slide.notes_text_frame.text = notes
    return s


def build(prs) -> int:
    n0 = len(prs.slides)
    C0 = 2.1  # content top

    # 1. Research question
    s = new_slide(prs, "Research question", "Does the residual revert enough to trade?", "3.1",
                  "Out of sample, walk-forward 1995–2026, 0.25 bp a leg. Sources: deck_numbers.md §E and §A (reversion vs round-trip cost).",
                  "Line to say: The question is simple. Once we hedge level and slope, does what is left mean-revert enough to "
                  "pay for trading it? The answer is that it does revert: the signal points the right way. But the move we "
                  "capture per trade, about 0.45 bp on average for PCA-2, is smaller than the 0.73 bp round trip. That gap is the whole story.")
    box(s, L, C0, W, 1.25, line="DCDFDD")
    text(s, L + 0.35, C0 + 0.15, W - 0.7, 0.95,
         ["“After hedging level and slope with PCA, does the Treasury curve residual mean-revert enough to trade "
          "profitably after realistic costs?”"], size=22, font=TITLE, italic=True, anchor=MSO_ANCHOR.MIDDLE)
    box(s, L, C0 + 1.45, W, 0.8, fill=TEAL, radius=0.1)
    text(s, L + 0.35, C0 + 1.45, W - 0.7, 0.8,
         [[("Answer: ", {"bold": True, "color": "FFFFFF"}),
           ("it reverts, but the reversion per trade is smaller than the round-trip cost.", {"color": "FFFFFF"})]],
         size=20, anchor=MSO_ANCHOR.MIDDLE)
    cw = (W - 0.5) / 3
    for k, (v, lab) in enumerate((("0.17–0.33", "Rank IC of −z with the next 10 bars on entry days (headline to PCA-3 sparse). The signal points the right way."),
                                  ("0.45 < 0.73", "bp. PCA-2: mean realised move per trade, against the median round-trip cost."),
                                  ("−0.60", "Headline net Sharpe at 0.25 bp a leg; 0.55 before cost."))):
        stat(s, L + k * (cw + 0.25), C0 + 2.5, cw, 2.0, v, lab)

    # 2. PCA in and out
    s = new_slide(prs, "PCA · in and out", "Three factors explain 99.6% of moves", "3.2",
                  "Fold with test window from 2022-12-12; eigenvector signs are arbitrary (level is negative on every tenor). Source: deck_numbers.md §B.",
                  "Line to say: The input is daily yield changes in basis points, six tenors, five years of training data, "
                  "about 1,260 days. The covariance matrix gives us eigenvalues and eigenvectors. In this fold the first factor, "
                  "level, explains 89% of variance, slope another 9%, and curvature 1.4%. Together that is 99.6%. "
                  "Everything we trade lives in the last half percent.")
    code(s, L, C0, 5.6, 0.85, ["Δy: daily changes, bp, 6 tenors, 1,259 days", "Σ = cov(Δy) = VΛVᵀ; L = top-k eigenvectors"], size=14)
    rows = [["Tenor", "Level", "Slope", "Curvature"],
            ["2y", "−0.337", "−0.516", "−0.552"], ["3y", "−0.391", "−0.425", "−0.131"], ["5y", "−0.438", "−0.165", "+0.384"],
            ["7y", "−0.451", "+0.073", "+0.426"], ["10y", "−0.442", "+0.295", "+0.200"], ["30y", "−0.379", "+0.659", "−0.557"],
            ["Eigenvalue, bp²", "138.0", "13.88", "2.14"], ["Variance share", "89.2%", "9.0%", "1.4%"]]
    table(s, L, C0 + 1.05, 5.6, [1.6, 1, 1, 1.1], rows, size=13, row_h=0.33, hl=(7, 8))
    picture(s, "03_pca_loadings.png", 6.75, C0 - 0.05, w=5.69)
    bullets(s, 6.85, C0 + 3.3, 5.6, 1.3,
            ["Level moves every tenor the same way; slope is front vs back; curvature is belly vs wings.",
             "Refit every fold on that fold's own 5-year training window."], size=14)

    # 3. W
    s = new_slide(prs, "The hedge matrix", "W = I − LLᵀ keeps what the factors miss", "3.3",
                  "Fold trained 2018-01 to 2022-11. Package weights in bp of yield per unit of the 5y residual. Source: deck_numbers.md §B.",
                  "Line to say: W is the hedge. It removes level and slope from every tenor, so W times L is zero. Its trace "
                  "equals tenors minus factors, six minus two, four. The diagonal says how much of each tenor's own direction "
                  "survives the hedge: most of the belly, less than half of the 30-year. One row of W is one tradeable package. "
                  "The 5-year row is a butterfly: long 0.78 of the 5-year, short its neighbours and the wings.")
    picture(s, "03_pca_W_diagonal.png", L - 0.05, C0, w=6.4)
    code(s, 7.55, C0, R - 7.55, 0.95, ["W = I − L Lᵀ,   Lᵀ W = 0", "trace(W) = 6 − 2 = 4"], size=15)
    text(s, 7.55, C0 + 1.15, R - 7.55, 0.3, [[("5y package, one row of W (bp of yield)", {"bold": True, "color": TEAL})]], size=14)
    table(s, 7.55, C0 + 1.5, R - 7.55, [1] * 6,
          [["2y", "3y", "5y", "7y", "10y", "30y"], ["−0.23", "−0.24", "+0.78", "−0.19", "−0.14", "−0.06"]],
          size=14, row_h=0.4, align=[PP_ALIGN.CENTER] * 6)
    bullets(s, 7.55, C0 + 2.55, R - 7.55, 2.0,
            ["Wᵢᵢ = ‖W eᵢ‖²: the share of tenor i's own direction left after the hedge.",
             "Butterfly: long the 5y, short both sides, so no level or slope exposure.",
             "Hedging curvature too (PCA-3) leaves only 0.32 of the 2y and 0.11 of the 30y."], size=14)

    # 4. Signal
    s = new_slide(prs, "The signal · OU s-score", "Fade the residual when it is stretched", "3.4",
                  "PCA-2, live test bars passing the screen. Code: sleeves/pca.py:50-69, models/ou.py:44-90. Source: deck_numbers.md §A, §C, §K.",
                  "Line to say: The signal is the Avellaneda-Lee s-score. On the last 60 days of the cumulative residual we fit "
                  "an AR(1), refit every day. That gives an equilibrium m, a stationary sigma, and z, how many sigmas we are away. "
                  "We only trade if the fit actually mean-reverts with a half-life under 30 days, which passes about 90% of days. "
                  "Positive z means the tenor's yield is high against the factors, so we buy the tenor and sell the hedge. "
                  "After SVB the 5-year was 2.87 sigmas out, about 8 basis points from its equilibrium.")
    code(s, L, C0, W, 1.3, ["C_t = a + b·C_{t−1} + e        last 60 bars, refit every bar",
                            "m = a/(1 − b)     σ_eq = sd(e)/√(1 − b²)     z = (C_t − m)/σ_eq",
                            "trade only if 0 < b < 1 and half-life = ln 2/κ = −ln 2/ln b < 30 bars"], size=15)
    rows = [["PCA-2 tenor", "Half-life, bars", "σ_eq, bp", "Screen pass"],
            ["2y", "6.36", "1.60", "88.0%"], ["3y", "5.50", "0.91", "94.6%"], ["5y", "5.98", "1.47", "89.6%"],
            ["7y", "7.08", "1.11", "87.1%"], ["10y", "6.28", "1.10", "91.6%"], ["30y", "6.99", "1.53", "88.7%"]]
    table(s, L, C0 + 1.55, 5.9, [1.3, 1.3, 1, 1.1], rows, size=13, row_h=0.36, hl=(3,))
    bullets(s, 7.1, C0 + 1.55, R - 7.1, 3.2,
            ["Entry ∈ {1.0 … 2.5}, exit ∈ {0 … 0.75}, set per fold by training net Sharpe; stop at |z| > 4.",
             "z > 0: tenor yield high vs the factors → buy the tenor, sell the hedge legs.",
             "5y on 2023-03-13: z = +2.87, σ_eq = 2.80 bp → 8.03 bp from equilibrium.",
             "Half-life is used only in the screen, never for sizing or timing."], size=15)

    # 5. SVB trade
    s = new_slide(prs, "One trade, end to end", "The 5y after SVB, entry to exit", "3.5",
                  "$/bp = P&L on $1m for +1 bp in that yield. Member pca2_ust_5y, fold from 2022-12-12, bands 2.5 / 0.5. Source: deck_numbers.md §K.",
                  "Line to say: Here is one real trade. The Monday after SVB failed, the 5-year residual jumped to 2.87 sigmas, "
                  "8 basis points cheap. We bought the 5-year and sold the other five tenors in the W weights: long about $2,000 "
                  "of 5-year DV01 per basis point, but net DV01 only about $200, so this is a curve trade, not a duration bet. "
                  "Twenty trading days later z fell to 0.49, inside the 0.5 exit band. The residual moved 6.2 basis points our way: "
                  "plus $15,300 gross, $2,400 cost, plus $12,900 net on a million.")
    rows = [["Entry (signal bar)", "2023-03-13"], ["z at entry", "+2.87"], ["σ_eq at entry", "2.80 bp"],
            ["Gap from equilibrium, z × σ_eq", "8.03 bp"], ["Size (vol 1.90 bp/day)", "−0.526 risk units"],
            ["Exit", "2023-04-10, 20 bars, z = 0.49"], ["Residual move", "−6.20 bp (in our favour)"],
            ["Gross / cost / net, risk units", "+3.26 / 0.52 / +2.74"], ["Gross / cost / net, $1m", "+$15.3k / $2.4k / +$12.9k"]]
    table(s, L, C0, 6.1, [1.45, 1.3], rows, size=14, row_h=0.4, header=False, hl=(8,),
          align=[PP_ALIGN.LEFT, PP_ALIGN.RIGHT])
    rows = [["Leg", "Weight", "$ per bp"], ["2y", "−0.233", "+600"], ["3y", "−0.241", "+622"], ["5y", "+0.781", "−2,012"],
            ["7y", "−0.185", "+477"], ["10y", "−0.145", "+372"], ["30y", "−0.057", "+147"], ["Net / gross", "", "+206 / 4,230"]]
    table(s, 7.35, C0, R - 7.35, [1, 1, 1.3], rows, size=14, row_h=0.4, hl=(3,))
    bullets(s, 7.35, C0 + 3.45, R - 7.35, 1.2,
            ["Long 5y DV01 against short wings: the trade is the residual, not duration."], size=14)

    # 6. Hit ratio
    s = new_slide(prs, "Hit ratio · conditional gain and loss", "Wins are frequent; losses are twice as big", "3.6",
                  "Units: risk units = one trailing daily σ at entry (not σ_eq, not bp). PCA-3 sparse: distinct trades. Source: deck_numbers.md §D.",
                  "Line to say: Every model wins more trades than it loses, even after cost. The problem is size: the average "
                  "loser costs about twice the average winner. With win over loss near 0.53, you need a 65% hit rate to break "
                  "even, and we get 44 to 58. The band is wide because cost is a fixed 0.73 basis points a round trip, so the "
                  "optimiser wants each trade to capture as much reversion as possible; it sits at the top of the grid in half "
                  "or more of the folds. Raising the hit rate will not fix this. Only something that cuts the losers, a stop or "
                  "a regime gate, shrinks L.")
    rows = [["Model", "Hit gross", "Hit net", "Gross", "Cost", "Net", "W", "L", "W ÷ L"],
            ["Headline", "68.1%", "54.5%", "+0.71", "1.33", "−0.62", "1.94", "3.68", "0.53"],
            ["PCA-3 sparse", "83.1%", "57.6%", "+1.74", "2.08", "−0.33", "1.52", "2.86", "0.53"],
            ["PCA-2 full", "67.2%", "54.4%", "+0.69", "1.39", "−0.70", "2.04", "3.97", "0.51"],
            ["Pairwise NN", "69.0%", "54.0%", "+0.48", "1.10", "−0.62", "1.89", "3.56", "0.53"],
            ["Fly 2s5s10s", "56.7%", "51.7%", "−0.19", "1.07", "−1.26", "1.90", "4.63", "0.41"],
            ["Fly 3s5s7s", "74.4%", "44.2%", "+1.12", "2.27", "−1.15", "1.74", "3.43", "0.51"]]
    table(s, L, C0, W, [1.9, 1, 1, 0.9, 0.9, 0.9, 0.8, 0.8, 0.9], rows, size=14, row_h=0.36, hl=(1,))
    cw = (W - 0.5) / 3
    card(s, L, C0 + 2.75, cw, 1.85, "Breakeven hit rate", "1/(1 + W÷L) = 1/1.53 = 65%",
         "Net hit rates are 44–58%: below it for every model.", size=13)
    card(s, L + cw + 0.25, C0 + 2.75, cw, 1.85, "Expectancy", "p·W − (1 − p)·L",
         "L ≈ 2×W everywhere, so only a stop or a regime gate shrinks it.", size=13)
    card(s, L + 2 * (cw + 0.25), C0 + 2.75, cw, 1.85, "Why the band is wide", "round trip 0.73 bp, fixed",
         "Captured ≈ (entry − exit)·σ_eq, so the band widens: entry at the grid top (2.5) in 45–70% of PCA-2 folds.", size=13)

    # 7. Quiet vs volatile
    s = new_slide(prs, "Quiet vs volatile entries", "Quiet markets can't pay a fixed cost", "3.7",
                  "Basis: member trades; terciles pooled per model over the full sample (ex post, descriptive). Net in risk units. Source: deck_numbers.md §F.",
                  "Line to say: We split trades by how volatile the residual was when we entered. Cost is fixed in basis points, "
                  "but we size to one over volatility, so in a quiet market we buy more DV01 for the same expected edge, and the "
                  "round trip costs up to twice a day's move. Low-volatility trades lose for every model. Only PCA-3 sparse's "
                  "high-volatility third makes money.")
    rows = [["Model", "Tercile cuts, bp/day", "Round trip ÷ vol, low", "Net, low", "Net, mid", "Net, high"],
            ["Headline", "0.50 / 1.03", "1.78", "−1.25", "−0.27", "−0.33"],
            ["PCA-3 sparse", "0.92 / 1.54", "2.32", "−0.48", "+0.04", "+0.48"],
            ["PCA-2 full", "0.47 / 0.77", "1.96", "−1.47", "−0.20", "−0.45"],
            ["Pairwise NN", "1.23 / 3.01", "1.30", "−1.77", "−0.01", "−0.09"]]
    table(s, L, C0, W, [1.6, 1.5, 1.6, 1, 1, 1], rows, size=15, row_h=0.42, hl=(2,))
    bullets(s, L, C0 + 2.45, W, 2.2,
            ["Entry vol = the trailing EW-60 residual vol on the entry bar (bp/day), the same vol used for sizing.",
             "Position = 1/vol, so a quiet market buys more DV01 per unit of edge: the round trip costs 1.3–2.3× a day's move in the low tercile.",
             "Low-vol trades lose for every model; only PCA-3 sparse's high-vol third earns (+0.48σ per trade)."], size=16)

    # 8. Per dollar
    s = new_slide(prs, "Per dollar", "$1 million in: earned vs paid", "3.8",
                  "Basis: $1m of capital at the 10% vol target (not notional); additive returns. DV01 is the mean over live days. Source: deck_numbers.md §G.",
                  "Line to say: Put a million dollars of capital behind PCA-3 sparse and it makes about $100,000 a year before "
                  "cost, then pays $108,000 to trade. Every dollar it earns costs a dollar and seven cents. The headline pays two "
                  "dollars ten for every dollar it earns. The book is large in gross DV01 but nearly flat in net DV01: it is long "
                  "and short the curve at the same time, which is why the cost adds up.")
    stat(s, L, C0, 3.6, 1.95, "$1.07", "cost per $1 of gross P&L, PCA-3 sparse")
    stat(s, L, C0 + 2.2, 3.6, 1.95, "$2.10", "cost per $1 of gross P&L, headline")
    rows = [["Model", "Gross $/yr", "Cost $/yr", "Net $/yr", "Take-home per $1", "Gross DV01 $/bp", "Net DV01 $/bp"],
            ["Headline", "45,500", "95,800", "−50,300", "−$1.10", "6,632", "446"],
            ["PCA-3 sparse", "100,500", "107,600", "−7,100", "−$0.07", "9,374", "175"],
            ["PCA-2 full", "38,500", "96,700", "−58,200", "−$1.51", "7,596", "498"],
            ["Pairwise NN", "35,600", "55,500", "−20,000", "−$0.56", "3,856", "263"]]
    table(s, 4.75, C0, R - 4.75, [1.45, 1, 1, 1, 1.05, 1.05, 0.95], rows, size=13, row_h=0.5, hl=(2,))
    bullets(s, 4.75, C0 + 3.05, R - 4.75, 1.4,
            ["Take-home per $1 of gross = 1 − cost ÷ gross.",
             "Gross DV01 $3.9k–9.4k per bp, net only $175–498: long and short the curve at once."], size=15)

    # 9. Best vs worst
    s = new_slide(prs, "Model comparison", "Best and worst models side by side", "3.9",
                  "Trades/yr = trades ÷ (live bars ÷ 252), identical trades merged; headline 13.1 = 426 trades. Source: deck_numbers.md §E.",
                  "Line to say: Same pipeline, same costs. The best model, PCA-3 sparse, has a gross Sharpe of 1.2 and breaks "
                  "even at 93% of the cost table, so it is close. The worst are the fixed butterflies traded with the same rules, "
                  "which tells you the PCA hedge is adding signal, not just the trading rule. Every model is positive before cost "
                  "and negative after it.")
    rows = [["Model", "Gross SR", "Net SR", "Breakeven ×", "IC, 10d", "Trades/yr", "Cost, % pa"],
            ["PCA-3 sparse (best)", "1.21", "−0.09", "0.93", "0.33", "7.9", "10.8"],
            ["Headline (PCA-2 + credit)", "0.55", "−0.60", "0.48", "0.17", "13.1", "9.6"],
            ["Pairwise NN", "0.46", "−0.26", "0.64", "0.13", "7.7", "5.6"],
            ["PCA-2 full", "0.45", "−0.67", "0.40", "0.20", "10.6", "9.7"],
            ["Fly 3s5s7s", "0.38", "−0.77", "0.32", "—", "43 total", "10.1"],
            ["Fly 2s5s10s (worst)", "0.11", "−0.42", "0.22", "—", "60 total", "3.9"]]
    table(s, L, C0, W, [2.3, 1, 1, 1.1, 1, 1.1, 1.1], rows, size=15, row_h=0.42, hl=(1,))
    bullets(s, L, C0 + 3.2, W, 1.5,
            ["Every model is positive gross and negative net.",
             "The fixed flies, traded with the same rules, earn far less gross (0.11–0.38): the PCA hedge adds signal."], size=16)

    # 10. Cost round 1
    s = new_slide(prs, "Cost round 1", "Freezing size helps; nothing clears zero", "3.10",
                  "Bands and the gate level X are fitted at 1x; 0.5x and 2x re-price the same positions. Source: deck_numbers.md §H, §J.",
                  "Line to say: We tried two cost cuts. Freezing the position size at entry has no parameter: it cuts cost by 11 "
                  "to 14% and adds 0.16 to 0.25 of net Sharpe. Adding a volatility gate, fitted per fold, halves cost again but "
                  "also halves PCA-3 sparse's gross Sharpe. The headline stays negative. PCA-3 sparse reaches plus 0.09, but with "
                  "47 trials its deflated Sharpe is 0.002. Honestly: not distinguishable from zero.")
    rows = [["Net Sharpe at cost ×", "0.5×", "1×", "2×", "Cost % pa"],
            ["Headline · baseline", "−0.03", "−0.60", "−1.64", "9.6"],
            ["Headline · freeze size", "+0.10", "−0.35", "−1.20", "8.2"],
            ["Headline · freeze + gate", "+0.06", "−0.22", "−0.76", "4.5"],
            ["PCA-3 sparse · baseline", "+0.58", "−0.09", "−1.28", "10.8"],
            ["PCA-3 sparse · freeze size", "+0.63", "+0.07", "−0.95", "9.5"],
            ["PCA-3 sparse · freeze + gate", "+0.33", "+0.09", "−0.38", "3.9"]]
    table(s, L, C0, 6.9, [2.5, 1.1, 0.8, 0.8, 1], rows, size=14, row_h=0.42, hl=(5, 6))
    bullets(s, 8.1, C0, R - 8.1, 4.4,
            ["Freeze size (no parameter): cost −11 to −14%, net SR +0.16 to +0.25.",
             "Vol gate (X fitted per fold, a trial): halves cost again, and halves PCA-3 sparse gross (1.17 → 0.56).",
             "Headline still negative at 1x (−0.22).",
             "PCA-3 sparse +0.09 at 1x; deflated Sharpe 0.002 at 47 trials."], size=15)

    # 11. Evolution
    s = new_slide(prs, "Project evolution", "Seven steps, 47 trials, one answer", "3.11",
                  "No git history: steps from dated table snapshots. Pairs are gross / net Sharpe. Source: deck_numbers.md §J.",
                  "Line to say: This is how we got here. We started with pairwise spreads, widened the universe, revised the "
                  "design four times, moved to PCA residuals with the headline named in advance, then tried filters, curve rules, "
                  "experiments and one cost round. That is 47 trials in all. Correcting for them, the probability that the best "
                  "variant's true Sharpe is above zero is 42% before cost and 0.2% after it.")
    steps = [("Pairwise, 10 spreads", "NN gate 0.51 / −0.01"), ("Universe to 27–31", "NN gate 0.46 / −0.26"),
             ("4 design revisions", "no Sharpe saved"), ("PCA models", "PCA-3 sparse 1.21 / −0.09"),
             ("Filters + curve rules", "16 variants"), ("Step-8 experiments", "best 1.20 / −0.00"),
             ("Cost round 1", "best 0.56 / +0.09")]
    n = len(steps)
    gap = W / n
    line = s.shapes.add_connector(1, Inches(L + gap / 2), Inches(C0 + 0.45), Inches(R - gap / 2), Inches(C0 + 0.45))
    line.line.color.rgb = rgb(BORDER)
    line.line.width = Pt(2)
    for k, (head, sr) in enumerate(steps):
        cx = L + gap * (k + 0.5)
        dot = box(s, cx - 0.22, C0 + 0.23, 0.44, 0.44, fill=TEAL if k in (3, 6) else CARD, line=TEAL, shape=MSO_SHAPE.OVAL)
        text(s, cx - 0.22, C0 + 0.23, 0.44, 0.44, [str(k + 1)], size=13, bold=True,
             color="FFFFFF" if k in (3, 6) else TEAL, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        dot.name = f"step{k + 1}"
        text(s, cx - gap / 2 + 0.05, C0 + 0.85, gap - 0.1, 0.6, [head], size=13, bold=True, align=PP_ALIGN.CENTER)
        text(s, cx - gap / 2 + 0.05, C0 + 1.45, gap - 0.1, 0.6, [sr], size=12, color=MUTED, align=PP_ALIGN.CENTER)
    cw = (W - 0.5) / 3
    for k, (v, lab) in enumerate((("47", "documented trials: 11 models, 8 cost filters, 8 curve rules, 4 revisions, 12 experiments, 4 cost-round configs"),
                                  ("0.002", "deflated Sharpe, net, of the best variant (PCA-3 sparse freeze + vol gate)"),
                                  ("0.42", "deflated Sharpe, gross, same variant: a 42% probability the true gross Sharpe is above zero"))):
        stat(s, L + k * (cw + 0.25), C0 + 2.35, cw, 2.15, v, lab, value_size=36)

    # 12. Warnings
    s = new_slide(prs, "Warnings", "Read the best result with care", "3.12",
                  "The defect is deliberately not fixed: a fix chosen after seeing results is another trial. Source: deck_numbers.md Defect, §I, §J, §A, §C.",
                  "Line to say: Four warnings before you believe the best number. One, a construction defect: in 2010 to 2013 the "
                  "3-year was hedged with nearly collinear tenors, weights of plus 68, minus 121, plus 69, and trades in those four "
                  "folds lost 34 risk units across 13 distinct trades. Two, overlap: six members are 2.3 independent bets, though "
                  "counting each portfolio once barely moves the Sharpe. Three, selection: 47 trials. Four, the 60-day OU fit is "
                  "biased fast, so the signal promises more than it delivers.")
    cw, ch = (W - 0.3) / 2, 2.15
    cards = [("1 · Hedge defect, 2010–13", "+68.2 / −120.9 / +68.9",
              "3y hedged by a near-singular 2y/5y/10y solve. Trades in the four folds lost 34 risk units across 13 distinct trades (20 from the 3y)."),
             ("2 · Overlap", "6 members = 2.3 effective bets",
              "The 2y, 3y and 5y members are often one portfolio. Counting each distinct portfolio once: net SR −0.097 vs −0.087."),
             ("3 · Selection", "47 trials → net DSR 0.002",
              "Several experiment settings were chosen after full-sample results were seen."),
             ("4 · Optimistic signal", "half-life 6.4 vs 103 bars",
              "The 60-bar AR(1) is biased fast (small sample); the entry band sits at the grid top in 45–70% of PCA-2 folds.")]
    for k, (h, m, b) in enumerate(cards):
        card(s, L + (k % 2) * (cw + 0.3), C0 + (k // 2) * (ch + 0.25), cw, ch, h, m, b, size=14)

    # References
    s = new_slide(prs, "References", "References", "23", "Data are Bloomberg terminal exports (licensed, not distributed). Full list: report.md.",
                  "Line to say: These are the works the code and the reasoning actually rely on: Avellaneda and Lee for the s-score, "
                  "Litterman and Scheinkman for the factors, Bailey and López de Prado for the deflated Sharpe, Kendall for the AR(1) "
                  "bias, Leung and Li for the optimal-stopping variant, and the cointegration and Kalman papers behind the pairwise model.")
    refs = ["Avellaneda, M., and Lee, J.-H. (2010). Statistical arbitrage in the U.S. equities market. Quantitative Finance, 10(7), 761–782.",
            "Bailey, D. H., and López de Prado, M. (2014). The deflated Sharpe ratio: Correcting for selection bias, backtest overfitting, and non-normality. Journal of Portfolio Management, 40(5), 94–107.",
            "Bloomberg L.P. (2026). Generic US Treasury yields (USGG2YR–USGG30YR), LF98OAS and LUACOAS OAS indices, CDX HY 5Y, HYG. Bloomberg Terminal.",
            "Board of Governors of the Federal Reserve System (2026). DFEDTAR, DFEDTARU. FRED, Federal Reserve Bank of St. Louis.",
            "Engle, R. F., and Granger, C. W. J. (1987). Co-integration and error correction: Representation, estimation, and testing. Econometrica, 55(2), 251–276.",
            "Johansen, S. (1991). Estimation and hypothesis testing of cointegration vectors in Gaussian vector autoregressive models. Econometrica, 59(6), 1551–1580.",
            "Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. Journal of Basic Engineering, 82(1), 35–45.",
            "Kendall, M. G. (1954). Note on bias in the estimation of autocorrelation. Biometrika, 41(3/4), 403–404.",
            "Leung, T., and Li, X. (2015). Optimal mean reversion trading with transaction costs and stop-loss exit. International Journal of Theoretical and Applied Finance, 18(3), 1550020.",
            "Litterman, R., and Scheinkman, J. (1991). Common factors affecting bond returns. Journal of Fixed Income, 1(1), 54–61."]
    text(s, L, C0 - 0.1, W, 4.7, refs, size=13, spacing=7)
    return n0


def fix_notes(prs) -> None:
    """Trial count and defect wording in the two existing notes that mention them (slide images are not edited)."""
    edits = {14: [("Trades in those four folds lost 67 risk units; the model as a whole made 9.",
                   "Trades in those four folds lost 34 risk units across 13 distinct trades, 20 of it from the 3y; counting overlapping members separately it is 67."),
                  ("we tried 31 things.", "we have tried 47 things in all."),
                  ("Gross passes with 1.00; net is 0.000.",
                   "For the best variant at 47 trials, gross is 0.42 and net is 0.002.")],
             17: [("And we tried 31 things, which the deflated Sharpe already accounts for.",
                   "And we tried 47 things, which the deflated Sharpe already accounts for.")]}
    for idx, pairs in edits.items():
        tf = prs.slides[idx - 1].notes_slide.notes_text_frame
        for old, new in pairs:
            hit = False
            for p in tf.paragraphs:
                for r in p.runs:
                    if old in r.text:
                        r.text = r.text.replace(old, new)
                        hit = True
            assert hit, (idx, old)


def main() -> None:
    prs = Presentation(str(SRC))
    n0 = build(prs)
    fix_notes(prs)
    ids = prs.slides._sldIdLst
    entries = list(ids)
    new = entries[n0:n0 + 12]  # the 12 content slides; References stays last
    for e in new:
        ids.remove(e)
    for k, e in enumerate(new):
        ids.insert(INSERT_AFTER + k, e)
    prs.save(str(OUT))
    print(OUT, len(prs.slides), "slides")


if __name__ == "__main__":
    main()
