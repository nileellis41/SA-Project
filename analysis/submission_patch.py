"""Cover stale numbers baked into four original slide images with opaque text patches.

    uv run --with python-pptx python -m analysis.submission_patch

Reads submission/Sleeve1_Final_Presentation_prepatch.pptx (the deck as built by submission_deck.py) and writes
submission/Sleeve1_Final_Presentation.pptx. Coordinates are in pixels of the 2000-px-wide slide image
(150 px per inch), measured from the images. At 47 trials the PCA-3 sparse baseline deflated Sharpe is still
0.000 net and 1.00 gross at the printed precision, so only the trial count changes.
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

SUB = Path(__file__).resolve().parents[1] / "submission"
SRC, OUT = SUB / "Sleeve1_Final_Presentation_prepatch.pptx", SUB / "Sleeve1_Final_Presentation.pptx"
PX = 1 / 150  # inches per image pixel
INK = "2A3238"
CARD, APPX = "FDFDFD", "EDF1F0"
FOOTNOTE = "Updated: 47 trials total; defect = 34 risk units across 13 distinct trades."

# deck position (1-based) -> patches: (x0, y_mid, x1, half_height, background, runs, size)
PATCHES = {
    26: [(1537, 410, 1571, 14, CARD, [("47", False)], 14),  # original slide 14: "best of 31 documented trials"
         (172, 638, 650, 17, CARD, [("Trades in those folds: ", False), ("−34", True), (" risk units", False)], 13.5),
         (172, 677, 650, 17, CARD, [("(13 distinct trades).", False)], 13.5)],  # replaces "Whole model: +9." (other basis)
    29: [(1341, 662, 1377, 14, CARD, [("47", False)], 14)],  # original 17: "31 documented trials"
    33: [(1631, 871, 1664, 13, APPX, [("47", False)], 13)],  # original 21: "0.000 net, N = 31"
    34: [(1726, 485, 1759, 13, APPX, [("47", False)], 13)],  # original 22: "With 31 trials"
}


def patch(slide, x0, ymid, x1, half, bg, runs, size, nudge=1.5):
    box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x0 * PX), Inches((ymid - half) * PX),
                                 Inches((x1 - x0) * PX), Inches(2 * half * PX))
    box.fill.solid()
    box.fill.fore_color.rgb = RGBColor.from_string(bg)
    box.line.fill.background()
    box.shadow.inherit = False
    box.name = "stale-number patch"
    tf = box.text_frame
    tf.margin_left = tf.margin_right = tf.margin_bottom = 0
    tf.margin_top = Inches(-nudge * PX)  # digits sit slightly above the line-box centre
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.word_wrap = False
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT if len(runs) > 1 or len(runs[0][0]) > 3 else PP_ALIGN.CENTER
    for t, bold in runs:
        r = p.add_run()
        r.text = t
        r.font.name, r.font.size, r.font.bold = "Segoe UI", Pt(size), bold
        r.font.color.rgb = RGBColor.from_string(INK)


def footnote(slide):
    tb = slide.shapes.add_textbox(Inches(134 * PX), Inches(985 * PX), Inches(1700 * PX), Inches(30 * PX))
    r = tb.text_frame.paragraphs[0].add_run()
    r.text = FOOTNOTE
    r.font.name, r.font.size, r.font.italic = "Segoe UI", Pt(12), True
    r.font.color.rgb = RGBColor.from_string("0C6A5A")


def main(fallback: tuple[int, ...] = ()) -> None:
    prs = Presentation(str(SRC))
    for pos, items in PATCHES.items():
        slide = prs.slides[pos - 1]
        if pos in fallback:
            footnote(slide)
            continue
        for item in items:
            patch(slide, *item)
    prs.save(str(OUT))
    print(OUT)


if __name__ == "__main__":
    import sys
    main(tuple(int(a) for a in sys.argv[1:]))
