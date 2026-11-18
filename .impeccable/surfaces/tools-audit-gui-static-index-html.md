---
version: 1
slug: "tools-audit-gui-static-index-html"
primary_target: "tools/audit_gui_static/index.html"
related_targets: ["tools/audit_gui_static/style.css","tools/audit_gui_static/app.js"]
---

# Surface brief: the audit interface

Scope: `tools/audit_gui_static/` (one page, three views), served locally by
`tools/audit_gui.py`. Visitor mode: Operate.

Audience and job: a researcher who has trained a classifier and has not read
the documentation. They must get from "I have a model" to a finished report
card through this page alone. Content: the contract check, the audit run, the
committed report card, the cross-classifier tables, and the steps for adding a
classifier. Constraints: offline, no text input, no post text, slurs masked by
default, nothing ranks classifiers.

Memorable moment: every figure and table carries a caption that says what it
cannot show.

Unresolved: the framework's name; whether the lexicon-category figure stays in
the first screen.

## Direction contract

THESIS: The audit is the results section the researcher is about to write:
numbered tables and figures whose captions state their limits. It refuses the
experiment dashboard of metric tiles, a dark rail of runs and ranked bars.

OWN-WORLD: A white page on a pale blue-grey desk. A citation-blue rail and
actions. Ink black. One serif for titles, captions and tables, one sans for
controls. Tables ruled heavy at top and bottom and light under the head.
Data in the plotting colours researchers know, plus a neutral. Sources as
footnotes.

STORY: The researcher sees whether their classifier is admitted, runs the
audit, and reads findings that say what the data can and cannot support.

FIRST VIEWPORT: Blue rail of classifiers at left. One white page: the
classifier's name as the title, a byline of model facts, then two columns.
Left, "1 Contract": the Run the audit button and Table 1, the contract items.
Right, "2 Where it stands": Figure 1, test accuracy as a dot plot, with its
caption. A footnote names the source file.

Signature interaction: a number and its source are linked. Pointing at or
focusing a footnote mark highlights the file it was read from, and during a
run each stage is typeset into the numbered list in place as it finishes.
Motion grammar: state changes happen in place in 120 to 200 ms, ease-out;
nothing animates on entry.

FORM: Proceedings Page, the top-ranked grounded direction of the second hand,
chosen by the user over the assigned one. seed:58fb07346e6c6c94

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
