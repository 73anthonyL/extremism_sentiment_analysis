---
name: Audit workbench
description: A proceedings page for auditing a text classifier of extremist content against one contract.
colors:
  desk: "#dfe5ee"
  page: "#ffffff"
  ink: "#111111"
  ink-2: "#333a45"
  ink-3: "#5a6372"
  rule: "#111111"
  hair: "#c6ccd6"
  wash: "#eef2f8"
  blue: "#1f4e9e"
  blue-deep: "#173d7d"
  on-blue: "#ffffff"
  on-blue-2: "#d6e2f6"
  rail-rule: "#3a66b0"
  cited: "#fff1a8"
  cat-slur: "#2a78d6"
  cat-extremist_framing: "#eb6834"
  cat-identity_term: "#1baf7a"
  cat-topical: "#b3bac6"
  ok: "#17803d"
  bad: "#b3261e"
typography:
  title:
    fontFamily: "STIX Two Text, STIX Two, Cambria, Georgia, serif"
    fontSize: "2rem"
    fontWeight: 700
    lineHeight: 1.1
    letterSpacing: "-0.012em"
  brand:
    fontFamily: "STIX Two Text, STIX Two, Cambria, Georgia, serif"
    fontSize: "1.4rem"
    fontWeight: 700
    lineHeight: 1.1
  section-heading:
    fontFamily: "STIX Two Text, STIX Two, Cambria, Georgia, serif"
    fontSize: "1.18rem"
    fontWeight: 700
    lineHeight: 1.2
  card-heading:
    fontFamily: "STIX Two Text, STIX Two, Cambria, Georgia, serif"
    fontSize: "1.12rem"
    fontWeight: 700
  lede:
    fontFamily: "STIX Two Text, STIX Two, Cambria, Georgia, serif"
    fontSize: "1.06rem"
    fontWeight: 400
    lineHeight: 1.45
  body:
    fontFamily: "STIX Two Text, STIX Two, Cambria, Georgia, serif"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.45
  table:
    fontFamily: "STIX Two Text, STIX Two, Cambria, Georgia, serif"
    fontSize: "0.92rem"
    fontWeight: 400
  caption:
    fontFamily: "STIX Two Text, STIX Two, Cambria, Georgia, serif"
    fontSize: "0.9rem"
    fontWeight: 400
  footnote:
    fontFamily: "STIX Two Text, STIX Two, Cambria, Georgia, serif"
    fontSize: "0.86rem"
    fontWeight: 400
  control:
    fontFamily: "Source Sans 3, Source Sans Pro, system-ui, -apple-system, Segoe UI, sans-serif"
    fontSize: "0.92rem"
    fontWeight: 600
  note:
    fontFamily: "Source Sans 3, Source Sans Pro, system-ui, -apple-system, Segoe UI, sans-serif"
    fontSize: "0.86rem"
    fontWeight: 400
    lineHeight: 1.4
  mark:
    fontFamily: "Source Sans 3, Source Sans Pro, system-ui, -apple-system, Segoe UI, sans-serif"
    fontSize: "0.86rem"
    fontWeight: 600
  chart-label:
    fontFamily: "Source Sans 3, Source Sans Pro, system-ui, -apple-system, Segoe UI, sans-serif"
    fontSize: "0.84rem"
    fontWeight: 400
  code:
    fontFamily: "ui-monospace, SF Mono, Menlo, Consolas, monospace"
    fontSize: "0.82em"
rounded:
  mark: "2px"
  tip: "3px"
  control: "4px"
  point: "50%"
spacing:
  rail: "18rem"
  measure: "68ch"
  page-max: "62.5rem"
  prose-max: "46rem"
  page-flow: "1.9rem"
  section-flow: "0.75rem"
  column-gap: "2.2rem"
  cell-gap: "0.9rem"
components:
  button-primary:
    backgroundColor: "{colors.blue}"
    textColor: "{colors.on-blue}"
    typography: "{typography.control}"
    rounded: "{rounded.control}"
    padding: "0.5rem 0.95rem"
  button-primary-hover:
    backgroundColor: "{colors.blue-deep}"
    textColor: "{colors.on-blue}"
  button-outline:
    backgroundColor: "{colors.page}"
    textColor: "{colors.blue}"
    typography: "{typography.control}"
    rounded: "{rounded.control}"
    padding: "0.5rem 0.95rem"
  button-outline-hover:
    backgroundColor: "{colors.wash}"
    textColor: "{colors.blue}"
  select:
    backgroundColor: "{colors.page}"
    textColor: "{colors.ink}"
    rounded: "{rounded.control}"
    padding: "0.3rem 0.45rem"
  rail-item:
    backgroundColor: "{colors.blue}"
    textColor: "{colors.on-blue}"
    rounded: "{rounded.control}"
    padding: "0.45rem 0.55rem"
  rail-item-hover:
    backgroundColor: "{colors.blue-deep}"
    textColor: "{colors.on-blue}"
  rail-item-current:
    backgroundColor: "{colors.page}"
    textColor: "{colors.ink}"
  view-tab:
    textColor: "{colors.ink-2}"
    typography: "{typography.control}"
    padding: "0.35rem 0"
  view-tab-current:
    textColor: "{colors.ink}"
  page-sheet:
    backgroundColor: "{colors.page}"
    textColor: "{colors.ink}"
    padding: "2.4rem 3.4rem 3rem"
    width: "{spacing.page-max}"
  figure-frame:
    backgroundColor: "{colors.page}"
    padding: "0.85rem 0.95rem 0.6rem"
  command-block:
    backgroundColor: "{colors.wash}"
    textColor: "{colors.ink}"
    typography: "{typography.code}"
    rounded: "{rounded.mark}"
    padding: "0.6rem 0.75rem"
  footnote-cited:
    backgroundColor: "{colors.cited}"
    textColor: "{colors.ink}"
    typography: "{typography.footnote}"
    rounded: "{rounded.mark}"
    padding: "0.12rem 0.25rem"
  matrix-cell-clear:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.page}"
    padding: "0.24rem 0.5rem"
  tooltip:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.page}"
    rounded: "{rounded.tip}"
    padding: "0.35rem 0.55rem"
---

# Design System: Audit workbench

## Overview

**Creative North Star: "The Proceedings Page"**

The audit interface is set as a page from conference proceedings: one white sheet on a pale blue-grey desk, with a citation-blue rail of classifiers beside it. The sheet reads like the results section the researcher is about to write. Sections are numbered, tables and figures are numbered as they are laid out, every caption says what the item cannot show, and every number names the file it was read from in a footnote set directly under the caption that first cites it.

The page is dense in the way a paper is dense. Type does the structural work: a bold serif title, an italic byline of model facts, numbered section heads, tables ruled heavy at top and bottom and light under the head. Controls are few and sit in a sans face so they never read as part of the typeset record. The page is light only (the stylesheet declares `color-scheme: light`), loads nothing from outside its own folder, and carries a print stylesheet that drops the rail, the view tabs, the actions and the disclosures and leaves the sheet.

The page refuses the experiment dashboard: there are no metric tiles, no dark rail of runs, no ranked bars. Classifiers are listed in their file order in every chart and table, never sorted by a value.

**Key Characteristics:**
- One white sheet (max width 62.5rem) on a pale desk, beside a fixed blue rail (18rem).
- Serif for everything a paper would typeset; sans for controls, status words, chart labels and notes.
- Tables ruled as in print: 2px rule above and below, 1px under the head, no vertical rules, no fills.
- Numbered sections, tables and figures; each caption states a limit.
- Sources as footnotes under the caption that first cites them, lit when their mark is pointed at or focused.
- Colour does four jobs only: the rail and actions, a status, an attribution category, a cited source.
- State changes happen in place in 120 to 200 ms; nothing animates on entry.

## Colors

A near-monochrome page with one citation blue. Every other colour on the sheet is data, a status, or the highlight of a cited source.

### Primary
- **Citation Blue** (`blue`): the rail, links, both buttons, the current view's underline, the focus ring, the select's border, disclosure summaries, the "running" status mark, and the one dot that stands for "this classifier" in the accuracy figure.
- **Citation Blue, deep** (`blue-deep`): hover on links, on the primary button and on rail items.
- **On Blue** (`on-blue`) and **On Blue, quiet** (`on-blue-2`): text on the rail. The quiet tint carries the rail's group labels, the brand note, the hint and the classifier numbers.
- **Rail Rule** (`rail-rule`): the border of the collapsed classifier list at 900px and below, and the rail's scrollbar thumb.

### Secondary
- **Cited Yellow** (`cited`): a footnote whose mark is pointed at, focused or targeted; the background of a footnote mark on hover; the flash a finished audit stage settles from; the text selection colour. It always means "this one, now" and never stays on the page at rest.

### Tertiary
The four attribution categories. The names and the count are fixed by the research; the stylesheet names each token after its category key, and the legend reads the labels from the server.
- **Slur** (`cat-slur`): blue.
- **Extremist framing** (`cat-extremist_framing`): orange.
- **Identity term** (`cat-identity_term`): green.
- **Topical** (`cat-topical`): a neutral grey for every word no lexicon lists. It is the default, not a fourth hue.

The stylesheet records the intent beside the tokens: three hues that stay apart under colour-vision deficiency, and a neutral for the default. Category colours appear only as small square legend keys (0.7rem) and as the segments of a share bar, always with the category's name in a legend and the values in the table that follows.

Statuses:
- **Met Green** (`ok`): the "met" and "done" marks.
- **Failed Red** (`bad`): the "missing", "invalid" and "failed" marks, and the 1px border of an error message.

### Neutral
- **Desk** (`desk`): the ground behind the sheet.
- **Page** (`page`): the sheet, button and select backgrounds, and the current rail item.
- **Ink** (`ink`): body text, table rules (`rule` holds the same value), the dots of every classifier but the selected one, a matrix cell that clears significance, and the tooltip.
- **Ink 2** (`ink-2`): the byline, notes, legends, stage messages and idle view tabs.
- **Ink 3** (`ink-3`): small detail lines in table cells, axis figures, stage numbers, the "not provided" and "not reached" marks, bars in the stability table, the dashed border of an empty state.
- **Hair** (`hair`): figure frames, axis ticks, the track line of a dot plot, the divider above classifiers outside the comparison.
- **Wash** (`wash`): command and log blocks, the detail line of a stopped run, hover on the outline button.

Three raw colours sit outside the tokens: the window scrollbar thumb (`#9aa6b8`), the lower edge of the sheet's shadow (`#b8c2d2`), and literal white (`#fff`) for the inner stroke of a filled status mark.

### Named Rules
**The Four Jobs Rule.** Colour does four jobs only: the citation-blue rail and actions, a status, an attribution category, and the mark of a cited source. A new element that needs colour for a fifth reason is set in ink instead.

**The Four Categories Rule.** There are exactly four attribution categories: slur, extremist framing, identity term, topical. Each has one fixed colour. There is no fifth colour and no generic series slot; a chart of anything other than categories uses ink, a ring and the one blue dot.

**The Mark And Word Rule.** A status is never colour alone. Each status has its own drawn shape (a filled circle with a tick, a filled circle with a cross, a ring with a centre dot, a ring with a bar, a dashed ring) and a word. On the rail, where every mark is white on blue, the shape carries the difference and the word is kept for screen readers.

## Typography

**Display Font:** STIX Two Text (with STIX Two, Cambria, Georgia, serif)
**Body Font:** STIX Two Text, roman and italic
**Label/Mono Font:** Source Sans 3 (with Source Sans Pro, system-ui, sans-serif) for controls and labels; the platform monospace (`ui-monospace`, SF Mono, Menlo, Consolas) for file paths and commands

**Character:** A scientific text face carries the whole record, title to footnote, so the page reads as one typeset document. A plain humanist sans marks everything the reader operates or that annotates a chart. Both faces are bundled as variable files (weights 400 to 700) under the page's own folder with their OFL notices; the monospace is not bundled.

The root size is 17px, and 16px at 900px and below. All sizes are in rem against it.

### Hierarchy
- **Title** (700, 2rem, 1.1, -0.012em; 1.55rem at 900px and below): the classifier's name or the view's name, once per sheet. Balanced wrapping; long names may break anywhere.
- **Byline** (italic 400, 1rem, max 68ch): model facts as one italic sentence under the title.
- **Section heading** (700, 1.18rem, 1.2): numbered section heads; the number sits in a 1.5rem slot before the title.
- **Card heading** (700, 1.12rem) and **sub-heading** (700, 1rem): headings inside a report card and inside sections.
- **Lede** (400, 1.06rem, max 68ch): the opening paragraph of the two reading views.
- **Body** (400, 1rem, 1.45, max 68ch; prose blocks max 46rem): running text.
- **Table** (400, 0.92rem; header 600): table text, with tabular lining figures. The stage list of a run shares this size.
- **Caption** (400, 0.9rem, max 68ch; label 700): "Table N." or "Figure N." in bold, then the sentence that states the limit.
- **Footnote** (400, 0.86rem): the source list under a caption. Footnote marks in text are sans at 0.68em.
- **Control** (sans 600, 0.92rem): buttons and view tabs.
- **Note** (sans 400, 0.86rem, 1.4, max 68ch): side notes, disclosure summaries, detail lines. Status marks use the same size at 600.
- **Chart label** (sans 400, 0.84rem): legends, chart rows, the pairwise matrix and stage messages; small detail under a table cell is 0.8rem, axis figures 0.76rem.
- **Code** (mono, 0.82em): paths and commands in running text. Inside a table, code takes the table's own face and size.

### Named Rules
**The Two Voices Rule.** Serif is for what a paper would typeset: titles, text, tables, captions, footnotes. Sans is for what the reader operates or what annotates a chart: buttons, tabs, status words, legends, axis figures, notes. Nothing else chooses the face.

**The Caption States A Limit Rule.** Every table and figure has a numbered caption directly beneath it, and the caption says what the item cannot show. Numbers run in the order the items are laid out on the sheet, per view.

**The One Line Rule.** A numeric table cell is right-aligned, header included, and never wraps. A column is numeric when every filled cell in it is a number. Short text cells (24 characters or fewer) are also kept on one line; cells of 36 characters or more wrap.

## Layout

A two-column shell: the rail (18rem, sticky, full viewport height, scrolls on its own) and the desk. On the desk, a row of three view tabs with the slur toggle pushed to the right, then the sheet (max 62.5rem, padding 2.4rem 3.4rem 3rem, min height 60vh). Top-level blocks on the sheet are 1.9rem apart; blocks inside a section 0.75rem; text measures stop at 68ch and prose blocks at 46rem.

The first view sets its first two sections side by side in two equal columns with a 2.2rem gap; everything after runs full width. Chart rows share one grid (label up to 13rem, track, a 3.2rem value) so dots and bars in different figures align.

One breakpoint, at 900px:
- The shell becomes one column. The rail moves above the desk, stops being sticky, and its list collapses into a disclosure whose summary names the current classifier or the count on file.
- The sheet's padding drops to 1.4rem 1.1rem 2rem and the desk's to 0.8rem 0.6rem 3rem; the two columns stack with a 1.9rem gap.
- Chart labels move to their own line above the track. Stage rows drop to number, name and mark, with the message on a second line.
- A table of four or more columns takes a minimum width of 34rem and keeps every row on one line, so it scrolls sideways inside its own box; a long text column in it is given 30rem and wraps. A three-column table fits the screen.

Any table box that is wider than its column, at any width, shows a soft ink gradient (0.6rem, 20% ink) on the side that has more, becomes focusable and is announced as a region that scrolls sideways.

Commands are split into tokens at spaces and each token is kept whole, so a flag is never parted from its dashes. A single token longer than the line is the only thing that breaks inside. Log output is not tokenised and may break anywhere.

## Elevation & Depth

Flat, with one exception: the sheet. It lies on the desk with a single paper shadow, and nothing on the sheet is raised above it. Structure inside the sheet comes from rules, hairlines and space. Figures are framed by a 1px hairline, not lifted.

### Shadow Vocabulary
- **Sheet on desk** (`box-shadow: 0 1px 0 #b8c2d2, 0 5px 16px rgba(0, 0, 0, 0.10)`): the sheet only. Removed in print.
- **Point halo** (`box-shadow: 0 0 0 2px var(--page)`): a page-coloured ring that separates a data point from the track line and from a neighbouring point. A ring point adds `inset 0 0 0 2px var(--ink)`.

### Named Rules
**The One Sheet Rule.** Only the sheet casts a shadow. Buttons, figures, tables and messages are flat at rest and flat on hover.

**The In Place Rule.** State changes happen where they are: a hover tint in 140 ms, a footnote lighting in 160 ms, a data point growing to 1.25 in 120 ms, a button sinking 1px when pressed, and one 200 ms settle from cited yellow to clear when an audit stage finishes. All use one ease-out curve, `cubic-bezier(.16, 1, .3, 1)`. Nothing animates on entry, and under reduced motion every transition and animation is off.

## Shapes

Square by default. The sheet, figure frames, tables, share-bar segments, legend keys, empty and error boxes have no radius. Controls take a small radius (4px: buttons, the select, rail items, the collapsed rail summary); highlights and blocks take a smaller one (2px: the focus ring, footnote highlights, command and log blocks); the tooltip is 3px. Full circles are reserved for data points and their legend keys, and for the outline of status marks.

Rules have three weights: 2px ink above and below a table or a stage list, 1px ink under a table head and above a footnote list, 1px hairline for figure frames and chart ticks. The current view tab carries a 3px blue underline. A dashed 1px border marks an empty state; a solid 1px red border marks an error. Share-bar segments are separated by a 2px gap of page.

Status marks are drawn at 16 by 16 units and shown at 1rem, in one weight: filled shapes with a 1.7 stroke inside, outline shapes with a 1.5 stroke.

## Components

### Buttons
- **Shape:** gently squared (4px), 1px blue border, sans 600 at 0.92rem, padding 0.5rem 0.95rem.
- **Primary:** blue fill, white text. One per view at most; it starts the audit.
- **Outline:** page fill, blue text; hover fills with wash.
- **Hover / Focus / Active:** background and colour change in 140 ms; press moves the button down 1px; focus is the shared 2px blue outline at 2px offset.
- **Disabled:** 55% opacity, default cursor, no press movement. The label says why ("Audit running", "Another audit is running").

### View tabs
Three text buttons in sans 600, ink-2 at rest, blue on hover, ink with a 3px blue underline when current. They wrap on narrow screens. The slur toggle is a native checkbox (0.95rem, blue accent) with its label in regular weight at the end of the row.

### The rail
- **Style:** citation-blue column with the page name in bold serif (1.4rem), a one-line note, and classifiers in two labelled groups: in the paper's comparison, and outside it.
- **Item:** a full-width button on a three-part grid: number (tabular, quiet tint), name (may break anywhere), contract mark. Hover is the deep blue; the current item is a white slip with ink text at 600 and its mark in status green or red.
- **Focus:** the outline turns white on the rail.
- **Narrow screens:** the list collapses into a bordered disclosure.

### Tables
- **Style:** full width, collapsed borders, 0.92rem serif with tabular lining figures. Header cells are 600, left-aligned, bottom-aligned, balanced. Cells are top-aligned with 0.9rem between columns and no padding on the outer edges.
- **Numbers:** right-aligned with their header, one line.
- **Detail lines:** a second line under a cell, sans 0.8rem in ink-3.
- **Pairwise matrix:** a compact sans variant (0.84rem, centred cells, natural width). A cell that clears the significance test is filled ink with page text at 700; every other cell is plain; the diagonal is a hairline-coloured dot. Each cell is focusable and explains itself in a tooltip.
- **Stage list:** an ordered list ruled like a table, on a four-part grid of number, stage name, status mark and message.

### Figures
- **Frame:** 1px hairline box, padding 0.85rem 0.95rem 0.6rem, legend first.
- **Dot plot:** one row per classifier on a hairline track with ticks; ink dots (0.66rem) with a page halo, a ring for the comparison condition, a 2px ink-3 link between paired points, and one larger blue dot (0.86rem) for the selected classifier. The axis is drawn under the rows in ink-3, and the caption says where it starts.
- **Share bar:** segments in the four category colours, 1.55rem tall when it stands alone (with the percentage of a segment of 15% or more written under it in ink, never on the category colour) and 0.85rem in a row per classifier. Each segment is focusable and names its category and share in a tooltip.
- **Inline bar:** a 0.5rem ink-3 bar beside its number inside a table cell.

### Captions and footnotes (signature)
A caption is one paragraph: bold label, the limit, then superscript marks in blue sans. The sources first cited in a section are listed directly beneath, under a 1px ink rule, sized to their content (min 11rem, max 46rem). Pointing at or focusing a mark tints the mark and lights its footnote in cited yellow; following the link does the same through the target. The mark's tooltip reads the file path.

### Status marks
Icon plus word in sans 0.86rem at 600: green for met and done, red for missing, invalid and failed, blue for running. "Not provided", "skipped" and "not reached" are ink-3 at regular weight. Marks never wrap.

### Commands, logs and disclosures
Command and log blocks sit on wash with a 2px radius and 0.6rem 0.75rem padding, in the monospace at 0.82em and 1.5 line height; a log is capped at 20rem and scrolls. Secondary material (the command-line equivalent, full output) sits behind a native disclosure with a blue sans summary.

### Messages
- **Empty state:** dashed 1px ink-3 border, a bold sentence saying what is missing and a sentence saying what produces it.
- **Error:** solid 1px red border, sans 0.92rem, the server's own words and the next step.
- **Detail:** a wash strip in sans 0.86rem for why a run stopped.
- **Tooltip:** ink box, page text, sans 0.82rem, 3px radius, max 19rem; placed above its target, or below when there is no room; shown on hover and on focus.

### Fields
The only field is a native select: page fill, 1px blue border, 4px radius, padding 0.3rem 0.45rem. There is no free-text input anywhere.

## Do's and Don'ts

### Do:
- **Do** mask slurs by default on every screen that shows words, and keep the one visible toggle that asks for them verbatim. A new screen that lists words inherits the mask.
- **Do** show counts, metrics, file paths and word-level tables only. Every number cites its file with a footnote mark.
- **Do** give every table and figure a numbered caption directly beneath it that states what the item cannot show.
- **Do** pair every status with its drawn shape and a word; where the word is hidden to save room, keep it for screen readers and let the shape, not the colour, carry the difference.
- **Do** use the four category colours for the four categories only, with the category's name in a legend and the values in a table nearby.
- **Do** right-align numeric columns with their headers and keep them on one line; let a table of four or more columns scroll sideways at 900px and below with the soft edge cue and keyboard focus.
- **Do** keep classifiers in file order in every chart and table, and say in the caption when an axis does not start at zero.
- **Do** set typeset content in the serif and controls and chart labels in the sans.
- **Do** keep state changes in place and under 300 ms, on the one ease-out curve, and off under reduced motion.
- **Do** keep every asset inside the page's own folder; the page works offline.

### Don't:
- **Don't** show post text, a row id or a text hash on any screen, in any example, tooltip or caption.
- **Don't** add a fifth attribution category, a fifth category colour, or a generic series colour.
- **Don't** signal a status, a category or a significant cell by colour alone.
- **Don't** rank classifiers: no sorting by a metric, no leaderboard, no "best" label, no metric tiles.
- **Don't** raise anything above the sheet: no shadows on buttons, figures or messages.
- **Don't** add vertical rules, row fills or zebra striping to tables.
- **Don't** split a command inside a token, or part a flag from its dashes.
- **Don't** animate anything on entry, and don't add a dark theme; the page is light only by decision.
- **Don't** load a font, script, image or stylesheet from outside the page's folder.
- **Don't** add a free-text input. The page audits classifiers; it does not classify text.
