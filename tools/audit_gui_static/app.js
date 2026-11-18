/* The audit interface: behaviour. Everything shown is read from the server's
   JSON, which reads committed files only. Text from those files is always
   placed as text, never as markup. The page is set like a proceedings page:
   sections, tables and figures are numbered as they are laid out, every
   caption says what the item cannot show, and sources are footnotes. */
(() => {
  "use strict";

  const TOKEN = document.querySelector('meta[name="audit-token"]').content;
  const sheet = document.getElementById("sheet");
  const rail = document.getElementById("rail");
  const layout = document.getElementById("layout");
  const tip = document.getElementById("tip");
  const verbatimBox = document.getElementById("verbatim");
  const SVG = "http://www.w3.org/2000/svg";
  const wide = window.matchMedia("(min-width: 901px)");

  const state = {
    view: "technique", name: null, overview: null, technique: null, compare: null, job: null,
    timer: null, request: 0, notes: [], counts: { section: 0, table: 0, figure: 0 }, settled: new Set(),
  };

  // ---- helpers -------------------------------------------------------------
  function h(tag, attrs, ...kids) {
    const el = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs || {})) {
      if (value === null || value === undefined || value === false) continue;
      if (key === "class") el.className = value;
      else if (key === "text") el.textContent = value;
      else if (key === "tip") el.dataset.tip = value;
      else if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
      else el.setAttribute(key, value === true ? "" : value);
    }
    for (const kid of kids.flat(Infinity)) {
      if (kid === null || kid === undefined || kid === false) continue;
      el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    }
    return el;
  }

  function shape(tag, attrs) {
    const el = document.createElementNS(SVG, tag);
    for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
    return el;
  }

  // Status marks, drawn in one weight. A mark never stands without its word.
  function icon(kind) {
    const svg = shape("svg", { viewBox: "0 0 16 16", "aria-hidden": "true", focusable: "false" });
    if (kind === "ok") {
      svg.append(shape("circle", { class: "out", cx: 8, cy: 8, r: 7 }), shape("path", { class: "in", d: "M4.7 8.3l2.2 2.2 4.5-4.7" }));
    } else if (kind === "bad") {
      svg.append(shape("circle", { class: "out", cx: 8, cy: 8, r: 7 }), shape("path", { class: "in", d: "M5.4 5.4l5.2 5.2M10.6 5.4l-5.2 5.2" }));
    } else if (kind === "run") {
      svg.append(shape("circle", { class: "line", cx: 8, cy: 8, r: 6.3 }), shape("circle", { class: "out", cx: 8, cy: 8, r: 3 }));
    } else if (kind === "none") {
      svg.append(shape("circle", { class: "line", cx: 8, cy: 8, r: 6.3 }), shape("path", { class: "line", d: "M5 8h6" }));
    } else {
      svg.append(shape("circle", { class: "line dashed", cx: 8, cy: 8, r: 6.3 }));
    }
    return svg;
  }

  function mark(kind, word, quiet) {
    return h("span", { class: `mark ${kind}` }, icon(kind), h("span", { class: quiet ? "sr" : null, text: word }));
  }

  const missing = (v) => v === null || v === undefined || Number.isNaN(v);
  const num = (v, digits = 3) => (missing(v) ? "not given" : Number(v).toFixed(digits));
  const pct = (v, digits = 1) => (missing(v) ? "not given" : `${(Number(v) * 100).toFixed(digits)}%`);
  const pval = (v) => (missing(v) ? "not given" : v < 0.001 ? Number(v).toExponential(1) : Number(v).toFixed(3));
  const signed = (v) => (missing(v) ? "not given" : `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v)}`);
  const code = (text) => h("code", { text });
  const plural = (n, one, many) => (n === 1 ? one : many);

  async function api(path, options) {
    let response;
    try {
      response = await fetch(path, options);
    } catch (_) {
      const error = new Error("The server did not answer.");
      error.kind = "network";
      throw error;
    }
    let body = null;
    try { body = await response.json(); } catch (_) { /* a non-JSON answer is reported below */ }
    if (!response.ok) {
      const error = new Error((body && body.error) || `The server answered ${response.status}.`);
      error.kind = "server";
      throw error;
    }
    return body;
  }

  function splitName(name) {
    const match = /^(\d{2})_(.*)$/.exec(name);
    return match ? { number: match[1], rest: match[2] } : { number: "", rest: name };
  }

  function categoryLabel(key) {
    const found = (state.overview ? state.overview.categories : []).find((c) => c.key === key);
    return found ? found.label : key;
  }

  // ---- numbering, captions and footnotes --------------------------------------
  function beginPage() {
    state.notes = [];
    state.fresh = [];
    state.jobNumber = null;
    state.counts = { section: 0, table: 0, figure: 0 };
  }

  // A superscript for each source path, numbered by first use on the page.
  function ref(paths) {
    const marks = (paths || []).filter(Boolean).map((path) => {
      let index = state.notes.indexOf(path);
      if (index < 0) { state.notes.push(path); index = state.notes.length - 1; state.fresh.push(index); }
      return h("a", { href: `#note-${index + 1}`, "data-note": String(index + 1), "aria-label": `Source ${index + 1}: ${path}`,
        tip: `Read from ${path}` }, String(index + 1));
    });
    if (!marks.length) return null;
    const sup = h("sup", { class: "ref" });
    marks.forEach((m, i) => { if (i) sup.append(","); sup.append(m); });
    return sup;
  }

  function caption(kind, text, paths) {
    state.counts[kind] += 1;
    const label = `${kind === "table" ? "Table" : "Figure"} ${state.counts[kind]}.`;
    return [h("p", { class: "cap" }, h("b", { text: label }), " ", text, ref(paths)), footNotes()];
  }

  // The sources first cited in a section are set at its foot, beside their marks.
  function footNotes() {
    const fresh = state.fresh;
    state.fresh = [];
    if (!fresh.length) return null;
    return h("ol", { class: "notes", "aria-label": "Sources" }, fresh.map((index) =>
      h("li", { id: `note-${index + 1}` }, h("sup", { text: String(index + 1) }),
        h("span", null, "Read from ", code(state.notes[index]), "."))));
  }

  function section(title, ...kids) {
    state.counts.section += 1;
    return h("section", { class: "sec" }, h("h2", null, h("span", { class: "n", text: String(state.counts.section) }), title), kids, footNotes());
  }

  // A number and its source are linked: pointing at a mark lights its footnote.
  function noteFor(event) {
    const link = event.target.closest ? event.target.closest("sup.ref a") : null;
    return link ? document.getElementById(`note-${link.dataset.note}`) : null;
  }
  document.addEventListener("mouseover", (event) => { const li = noteFor(event); if (li) li.classList.add("cited"); });
  document.addEventListener("mouseout", (event) => { const li = noteFor(event); if (li) li.classList.remove("cited"); });
  document.addEventListener("focusin", (event) => { const li = noteFor(event); if (li) li.classList.add("cited"); });
  document.addEventListener("focusout", (event) => { const li = noteFor(event); if (li) li.classList.remove("cited"); });

  // ---- tables ---------------------------------------------------------------
  // A column is right-aligned, header included, when every filled cell in it is a number.
  const NUMERIC = /^[+\-\u2212]?\d[\d.,]*(?:e[+\-]?\d+)?[%\u00d7]?(?: \(\d+\)| \u00b1 [\d.]+%)?$/i;
  function cellText(cell) {
    if (cell === null || cell === undefined) return "";
    if (Array.isArray(cell)) return cell.map(cellText).join("").trim();
    if (cell.nodeType) return cell.classList && (cell.classList.contains("row") || cell.classList.contains("mark")) ? "mark" : cell.textContent.trim();
    return String(cell).trim();
  }
  function table(head, rows, className) {
    const numeric = head.map((_, i) => {
      const texts = rows.map((row) => cellText(row[i]));
      return texts.some((text) => text !== "") && texts.every((text) => text === "" || NUMERIC.test(text));
    });
    const classes = [className, head.length >= 4 ? "wide" : null].filter(Boolean).join(" ");
    return h("div", { class: "scroll" },
      h("table", { class: classes || null },
        h("thead", null, h("tr", null, head.map((cell, i) => h("th", { scope: "col", class: numeric[i] ? "n" : null }, cell)))),
        h("tbody", null, rows.map((row) => h("tr", null, row.map((cell, i) =>
          h("td", { class: numeric[i] ? "n" : null }, cell)))))));
  }

  // A table wider than its column scrolls sideways: say so, and let the keyboard reach it.
  function markScrollers() {
    for (const box of document.querySelectorAll(".scroll")) {
      const cut = box.scrollWidth > box.clientWidth + 1;
      box.classList.toggle("cut", cut);
      if (cut) {
        box.setAttribute("tabindex", "0");
        box.setAttribute("role", "region");
        box.setAttribute("aria-label", "Table, scrolls sideways");
      } else {
        box.removeAttribute("tabindex");
        box.removeAttribute("role");
        box.removeAttribute("aria-label");
      }
    }
  }
  window.addEventListener("resize", markScrollers);

  function showError(error) {
    const recovery = error.kind === "server"
      ? "The server is running and reported this itself. Fix what it names, then reload the page."
      : "Check that the server is still running in its terminal, then reload the page.";
    sheet.replaceChildren(h("div", { class: "error" },
      h("p", null, h("strong", { text: "This could not be loaded. " }), String(error.message || error)),
      h("p", { text: recovery })));
  }

  // ---- tooltip ---------------------------------------------------------------
  function showTip(target) {
    tip.textContent = target.dataset.tip;
    tip.hidden = false;
    const box = target.getBoundingClientRect();
    const own = tip.getBoundingClientRect();
    const left = Math.min(Math.max(8, box.left + box.width / 2 - own.width / 2), window.innerWidth - own.width - 8);
    const above = box.top - own.height - 8;
    tip.style.left = `${left}px`;
    tip.style.top = `${above < 8 ? box.bottom + 8 : above}px`;
  }
  const tipTarget = (event) => (event.target.closest ? event.target.closest("[data-tip]") : null);
  document.addEventListener("mouseover", (event) => { const el = tipTarget(event); if (el) showTip(el); });
  document.addEventListener("focusin", (event) => { const el = tipTarget(event); if (el) showTip(el); });
  document.addEventListener("mouseout", (event) => { if (tipTarget(event)) tip.hidden = true; });
  document.addEventListener("focusout", (event) => { if (tipTarget(event)) tip.hidden = true; });
  document.addEventListener("scroll", () => { tip.hidden = true; }, true);

  // ---- chart pieces ----------------------------------------------------------
  function categoryLegend() {
    return h("div", { class: "legend" }, state.overview.categories.map((c) =>
      h("span", null, h("span", { class: `key c-${c.key}` }), c.label)));
  }

  function segments(categories, tall) {
    const parts = categories.filter((c) => !missing(c.share) && c.share > 0);
    const summary = categories.map((c) => `${categoryLabel(c.key)} ${pct(c.share)}`).join(", ");
    return h("div", { class: tall ? "seg" : "seg thin", role: "img", "aria-label": summary },
      parts.map((c) => {
        const el = h("span", { class: `c-${c.key}`, tabindex: "0", tip: `${categoryLabel(c.key)}: ${pct(c.share)} of positive attribution` },
          tall && c.share >= 0.15 ? pct(c.share) : null);
        el.style.flex = `${c.share} 1 0`;
        return el;
      }));
  }

  function axis(lo, hi, step, format) {
    const el = h("div", { class: "axis", "aria-hidden": "true" });
    for (let v = lo; v <= hi + 1e-9; v += step) {
      const tick = h("span", { text: format(v) });
      tick.style.left = `${((v - lo) / (hi - lo)) * 100}%`;
      el.append(tick);
    }
    return el;
  }

  function track(lo, hi, step, points, link) {
    const el = h("div", { class: "track" });
    for (let v = lo; v <= hi + 1e-9; v += step) {
      const tick = h("span", { class: "tick" });
      tick.style.left = `${((v - lo) / (hi - lo)) * 100}%`;
      el.append(tick);
    }
    const at = (v) => `${Math.min(100, Math.max(0, ((v - lo) / (hi - lo)) * 100))}%`;
    if (link && !missing(link[0]) && !missing(link[1])) {
      const bar = h("span", { class: "link" });
      bar.style.left = at(Math.min(link[0], link[1]));
      bar.style.width = `${(Math.abs(link[1] - link[0]) / (hi - lo)) * 100}%`;
      el.append(bar);
    }
    for (const point of points) {
      if (missing(point.value)) continue;
      const dot = h("span", { class: `pt${point.ring ? " ring" : ""}${point.own ? " this" : ""}`, tabindex: "0", tip: point.tip });
      dot.style.left = at(point.value);
      el.append(dot);
    }
    return el;
  }

  function who(row, own) {
    const parts = splitName(row.technique);
    return h("span", { class: own ? "who this" : "who" }, `${parts.number} ${parts.rest}`.trim(),
      row.in_comparison === false ? h("small", { text: "outside the comparison" }) : null);
  }

  function chartRows(rows, build) {
    let divided = false;
    return h("div", { class: "rows" }, rows.map((row) => {
      const el = build(row);
      if (row.in_comparison === false && !divided) { el.classList.add("divide"); divided = true; }
      return el;
    }));
  }

  function accuracyScale(values) {
    return { lo: Math.floor((Math.min(...values) - 0.005) * 20) / 20, hi: Math.ceil((Math.max(...values) + 0.005) * 20) / 20 };
  }

  function accuracyFigure(data, own) {
    const order = [...data.in_comparison, ...data.outside_comparison];
    const rows = [...data.rq1.accuracy].sort((a, b) => order.indexOf(a.technique) - order.indexOf(b.technique));
    const { lo, hi } = accuracyScale(rows.map((row) => row.accuracy));
    const drawn = rows.some((row) => row.technique === own);
    return {
      lo,
      drawn,
      node: h("div", { class: "fig" },
        drawn ? h("div", { class: "legend" }, h("span", null, h("span", { class: "key this" }), "This classifier"),
          h("span", null, h("span", { class: "key dot" }), "The others")) : null,
        chartRows(rows, (row) => h("div", { class: "row" }, who(row, row.technique === own),
          track(lo, hi, 0.05, [{ value: row.accuracy, own: row.technique === own, tip: `${row.technique}: ${num(row.accuracy)}` }]),
          h("span", { class: "val", text: num(row.accuracy) }))),
        h("div", { class: "row", "aria-hidden": "true" }, h("span", { class: "who" }), axis(lo, hi, 0.05, (v) => v.toFixed(2)), h("span"))),
    };
  }

  function fprScale(values) {
    const top = Math.max(0.05, ...values.filter((v) => !missing(v)));
    const hi = Math.ceil(top * 20) / 20;
    return { lo: 0, hi, step: hi > 0.3 ? 0.1 : 0.05 };
  }

  function fprLegend() {
    return h("div", { class: "legend" },
      h("span", null, h("span", { class: "key ring" }), "Posts without an identity term"),
      h("span", null, h("span", { class: "key dot" }), "Posts with an identity term"));
  }

  // ---- markdown, as much of it as a report card uses ---------------------------
  function inline(text) {
    const out = [];
    const pattern = /(`[^`]+`|\*\*[^*]+\*\*)/g;
    let last = 0;
    let match;
    while ((match = pattern.exec(text))) {
      if (match.index > last) out.push(text.slice(last, match.index));
      const token = match[0];
      out.push(token[0] === "`" ? code(token.slice(1, -1)) : h("strong", { text: token.slice(2, -2) }));
      last = match.index + token.length;
    }
    if (last < text.length) out.push(text.slice(last));
    return out;
  }

  // What each table of a report card cannot show, by the section it sits in.
  const CARD_CAPTIONS = [
    [/^RQ1/, ["Accuracy gaps against every other classifier. A gap that does not clear Holm-corrected McNemar significance is not a difference."]],
    [/^SHAP pipeline check/, ["The explainer against known coefficients. It checks the pipeline on a linear model, not the explanations of any other model."]],
    [/^RQ2 support/, ["The deletion test. It shows that the listed words move this model, not that the model is right to use them."]],
    [/^RQ2/, ["The words with the largest positive mean attribution. A word with support 1 comes from a single post.",
      "Word-level agreement with the other classifiers, over the words they share."]],
    [/^RQ3/, ["Agreement between this classifier's runs, which explain different posts."]],
    [/^RQ4/, ["False positives with and without identity terms. A correlation over few posts."]],
    [/^WP7/, ["The identity counterfactual. A flip shows that one edit crossed the locked threshold, not that the term causes the error."]],
  ];
  function cardCaption(heading, index) {
    const found = CARD_CAPTIONS.find(([pattern]) => pattern.test(heading));
    const list = found ? found[1] : [];
    return list[Math.min(index, list.length - 1)] || "From the report card as written.";
  }

  function renderMarkdown(markdown) {
    const lines = markdown.split("\n");
    const nodes = [];
    let heading = "";
    let tablesHere = 0;
    const cells = (line) => line.trim().replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim());
    const isRow = (line) => line.trim().startsWith("|");
    const isItem = (line) => /^\s*[*-] /.test(line);
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (!line.trim() || line.startsWith("# ")) { i += 1; continue; } // the page already carries the title
      if (line.startsWith("## ")) { heading = line.slice(3); tablesHere = 0; nodes.push(h("h3", null, inline(heading))); i += 1; continue; }
      if (line.startsWith("### ")) { nodes.push(h("h4", null, inline(line.slice(4)))); i += 1; continue; }
      if (isRow(line) && i + 1 < lines.length && /^\s*\|[\s:|-]+\|\s*$/.test(lines[i + 1])) {
        const head = cells(line).map(inline);
        const body = [];
        i += 2;
        while (i < lines.length && isRow(lines[i])) { body.push(cells(lines[i]).map(inline)); i += 1; }
        nodes.push(table(head, body), caption("table", cardCaption(heading, tablesHere)));
        tablesHere += 1;
        continue;
      }
      if (isItem(line)) {
        const items = [];
        while (i < lines.length && isItem(lines[i])) { items.push(h("li", null, inline(lines[i].replace(/^\s*[*-] /, "")))); i += 1; }
        nodes.push(h("ul", { class: "bullets" }, items));
        continue;
      }
      const paragraph = [];
      while (i < lines.length && lines[i].trim() && !lines[i].startsWith("#") && !isRow(lines[i]) && !isItem(lines[i])) {
        paragraph.push(lines[i].trim());
        i += 1;
      }
      nodes.push(h("p", null, inline(paragraph.join(" "))));
    }
    return nodes;
  }

  // ---- the rail ----------------------------------------------------------------
  function renderRail() {
    const techniques = state.overview.techniques;
    const item = (t) => {
      const parts = splitName(t.name);
      return h("li", null, h("button", {
        class: "tech", type: "button", "aria-current": t.name === state.name && state.view === "technique" ? "true" : null,
        onclick: () => selectTechnique(t.name),
      },
      h("span", { class: "num", text: parts.number }),
      h("span", { class: "nm", text: parts.rest }),
      h("span", { class: "flag" }, t.contract_met ? mark("ok", "contract met", true) : mark("bad", "contract not met", true))));
    };
    const inside = techniques.filter((t) => t.in_comparison);
    const outside = techniques.filter((t) => !t.in_comparison);
    const kids = [];
    if (inside.length) kids.push(h("h2", { text: "In the paper's comparison" }), h("ul", null, inside.map(item)));
    if (outside.length) kids.push(h("h2", { text: "Outside the comparison" }), h("ul", null, outside.map(item)));
    kids.push(h("p", { class: "hint" }, "A classifier is listed once its folder exists under ", code(`${state.overview.results_dir}/`),
      ". ", h("a", { href: "#add", text: "How to add yours" }), "."));
    const current = state.view === "technique" && state.name ? state.name : `${techniques.length} on file`;
    rail.replaceChildren(h("details", { class: "rail-list", open: wide.matches },
      h("summary", null, "Classifiers: ", h("b", { text: current })), kids));
  }

  // ---- one classifier --------------------------------------------------------------
  const ITEM_NAMES = [
    ["probability artifact (test)", "Test probabilities"],
    ["probability artifact (validation)", "Validation probabilities"],
    ["locked validation threshold", "Locked threshold"],
    ["whole-model attribution run (test)", "SHAP run on test posts"],
    ["whole-model attribution run (validation)", "SHAP run on validation posts"],
    ["coefficients.csv", "Word coefficients"],
    ["restorable predictor", "Restorable predictor"],
  ];
  const itemName = (name) => (ITEM_NAMES.find(([prefix]) => name.startsWith(prefix)) || [null, name])[1];

  function itemMark(item) {
    if (item.status === "OK") return mark("ok", "met");
    if (item.required) return mark("bad", item.status === "MISSING" ? "missing" : "invalid");
    return mark("none", item.status === "MISSING" ? "not provided" : "not available");
  }

  function head(t) {
    const facts = [];
    const words = (value) => String(value).replace(/_/g, " ");
    const sentence = (text) => text.charAt(0).toUpperCase() + text.slice(1);
    if (t.config.model_family) {
      facts.push(sentence(`${words(t.config.model_family)}${t.config.feature_family ? `, on ${words(t.config.feature_family)}` : ""}`));
    }
    if (!missing(t.config.selected_threshold)) facts.push(`Threshold ${t.config.selected_threshold}, locked on validation`);
    if (!missing(t.metrics.accuracy)) facts.push(`Test accuracy ${num(t.metrics.accuracy)} on ${state.overview.split.test} posts`);
    const more = [["balanced accuracy", t.metrics.balanced_accuracy], ["macro F1", t.metrics.f1_macro],
      ["ROC-AUC", t.metrics.roc_auc], ["PR-AUC", t.metrics.pr_auc]].filter(([, v]) => !missing(v)).map(([k, v]) => `${k} ${num(v)}`);
    return h("header", { class: "head" },
      h("h1", { class: "title", text: t.name }),
      facts.length ? h("p", { class: "byline", text: `${facts.join(". ")}.` }) : null,
      h("p", { class: "note", text: `${t.in_comparison
        ? "In the paper's comparison."
        : "Outside the paper's comparison: it has its own Holm family, and no table in the paper changes."}${more.length ? ` Also: ${more.join(", ")}.` : ""}` }));
  }

  function contractSection(t) {
    const required = t.contract.filter((item) => item.required);
    const failed = required.filter((item) => item.status !== "OK");
    const noRuns = failed.some((item) => item.name.startsWith("whole-model attribution"));
    const running = state.job && state.job.state === "running";
    const verdict = t.contract_met
      ? "Every required item is in place, so the audit can run."
      : `${failed.length} required ${plural(failed.length, "item is", "items are")} missing or invalid, so the audit cannot run yet.${noRuns
        ? " Without word-level attribution runs for test and validation a classifier cannot take part in RQ2 to RQ4, whatever its accuracy." : ""}`;
    return section("Contract",
      h("p", null, verdict, t.contract_met ? null : [" ", h("a", { href: "#add", text: "How to export the missing files" }), "."]),
      h("div", { class: "actions" },
        h("button", { class: "btn primary", type: "button", disabled: !t.contract_met || running, onclick: () => startRun(t.name) },
          running ? "Audit running" : "Run the audit"),
        h("button", { class: "btn", type: "button", onclick: () => refresh() }, "Check the contract again")),
      table(["Item", "Required", "Status"], t.contract.map((item) => [
        h("span", null, itemName(item.name), h("small", { text: item.detail })),
        item.required ? "yes" : "no",
        itemMark(item)])),
      caption("table", `The contract, checked now. ${required.length - failed.length} of ${required.length} required items are met.`),
      h("details", null, h("summary", { text: "The same on the command line" }),
        h("pre", { class: "cmd", text: `${t.commands.check}\n${t.commands.run}` }),
        h("p", { class: "note" }, "The identity counterfactual and the deletion faithfulness test load the model's weights, so this page does not start them: ",
          code(t.commands.perturbation))));
  }

  function standingSection(t) {
    const data = state.compare;
    if (!data || !data.rq1.accuracy.length) return null;
    const figure = accuracyFigure(data, t.name);
    const cleared = data.rq1.pairs.filter((pair) => pair.significant).length;
    const kids = [figure.node,
      caption("figure", `Test accuracy; the axis starts at ${figure.lo.toFixed(2)}, not at zero. The order is descriptive: ${
        data.rq1.pairs.length ? `${cleared} of the ${data.rq1.pairs.length} pairs in the comparison differ after correction.` : "no pair table has been written yet."}${
        figure.drawn ? "" : " This classifier has no test accuracy on file yet, so it is not drawn."}`,
      data.rq1.sources)];
    const pairs = t.card && t.card.rq1 ? t.card.rq1.pairs || [] : [];
    if (pairs.length) {
      kids.push(table(["Against", "Row gap", "Holm p", "Reading"], pairs.map((pair) => [
        pair.other_technique, signed(pair.row_gap), pval(pair.holm_p),
        pair.significant_holm ? "a measured difference" : "not distinguishable"])),
      caption("table", `Gaps in correct test rows, this classifier minus the other, with Holm-corrected McNemar p over ${t.card.rq1.holm_family_size} pairs. A gap that does not clear the correction is not a difference.`,
        t.card.rq1.sources));
    }
    return section("Where it stands", kids);
  }

  function stageMark(status) {
    if (status === "running") return mark("run", "running");
    if (status === "ok") return mark("ok", "done");
    if (status === "fail") return mark("bad", "failed");
    if (status === "skip") return mark("none", "skipped");
    return mark("pending", "not reached");
  }

  function jobPanel() {
    const job = state.job;
    if (!job) return h("div", { id: "job", hidden: true });
    const running = job.state === "running";
    const kids = [h("h2", null, running ? `Auditing ${job.technique}` : `The audit of ${job.technique}`),
      h("p", { role: "status", text: running ? "Running. Each stage is set down as it finishes." : job.outcome })];
    if (job.stages.stopped_at === "check-contract") {
      kids.push(h("p", { class: "detail", text: `Stopped at the contract check. ${job.stages.stopped_message || "A required item is missing or invalid."}` }));
    }
    const inHand = running ? job.stages.stages.findIndex((stage) => stage.status === "pending") : -1;
    kids.push(h("ol", { class: "stages" }, job.stages.stages.map((stage, index) => {
      const key = `${job.id}:${stage.name}`;
      const fresh = stage.status !== "pending" && !state.settled.has(key);
      if (stage.status !== "pending") state.settled.add(key);
      return h("li", { class: fresh ? "just" : null }, h("span", { class: "stage-name", text: stage.name }),
        stageMark(index === inHand ? "running" : stage.status), h("span", { class: "stage-msg", text: stage.message }));
    })));
    if (state.jobNumber === null) { state.counts.table += 1; state.jobNumber = state.counts.table; }
    kids.push(h("p", { class: "cap" }, h("b", { text: `Table ${state.jobNumber}.` }),
      " The audit chain, in the order it runs. A skipped stage did not run, and the table it would have written is left as it was."));
    if (!running && Array.isArray(job.changed)) {
      kids.push(h("h3", { text: "Files this run changed" }));
      kids.push(job.changed.length
        ? h("ul", { class: "changed" }, job.changed.map((file) => h("li", null, `${file.state}: `, code(file.path))))
        : h("p", { text: "None. The chain reproduced every table and card byte for byte." }));
    }
    kids.push(h("details", null, h("summary", { text: "Full output" }),
      h("pre", { class: "log", text: job.lines.join("\n") + (job.truncated ? "\n(the output was cut here)" : "") })));
    return h("section", { class: "sec", id: "job" }, kids);
  }

  function relianceSection(t) {
    const s = t.shares;
    return section("What it relies on",
      h("div", { class: "fig" }, categoryLegend(), segments(s.categories, true)),
      caption("figure", `Share of positive attribution toward EXTREMIST among the top ${s.top_k} words of the whole-model ${s.split} run, by lexicon category. Topical is every word no lexicon lists.`, state.compare ? state.compare.rq2.sources : null),
      table(["Category", "Share", "SD across runs", `Words in the top ${s.top_k}`], s.categories.map((c) => [
        h("span", { class: "cat" }, h("span", { class: `key c-${c.key}` }), categoryLabel(c.key)),
        pct(c.share), missing(c.std) ? "one run" : pct(c.std, 2), missing(c.n_words) ? "not given" : String(c.n_words)])),
      caption("table", "The same shares with their standard deviation across runs. A difference between two classifiers counts only when it exceeds the pooled standard deviation of the two."));
  }

  function fprSection(t) {
    const r = t.card.rq4;
    const scale = fprScale([r.fpr_with_identity_terms, r.fpr_without_identity_terms]);
    return section("False positives on posts that mention a community",
      h("div", { class: "fig" }, fprLegend(),
        h("div", { class: "rows" }, h("div", { class: "row" }, h("span", { class: "who", text: "Flagged in error" }),
          track(scale.lo, scale.hi, scale.step, [
            { value: r.fpr_without_identity_terms, ring: true, tip: `Without an identity term: ${pct(r.fpr_without_identity_terms)} flagged` },
            { value: r.fpr_with_identity_terms, tip: `With an identity term: ${pct(r.fpr_with_identity_terms)} flagged` },
          ], [r.fpr_without_identity_terms, r.fpr_with_identity_terms]), h("span")),
        h("div", { class: "row", "aria-hidden": "true" }, h("span", { class: "who" }), axis(scale.lo, scale.hi, scale.step, (v) => pct(v, 0)), h("span")))),
      caption("figure", `Non-extremist test posts flagged at the locked threshold ${r.threshold}. ${r.n_negative_with_identity_terms} of the ${r.n_negative} contain an identity term.`, r.sources),
      table(["Without a term", "With a term", "Ratio", "Fisher exact p"], [[
        pct(r.fpr_without_identity_terms), pct(r.fpr_with_identity_terms), num(r.fpr_ratio, 2), pval(r.fisher_p_value)]]),
      caption("table", "A correlation over few posts. A result that is not significant is not evidence that the effect is absent."));
  }

  function cardSection(t) {
    if (!t.card_markdown) {
      return section("Report card", h("div", { class: "empty" },
        h("p", null, h("strong", { text: "No report card yet." })),
        h("p", { text: t.contract_met ? "Run the audit above to write it." : "It is written by the audit, which runs once the contract is met." })));
    }
    const masked = t.slurs_masked
      ? ` ${t.slurs_masked} ${plural(t.slurs_masked, "word the lexicon marks as a slur is", "words the lexicon marks as slurs are")} masked here; the file keeps ${plural(t.slurs_masked, "it", "them")}.`
      : "";
    return section("Report card",
      h("p", { class: "note" }, "As written by ", code("audit.py report"), ".", ref([t.card_source]), masked),
      footNotes(),
      h("div", { class: "card" }, renderMarkdown(t.card_markdown)));
  }

  function renderTechnique() {
    const t = state.technique;
    if (!t) return;
    beginPage();
    const kids = [head(t), h("div", { class: "cols" }, contractSection(t), standingSection(t)), jobPanel()];
    if (t.shares && t.shares.categories) kids.push(relianceSection(t));
    if (t.card && t.card.rq4) kids.push(fprSection(t));
    kids.push(cardSection(t));
    sheet.replaceChildren(...kids.filter(Boolean));
    markScrollers();
  }

  // ---- across classifiers -----------------------------------------------------------
  function renderCompare(data) {
    beginPage();
    const order = [...data.in_comparison, ...data.outside_comparison];
    const byOrder = (rows) => [...rows].sort((a, b) => order.indexOf(a.technique) - order.indexOf(b.technique));
    const short = (name) => splitName(name).number || name;
    const kids = [h("header", { class: "head" },
      h("h1", { class: "title", text: "Across classifiers" }),
      h("p", { class: "lede", text: "The committed tables, one section per research question. Nothing here ranks the classifiers: a gap the tests below do not support is a description of this split, not a difference." }))];

    if (data.rq1.accuracy.length) {
      const figure = accuracyFigure(data, null);
      const cleared = data.rq1.pairs.filter((pair) => pair.significant).length;
      const inside = data.in_comparison;
      const gap = {};
      for (const pair of data.rq1.pairs) {
        gap[`${pair.a}|${pair.b}`] = { value: pair.row_gap, pair };
        gap[`${pair.b}|${pair.a}`] = { value: missing(pair.row_gap) ? null : -pair.row_gap, pair };
      }
      const parts = [figure.node,
        caption("figure", `Test accuracy on ${data.n_test_rows} posts; the axis starts at ${figure.lo.toFixed(2)}, not at zero.`, data.rq1.sources.slice(0, 1))];
      if (data.rq1.pairs.length) {
        parts.push(h("div", { class: "scroll" }, h("table", { class: "matrix" },
          h("thead", null, h("tr", null, h("th", null, ""), inside.map((name) => h("th", { scope: "col", tip: name }, short(name))))),
          h("tbody", null, inside.map((row) => h("tr", null,
            h("th", { class: "side", scope: "row" }, `${short(row)} ${splitName(row).rest}`),
            inside.map((column) => {
              if (row === column) return h("td", { class: "self" }, "·");
              const cell = gap[`${row}|${column}`];
              if (!cell) return h("td", null, "");
              return h("td", {
                class: cell.pair.significant ? "clear" : null, tabindex: "0",
                tip: `${short(row)} minus ${short(column)}: ${signed(cell.value)} rows, ${cell.pair.n_discordant} discordant, Holm p ${pval(cell.pair.holm_p)}`,
              }, signed(cell.value));
            })))))),
        caption("table", `Gaps in correct test rows, row minus column. A filled cell clears Holm-corrected McNemar significance: ${cleared} of ${data.rq1.pairs.length} pairs do. Every other gap is not distinguishable at this sample size.`, data.rq1.sources.slice(1)));
      }
      if (data.rq1.pairs_outside.length) {
        parts.push(table(["Classifier", "Against", "Row gap", "McNemar p", "Holm p", "Clears Holm"], data.rq1.pairs_outside.map((pair) => [
          pair.a, pair.b, signed(pair.row_gap), pval(pair.mcnemar_p), pval(pair.holm_p), pair.significant ? "yes" : "no"])),
        caption("table", "Pairs with classifiers outside the comparison, Holm-corrected over these pairs only."));
      }
      kids.push(section("Accuracy, and which gaps are real", parts));
    }

    const shares = byOrder(data.rq2.shares).filter((row) => row.categories);
    if (shares.length) {
      const withSd = (c) => (missing(c.share) ? "not given" : `${pct(c.share)}${missing(c.std) ? "" : ` ± ${pct(c.std, 2)}`}`);
      kids.push(section("What each model relies on",
        h("div", { class: "fig" }, categoryLegend(),
          chartRows(shares, (row) => h("div", { class: "row" }, who(row), segments(row.categories, false), h("span")))),
        caption("figure", `Share of positive attribution toward EXTREMIST among each classifier's top ${shares[0].top_k} words, whole-model test run, by lexicon category.`, data.rq2.sources.slice(0, 1)),
        table(["Classifier", ...state.overview.categories.map((c) => c.label)], shares.map((row) => [
          row.technique, ...state.overview.categories.map((c) => withSd(row.categories.find((x) => x.key === c.key) || {}))])),
        caption("table", "Each share with its standard deviation across the classifier's whole-model runs. A difference counts only when it exceeds the pooled standard deviation of the two.", data.rq2.sources.slice(1))));
    }

    const stability = byOrder(data.rq3.stability);
    if (stability.length) {
      const bar = (value) => {
        const el = h("div", { class: "bar" });
        el.style.width = `${Math.max(0, Math.min(1, value || 0)) * 100}%`;
        return h("div", { class: "row" }, h("span", { class: "val", text: num(value) }), el, h("span"));
      };
      kids.push(section("How stable that reliance is across posts",
        table(["Classifier", "Runs", "Spearman of absolute attribution", "Top-50 Jaccard"], stability.map((row) => [
          row.technique + (row.in_comparison === false ? " (outside the comparison)" : ""),
          missing(row.n_runs) ? "not given" : String(row.n_runs), bar(row.spearman), bar(row.jaccard)])),
        caption("table", "Agreement between each classifier's whole-model runs, which explain different posts. Low agreement means claims about single words of that model are fragile.", data.rq3.sources)));
    }

    const fpr = byOrder(data.rq4.fpr);
    if (fpr.length) {
      const scale = fprScale(fpr.flatMap((row) => [row.with_identity, row.without_identity]));
      kids.push(section("False positives on posts that mention a community",
        h("div", { class: "fig" }, fprLegend(),
          chartRows(fpr, (row) => h("div", { class: "row" }, who(row),
            track(scale.lo, scale.hi, scale.step, [
              { value: row.without_identity, ring: true, tip: `${row.technique}, without an identity term: ${pct(row.without_identity)} flagged` },
              { value: row.with_identity, tip: `${row.technique}, with an identity term: ${pct(row.with_identity)} flagged` },
            ], [row.without_identity, row.with_identity]),
            h("span", { class: "val", text: missing(row.ratio) ? "" : `${num(row.ratio, 2)}×`, tip: "With a term, over without" }))),
          h("div", { class: "row", "aria-hidden": "true" }, h("span", { class: "who" }), axis(scale.lo, scale.hi, scale.step, (v) => pct(v, 0)), h("span"))),
        caption("figure", "Share of non-extremist test posts flagged in error, with and without an identity term, at each classifier's locked threshold. The figure at the right is the ratio.", data.rq4.sources),
        table(["Classifier", "Without a term", "With a term", "Ratio", "Fisher exact p", "Posts with a term"], fpr.map((row) => [
          row.technique, pct(row.without_identity), pct(row.with_identity), num(row.ratio, 2), pval(row.fisher_p),
          missing(row.n_with) ? "not given" : `${row.n_with} of ${row.n_negative}`])),
        caption("table", "The p values are unadjusted. With this few posts only large ratios are detectable, and a result that is not significant is not evidence that the effect is absent.")));
    }

    if (kids.length === 1) {
      kids.push(h("div", { class: "empty" }, h("p", null, h("strong", { text: "No cross-classifier table yet." })),
        h("p", { text: "They are written when a classifier is audited." })));
    }
    sheet.replaceChildren(...kids.filter(Boolean));
    markScrollers();
  }

  // ---- add your classifier ------------------------------------------------------------
  function renderAdd() {
    sheet.replaceChildren(document.getElementById("tpl-add").content.cloneNode(true));
    const select = sheet.querySelector("#role");
    const outline = sheet.querySelector("#template-outline");
    const load = async () => {
      try {
        const data = await api(`/api/template?role=${encodeURIComponent(select.value)}`);
        outline.textContent = `$ ${data.command}\n\n${data.outline.trimEnd()}`;
      } catch (error) {
        outline.textContent = `The outline could not be loaded: ${error.message}`;
      }
    };
    for (const role of state.overview.notebook_roles) select.append(h("option", { value: role, text: role }));
    select.addEventListener("change", load);
    load();
    markScrollers();
  }

  // ---- runs -------------------------------------------------------------------------
  function replaceJobPanel() {
    const current = document.getElementById("job");
    if (current) current.replaceWith(jobPanel());
    markScrollers();
  }

  async function pollJob() {
    clearTimeout(state.timer);
    try {
      state.job = (await api("/api/job")).job;
    } catch (_) {
      return; // the server stopped; the next action reports it
    }
    if (state.job && state.job.state === "running") {
      if (state.view === "technique") {
        // The first sight of a run redraws the sheet, so the button and the numbering follow it.
        if (document.querySelector("#job .stages")) replaceJobPanel(); else renderTechnique();
      }
      state.timer = setTimeout(pollJob, 800);
    } else if (state.job) {
      await refresh(); // the run may have rewritten the card and the tables
    }
  }

  async function startRun(name) {
    try {
      state.job = (await api("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-Audit-Token": TOKEN },
        body: JSON.stringify({ technique: name }),
      })).job;
      renderTechnique();
      const panel = document.getElementById("job");
      if (panel) panel.scrollIntoView({ block: "nearest" });
      state.timer = setTimeout(pollJob, 500);
    } catch (error) {
      const current = document.getElementById("job");
      const message = h("p", { class: "error", id: "job", text: `The audit did not start: ${error.message}` });
      if (current) current.replaceWith(message);
    }
  }

  // ---- navigation -------------------------------------------------------------------
  async function loadTechnique() {
    const request = ++state.request;
    const query = `/api/technique?name=${encodeURIComponent(state.name)}${verbatimBox.checked ? "&verbatim=1" : ""}`;
    const [data, compare] = await Promise.all([api(query), api("/api/compare")]);
    if (request !== state.request) return; // a newer selection overtook this one
    state.technique = data;
    state.compare = compare;
    renderTechnique();
  }

  function selectTechnique(name) {
    location.hash = `technique=${encodeURIComponent(name)}`;
  }

  async function refresh() {
    try {
      state.overview = await api("/api/overview");
      const names = state.overview.techniques.map((t) => t.name);
      if (!names.includes(state.name)) state.name = names[0] || null;
      for (const button of document.querySelectorAll(".views button")) {
        if (button.dataset.view === state.view) button.setAttribute("aria-current", "page");
        else button.removeAttribute("aria-current");
      }
      renderRail();
      if (state.view === "compare") { renderCompare(await api("/api/compare")); return; }
      if (state.view === "add") { renderAdd(); return; }
      if (!state.name) {
        sheet.replaceChildren(h("div", { class: "empty" },
          h("p", null, h("strong", { text: "No classifier on file yet." })),
          h("p", null, "Nothing is under ", code(`${state.overview.results_dir}/`), ". ", h("a", { href: "#add", text: "Add your classifier" }), ".")));
        return;
      }
      await loadTechnique();
    } catch (error) {
      showError(error);
    }
  }

  function readHash() {
    const hash = decodeURIComponent(location.hash.replace(/^#/, ""));
    if (hash.startsWith("note-")) return false; // a footnote link, not a view
    if (hash === "compare" || hash === "add") { state.view = hash; return true; }
    state.view = "technique";
    if (hash.startsWith("technique=")) state.name = hash.slice("technique=".length);
    return true;
  }

  for (const button of document.querySelectorAll(".views button")) {
    button.addEventListener("click", () => {
      const view = button.dataset.view;
      const target = view === "technique" ? (state.name ? `technique=${encodeURIComponent(state.name)}` : "") : view;
      if (decodeURIComponent(location.hash.replace(/^#/, "")) === decodeURIComponent(target)) { state.view = view; refresh(); }
      else location.hash = target;
    });
  }
  verbatimBox.addEventListener("change", () => { if (state.view === "technique" && state.name) loadTechnique().catch(showError); });
  window.addEventListener("hashchange", async () => {
    if (!readHash()) return;
    await refresh();
    sheet.focus({ preventScroll: true });
    window.scrollTo(0, 0);
  });

  void layout;
  wide.addEventListener("change", () => { if (state.overview) renderRail(); });
  readHash();
  refresh().then(pollJob);
})();
