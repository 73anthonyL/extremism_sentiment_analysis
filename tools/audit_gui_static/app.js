/* Audit workbench: the page's behaviour. Everything shown is read from the
   server's JSON, which reads committed files only. Text from those files is
   always placed with textContent, never as markup. */
(() => {
  "use strict";

  const TOKEN = document.querySelector('meta[name="audit-token"]').content;
  const sheet = document.getElementById("sheet");
  const rail = document.getElementById("rail");
  const layout = document.getElementById("layout");
  const tip = document.getElementById("tip");
  const verbatimBox = document.getElementById("verbatim");

  const state = {
    view: "technique", name: null, overview: null, technique: null, job: null,
    openItem: null, timer: null, request: 0,
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

  const missing = (v) => v === null || v === undefined || Number.isNaN(v);
  const num = (v, digits = 3) => (missing(v) ? "not given" : Number(v).toFixed(digits));
  const pct = (v, digits = 1) => (missing(v) ? "not given" : `${(Number(v) * 100).toFixed(digits)}%`);
  const pval = (v) => (missing(v) ? "not given" : v < 0.001 ? Number(v).toExponential(1) : Number(v).toFixed(3));
  const signed = (v) => (missing(v) ? "not given" : `${v > 0 ? "+" : ""}${v}`);
  const code = (text) => h("code", { text });

  async function api(path, options) {
    const response = await fetch(path, options);
    let body = null;
    try { body = await response.json(); } catch (_) { /* a non-JSON answer is reported below */ }
    if (!response.ok) throw new Error((body && body.error) || `the server answered ${response.status}`);
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

  function mark(kind, symbol, word) {
    return h("span", { class: `mark ${kind}` }, h("span", { "aria-hidden": "true", text: symbol }), h("span", { class: "sr", text: word }));
  }

  // A column is right-aligned, header included, when every filled cell in it is a number.
  const NUMERIC = /^[+\-]?\d[\d.,]*(?:e[+\-]?\d+)?[%×]?(?: \(\d+\))?$/i;
  function cellText(cell) {
    if (cell === null || cell === undefined) return "";
    if (Array.isArray(cell)) return cell.map(cellText).join("").trim();
    if (cell.nodeType) return cell.classList && cell.classList.contains("row") ? "bar" : cell.textContent.trim();
    return String(cell).trim();
  }
  function table(head, rows, className) {
    const numeric = head.map((_, i) => {
      const texts = rows.map((row) => cellText(row[i]));
      return texts.some((text) => text !== "") && texts.every((text) => text === "" || NUMERIC.test(text));
    });
    return h("div", { class: "scroll" },
      h("table", { class: className || null },
        h("thead", null, h("tr", null, head.map((cell, i) => h("th", { scope: "col", class: numeric[i] ? "n" : null }, cell)))),
        h("tbody", null, rows.map((row) => h("tr", null, row.map((cell, i) =>
          h("td", { class: numeric[i] ? "n" : null }, cell)))))));
  }

  function sources(list) {
    if (!list || !list.length) return null;
    const kids = ["Read from "];
    list.forEach((path, i) => { if (i) kids.push(i === list.length - 1 ? " and " : ", "); kids.push(code(path)); });
    kids.push(".");
    return h("p", { class: "source" }, kids);
  }

  function showError(error) {
    sheet.replaceChildren(h("div", { class: "error" },
      h("p", null, h("strong", { text: "This could not be loaded. " }), String(error.message || error)),
      h("p", { class: "note", text: "Check that the server is still running in its terminal, then reload the page." })));
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
      const dot = h("span", { class: `pt${point.ring ? " ring" : ""}`, tabindex: "0", tip: point.tip });
      dot.style.left = at(point.value);
      el.append(dot);
    }
    return el;
  }

  function who(row) {
    const parts = splitName(row.technique);
    return h("span", { class: "who" }, `${parts.number} ${parts.rest}`.trim(),
      row.in_comparison === false ? h("small", { text: "outside the comparison" }) : null);
  }

  function chartRows(rows, build) {
    let dividerDone = false;
    return h("div", { class: "rows" }, rows.map((row) => {
      const el = build(row);
      if (row.in_comparison === false && !dividerDone) { el.classList.add("divide"); dividerDone = true; }
      return el;
    }));
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

  function renderMarkdown(markdown) {
    const lines = markdown.split("\n");
    const nodes = [];
    const cells = (line) => line.trim().replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim());
    const isRow = (line) => line.trim().startsWith("|");
    const isItem = (line) => /^\s*[*-] /.test(line);
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (!line.trim() || line.startsWith("# ")) { i += 1; continue; } // the sheet already carries the title
      if (line.startsWith("## ")) { nodes.push(h("h3", null, inline(line.slice(3)))); i += 1; continue; }
      if (line.startsWith("### ")) { nodes.push(h("h4", null, inline(line.slice(4)))); i += 1; continue; }
      if (isRow(line) && i + 1 < lines.length && /^\s*\|[\s:|-]+\|\s*$/.test(lines[i + 1])) {
        const head = cells(line).map(inline);
        const body = [];
        i += 2;
        while (i < lines.length && isRow(lines[i])) { body.push(cells(lines[i]).map(inline)); i += 1; }
        nodes.push(table(head, body));
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
        class: "tech", type: "button", "aria-current": t.name === state.name ? "true" : null,
        onclick: () => selectTechnique(t.name),
      },
      h("span", { class: "num", text: parts.number }),
      h("span", { class: "nm", text: parts.rest }),
      t.contract_met ? mark("ok", "✓", "contract met") : mark("bad", "✕", "contract not met")));
    };
    const inside = techniques.filter((t) => t.in_comparison);
    const outside = techniques.filter((t) => !t.in_comparison);
    const kids = [];
    if (inside.length) kids.push(h("div", null, h("h2", { text: "In the paper's comparison" }), h("ul", null, inside.map(item))));
    if (outside.length) kids.push(h("div", { class: "group" }, h("h2", { text: "Outside the comparison" }), h("ul", null, outside.map(item))));
    kids.push(h("p", { class: "hint" }, "A technique is listed once its folder exists under ", code(`${state.overview.results_dir}/`),
      ". ", h("a", { href: "#add", text: "How to add yours" }), "."));
    rail.replaceChildren(...kids);
  }

  // ---- one technique -------------------------------------------------------------
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

  function itemState(item) {
    if (item.status === "OK") return { kind: "ok", symbol: "✓", word: "met" };
    if (item.required) return { kind: "bad", symbol: "✕", word: item.status === "MISSING" ? "missing" : "invalid" };
    return { kind: "none", symbol: "–", word: item.status === "MISSING" ? "not provided, optional" : "not available, optional" };
  }

  function contractSection(t) {
    const failed = t.contract.filter((item) => item.required && item.status !== "OK");
    const noRuns = failed.some((item) => item.name.startsWith("whole-model attribution"));
    const verdict = t.contract_met
      ? "Every required item is in place, so this technique can be audited."
      : `${failed.length} required item${failed.length === 1 ? " is" : "s are"} missing or invalid.${noRuns
        ? " Without word-level attribution runs for test and validation it cannot take part in RQ2 to RQ4, whatever its accuracy." : ""}`;
    const open = state.openItem === null ? null : t.contract[state.openItem];
    return h("section", { class: "contract", "aria-label": "Contract" },
      h("div", { class: "verdict" },
        h("strong", { text: t.contract_met ? "Contract met" : "Contract not met" }),
        h("span", { class: "note", text: verdict })),
      h("div", { class: "stamps" }, t.contract.map((item, index) => {
        const s = itemState(item);
        return h("button", {
          class: "stamp", type: "button", "aria-expanded": state.openItem === index ? "true" : "false",
          onclick: () => { state.openItem = state.openItem === index ? null : index; renderTechnique(); },
        }, mark(s.kind, s.symbol, s.word),
        h("span", null, h("span", { class: "what", text: itemName(item.name) }), h("br"), h("span", { class: "state", text: s.word })));
      })),
      open ? h("p", { class: "detail" }, h("b", { text: `${open.name}${open.required ? "" : " (optional)"}. ` }), open.detail)
        : h("p", { class: "note", text: "Select an item to read what the validator found." }));
  }

  function facts(t) {
    const rows = [];
    const add = (label, value) => { if (value !== null && value !== undefined && value !== "") rows.push(h("div", null, h("dt", { text: label }), h("dd", { text: value }))); };
    add("Model", t.config.model_family);
    add("Features", t.config.feature_family);
    if (!missing(t.config.selected_threshold)) add("Threshold, locked on validation", String(t.config.selected_threshold));
    if (!missing(t.metrics.accuracy)) add("Test accuracy", `${num(t.metrics.accuracy)} on ${state.overview.split.test} rows`);
    if (!missing(t.metrics.balanced_accuracy)) add("Balanced accuracy", num(t.metrics.balanced_accuracy));
    if (!missing(t.metrics.f1_macro)) add("Macro F1", num(t.metrics.f1_macro));
    if (!missing(t.metrics.roc_auc)) add("ROC-AUC", num(t.metrics.roc_auc));
    if (!missing(t.metrics.pr_auc)) add("PR-AUC", num(t.metrics.pr_auc));
    return rows.length ? h("dl", { class: "facts" }, rows) : null;
  }

  function stageState(status) {
    if (status === "ok") return { kind: "ok", symbol: "✓", word: "done" };
    if (status === "fail") return { kind: "bad", symbol: "✕", word: "failed" };
    if (status === "skip") return { kind: "none", symbol: "–", word: "skipped" };
    return { kind: "pending", symbol: "", word: "not reached" };
  }

  function jobPanel() {
    const job = state.job;
    if (!job) return h("div", { id: "job" });
    const running = job.state === "running";
    const kids = [h("h3", { text: running ? `Auditing ${job.technique}` : `Audit of ${job.technique}` }),
      h("p", { class: "note", role: "status", text: running ? "Running. Each stage appears as it finishes." : job.outcome })];
    if (job.stages.stopped_at === "check-contract") {
      kids.push(h("p", { class: "detail", text: `Stopped at the contract check. ${job.stages.stopped_message || "A required item is missing or invalid."}` }));
    }
    kids.push(h("ul", { class: "stages" }, job.stages.stages.map((stage) => {
      const s = stageState(stage.status);
      return h("li", null, mark(s.kind, s.symbol, s.word), h("span", { class: "stage-name", text: stage.name }),
        h("span", { class: "stage-msg", text: stage.message || (stage.status === "pending" ? "" : s.word) }));
    })));
    if (!running && Array.isArray(job.changed)) {
      kids.push(h("h3", { text: "Files this run changed" }));
      kids.push(job.changed.length
        ? h("ul", { class: "changed" }, job.changed.map((file) => h("li", null, code(file.state), " ", file.path)))
        : h("p", { class: "note", text: "None. The chain reproduced every table and card byte for byte." }));
    }
    kids.push(h("details", null, h("summary", { text: "Full output" }),
      h("pre", { class: "log", text: job.lines.join("\n") + (job.truncated ? "\n(the output was cut here)" : "") })));
    return h("div", { class: "run", id: "job" }, kids);
  }

  function runSection(t) {
    const running = state.job && state.job.state === "running";
    return h("section", null,
      h("h2", { text: "Audit" }),
      h("p", { class: "note", text: t.contract_met
        ? "Runs the documented tool chain for this technique, then rewrites its tables and report card. On the committed state the result is byte-identical."
        : "The audit runs once the contract is met." }),
      h("div", { class: "actions" },
        h("button", { class: "btn primary", type: "button", disabled: !t.contract_met || running, onclick: () => startRun(t.name) },
          running ? "Audit running" : "Run the audit"),
        h("button", { class: "btn", type: "button", onclick: () => refresh() }, "Check the contract again")),
      h("details", null, h("summary", { text: "The same on the command line" }),
        h("pre", { class: "cmd", text: `${t.commands.check}\n${t.commands.run}` }),
        h("p", { class: "note" }, "The identity counterfactual and the deletion faithfulness test load the model's weights, so this page does not start them: ",
          code(t.commands.perturbation))),
      jobPanel());
  }

  function relianceFigure(t) {
    const s = t.shares;
    return h("section", { class: "figure" },
      h("h2", { text: "What it relies on" }),
      h("p", { class: "note", text: `Share of positive attribution toward EXTREMIST among the top ${s.top_k} words of the whole-model ${s.split} run, by lexicon category.` }),
      categoryLegend(),
      segments(s.categories, true),
      table(["Category", "Share", "SD across runs", `Words in the top ${s.top_k}`], s.categories.map((c) => [
        h("span", { class: "cat" }, h("span", { class: `key c-${c.key}` }), categoryLabel(c.key)),
        pct(c.share), missing(c.std) ? "one run" : pct(c.std, 2), missing(c.n_words) ? "not given" : String(c.n_words)])),
      h("p", { class: "note", text: "A difference in share between two techniques counts as a difference only when it exceeds the pooled across-run standard deviation of the two." }));
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

  function fprFigure(t) {
    const r = t.card.rq4;
    const scale = fprScale([r.fpr_with_identity_terms, r.fpr_without_identity_terms]);
    return h("section", { class: "figure" },
      h("h2", { text: "False positives on posts that mention a community" }),
      h("p", { class: "note", text: `Non-extremist test posts flagged at the locked threshold ${r.threshold}. ${r.n_negative_with_identity_terms} of the ${r.n_negative} contain an identity term.` }),
      fprLegend(),
      track(scale.lo, scale.hi, scale.step, [
        { value: r.fpr_without_identity_terms, ring: true, tip: `Without an identity term: ${pct(r.fpr_without_identity_terms)} flagged` },
        { value: r.fpr_with_identity_terms, tip: `With an identity term: ${pct(r.fpr_with_identity_terms)} flagged` },
      ], [r.fpr_without_identity_terms, r.fpr_with_identity_terms]),
      axis(scale.lo, scale.hi, scale.step, (v) => pct(v, 0)),
      table(["Without a term", "With a term", "Ratio", "Fisher exact p"], [[
        pct(r.fpr_without_identity_terms), pct(r.fpr_with_identity_terms), num(r.fpr_ratio, 2), pval(r.fisher_p_value)]]),
      h("p", { class: "note", text: "A correlation over few posts. A result that is not significant is not evidence that the effect is absent." }));
  }

  function cardSection(t) {
    if (!t.card_markdown) {
      return h("section", { class: "section-rule" }, h("h2", { text: "Report card" }),
        h("div", { class: "empty" },
          h("p", null, h("strong", { text: "No report card yet." })),
          h("p", { text: t.contract_met ? "Run the audit above to write it." : "It is written by the audit, which runs once the contract is met." })));
    }
    const maskNote = t.slurs_masked
      ? ` ${t.slurs_masked} word${t.slurs_masked === 1 ? "" : "s"} the lexicon marks as a slur ${t.slurs_masked === 1 ? "is" : "are"} masked here; the file keeps ${t.slurs_masked === 1 ? "it" : "them"}.`
      : "";
    return h("section", { class: "section-rule" },
      h("h2", { text: "Report card" }),
      h("p", { class: "source" }, "Read from ", code(t.card_source), ", written by ", code("audit.py report"), ".", maskNote),
      h("div", { class: "card" }, renderMarkdown(t.card_markdown)));
  }

  function renderTechnique() {
    const t = state.technique;
    if (!t) return;
    const kids = [
      h("header", { class: "head" },
        h("h1", { class: "title", text: t.name }),
        h("p", { class: `standing${t.in_comparison ? " inside" : ""}`, text: t.in_comparison
          ? "In the paper's comparison"
          : "Outside the paper's comparison: it has its own Holm family, and no table in the paper changes" }),
        facts(t)),
      contractSection(t),
      runSection(t),
    ];
    if (t.shares && t.shares.categories) kids.push(relianceFigure(t));
    if (t.card && t.card.rq4) kids.push(fprFigure(t));
    kids.push(cardSection(t));
    sheet.replaceChildren(...kids);
  }

  // ---- across techniques -----------------------------------------------------------
  function renderCompare(data) {
    const order = [...data.in_comparison, ...data.outside_comparison];
    const byOrder = (rows) => [...rows].sort((a, b) => order.indexOf(a.technique) - order.indexOf(b.technique));
    const short = (name) => splitName(name).number || name;
    const kids = [h("header", { class: "head" },
      h("h1", { class: "title", text: "Across techniques" }),
      h("p", { class: "lede", text: "The committed tables, one view per research question. Nothing here ranks the techniques: a gap that the tests below do not support is a description of this split, not a difference." }))];

    // RQ1
    const accuracy = byOrder(data.rq1.accuracy);
    if (accuracy.length) {
      const values = accuracy.map((row) => row.accuracy);
      const lo = Math.floor((Math.min(...values) - 0.005) * 20) / 20;
      const hi = Math.ceil((Math.max(...values) + 0.005) * 20) / 20;
      const cleared = data.rq1.pairs.filter((pair) => pair.significant).length;
      const inside = data.in_comparison;
      const gap = {};
      for (const pair of data.rq1.pairs) {
        gap[`${pair.a}|${pair.b}`] = { value: pair.row_gap, pair };
        gap[`${pair.b}|${pair.a}`] = { value: missing(pair.row_gap) ? null : -pair.row_gap, pair };
      }
      const matrix = h("div", { class: "scroll" }, h("table", { class: "matrix" },
        h("thead", null, h("tr", null, h("th", null, ""), inside.map((name) => h("th", { scope: "col", tip: name }, short(name))))),
        h("tbody", null, inside.map((row) => h("tr", null,
          h("th", { class: "side", scope: "row" }, `${short(row)} ${splitName(row).rest}`),
          inside.map((column) => {
            if (row === column) return h("td", { class: "self" }, "");
            const cell = gap[`${row}|${column}`];
            if (!cell) return h("td", null, "");
            return h("td", {
              class: cell.pair.significant ? "clear" : null, tabindex: "0",
              tip: `${short(row)} minus ${short(column)}: ${signed(cell.value)} rows, ${cell.pair.n_discordant} discordant, Holm p ${pval(cell.pair.holm_p)}`,
            }, signed(cell.value));
          }))))));
      const section = h("section", { class: "section-rule" },
        h("h2", { text: "RQ1. Accuracy, and which gaps are real" }),
        h("p", { class: "note", text: data.rq1.pairs.length
          ? `${cleared} of the ${data.rq1.pairs.length} pairs in the comparison clear Holm-corrected McNemar significance on the ${data.n_test_rows} test rows. Every other gap is not distinguishable at this sample size.`
          : "No pair table has been written yet, so no gap can be read as a difference." }),
        h("div", { class: "figure" },
          h("h3", { text: "Test accuracy" }),
          chartRows(accuracy, (row) => h("div", { class: "row" }, who(row),
            track(lo, hi, 0.05, [{ value: row.accuracy, tip: `${row.technique}: ${num(row.accuracy)}` }]),
            h("span", { class: "val", text: num(row.accuracy) }))),
          h("div", { class: "row", "aria-hidden": "true" }, h("span"), axis(lo, hi, 0.05, (v) => v.toFixed(2)), h("span", { class: "val" })),
          h("p", { class: "note", text: `The axis starts at ${lo.toFixed(2)}, not at zero.` })),
        data.rq1.pairs.length ? h("div", { class: "figure" },
          h("h3", { text: "Row gaps between pairs" }),
          matrix,
          h("p", { class: "note", text: "Each cell is the row technique's correct test rows minus the column's. A filled cell clears Holm-corrected significance; an unfilled one does not." })) : null,
        data.rq1.pairs_outside.length ? h("div", { class: "figure" },
          h("h3", { text: "Pairs with techniques outside the comparison" }),
          table(["Technique", "Against", "Row gap", "McNemar p", "Holm p", "Clears Holm"], data.rq1.pairs_outside.map((pair) => [
            pair.a, pair.b, signed(pair.row_gap), pval(pair.mcnemar_p), pval(pair.holm_p), pair.significant ? "yes" : "no"])),
          h("p", { class: "note", text: "Holm-corrected over these pairs only, a family of their own." })) : null,
        sources(data.rq1.sources));
      kids.push(section);
    }

    // RQ2
    const shares = byOrder(data.rq2.shares).filter((row) => row.categories);
    if (shares.length) {
      const withSd = (c) => (missing(c.share) ? "not given" : `${pct(c.share)}${missing(c.std) ? "" : ` ± ${pct(c.std, 2)}`}`);
      kids.push(h("section", { class: "section-rule" },
        h("h2", { text: "RQ2. What each model relies on" }),
        h("p", { class: "note", text: `Share of positive attribution toward EXTREMIST among each technique's top ${shares[0].top_k} words, whole-model test run, by lexicon category.` }),
        h("div", { class: "figure" },
          categoryLegend(),
          chartRows(shares, (row) => h("div", { class: "row" }, who(row), segments(row.categories, false), h("span", { class: "val" }))),
          table(["Technique", ...state.overview.categories.map((c) => c.label)], shares.map((row) => [
            row.technique, ...state.overview.categories.map((c) => withSd(row.categories.find((x) => x.key === c.key) || {}))])),
          h("p", { class: "note", text: "Each share is given with its standard deviation across the technique's whole-model runs. A difference between two techniques counts only when it exceeds the pooled standard deviation of the two." })),
        sources(data.rq2.sources)));
    }

    // RQ3
    const stability = byOrder(data.rq3.stability);
    if (stability.length) {
      const bar = (value) => {
        const el = h("div", { class: "bar" });
        el.style.width = `${Math.max(0, Math.min(1, value || 0)) * 100}%`;
        return h("div", { class: "row" }, h("span", { class: "val", text: num(value) }), el, h("span"));
      };
      kids.push(h("section", { class: "section-rule" },
        h("h2", { text: "RQ3. How stable that reliance is across posts" }),
        h("p", { class: "note", text: "Agreement between each technique's whole-model runs, which explain different posts. Low agreement means claims about single words of that model are fragile." }),
        h("div", { class: "figure" },
          table(["Technique", "Runs", "Spearman of absolute attribution", "Top-50 Jaccard"], stability.map((row) => [
            row.technique + (row.in_comparison === false ? " (outside the comparison)" : ""),
            missing(row.n_runs) ? "not given" : String(row.n_runs), bar(row.spearman), bar(row.jaccard)]))),
        sources(data.rq3.sources)));
    }

    // RQ4
    const fpr = byOrder(data.rq4.fpr);
    if (fpr.length) {
      const scale = fprScale(fpr.flatMap((row) => [row.with_identity, row.without_identity]));
      kids.push(h("section", { class: "section-rule" },
        h("h2", { text: "RQ4. False positives on posts that mention a community" }),
        h("p", { class: "note", text: "False-positive rate on non-extremist test posts, with and without an identity term, at each technique's locked threshold." }),
        h("div", { class: "figure" },
          fprLegend(),
          chartRows(fpr, (row) => h("div", { class: "row" }, who(row),
            track(scale.lo, scale.hi, scale.step, [
              { value: row.without_identity, ring: true, tip: `${row.technique}, without an identity term: ${pct(row.without_identity)} flagged` },
              { value: row.with_identity, tip: `${row.technique}, with an identity term: ${pct(row.with_identity)} flagged` },
            ], [row.without_identity, row.with_identity]),
            h("span", { class: "val", text: missing(row.ratio) ? "" : `${num(row.ratio, 2)}×`, tip: "With a term, over without" }))),
          h("div", { class: "row", "aria-hidden": "true" }, h("span"), axis(scale.lo, scale.hi, scale.step, (v) => pct(v, 0)), h("span", { class: "val" })),
          table(["Technique", "Without a term", "With a term", "Ratio", "Fisher exact p", "Posts with a term"], fpr.map((row) => [
            row.technique, pct(row.without_identity), pct(row.with_identity), num(row.ratio, 2), pval(row.fisher_p),
            missing(row.n_with) ? "not given" : `${row.n_with} of ${row.n_negative}`])),
          h("p", { class: "note", text: "The p values are unadjusted. With this few posts only large ratios are detectable, and a result that is not significant is not evidence that the effect is absent." })),
        sources(data.rq4.sources)));
    }

    if (kids.length === 1) {
      kids.push(h("div", { class: "empty" }, h("p", null, h("strong", { text: "No cross-technique table yet." })),
        h("p", { text: "They are written when a technique is audited." })));
    }
    sheet.replaceChildren(...kids);
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
  }

  // ---- runs -------------------------------------------------------------------------
  function replaceJobPanel() {
    const current = document.getElementById("job");
    if (current) current.replaceWith(jobPanel());
  }

  async function pollJob() {
    clearTimeout(state.timer);
    try {
      state.job = (await api("/api/job")).job;
    } catch (_) {
      return; // the server stopped; the next action reports it
    }
    if (state.job && state.job.state === "running") {
      if (state.view === "technique") replaceJobPanel();
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
    const data = await api(query);
    if (request !== state.request) return; // a newer selection overtook this one
    state.technique = data;
    renderTechnique();
  }

  function selectTechnique(name) {
    state.openItem = null;
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
      layout.classList.toggle("wide", state.view !== "technique");
      if (state.view === "compare") { renderCompare(await api("/api/compare")); return; }
      if (state.view === "add") { renderAdd(); return; }
      renderRail();
      if (!state.name) {
        sheet.replaceChildren(h("div", { class: "empty" },
          h("p", null, h("strong", { text: "No technique folder yet." })),
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
    if (hash === "compare" || hash === "add") { state.view = hash; return; }
    state.view = "technique";
    if (hash.startsWith("technique=")) state.name = hash.slice("technique=".length);
  }

  for (const button of document.querySelectorAll(".views button")) {
    button.addEventListener("click", () => {
      const view = button.dataset.view;
      location.hash = view === "technique" ? (state.name ? `technique=${encodeURIComponent(state.name)}` : "") : view;
      if (view === "technique" && !state.name) { state.view = "technique"; refresh(); }
    });
  }
  verbatimBox.addEventListener("change", () => { if (state.view === "technique" && state.name) loadTechnique().catch(showError); });
  window.addEventListener("hashchange", async () => { readHash(); await refresh(); sheet.focus({ preventScroll: true }); window.scrollTo(0, 0); });

  readHash();
  refresh().then(pollJob);
})();
