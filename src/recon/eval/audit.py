"""A self-contained HTML audit trail: one page, no server, no network.

Every asserted match renders with the records that went into it, the layer and
rule that made it, the confidence, and the human-readable evidence.  Every
exception renders with the reason it was raised and what was considered.

Two deliberate constraints:

* **One file, no dependencies.**  CSS and JS are inlined and there are no CDN
  references, so the page opens from the filesystem, survives being emailed,
  and honours the project's no-network rule.  It is also what makes it
  screenshot-able without standing up a server first.
* **Verification is shown, not claimed.**  Rendered in evaluation mode, each
  row carries a verdict against ground truth.  A demo that shows matches
  without showing which ones are wrong is asking to be taken on trust, and this
  project's whole argument is that you should not have to.  The page states in
  its header that ground truth is present, because in production it would not
  be.
"""

from __future__ import annotations

import html
import subprocess
from datetime import UTC, datetime

from recon.eval.breakdown import Breakdown
from recon.eval.scoring import ScoreCard
from recon.matcher.types import LinkType, Match, ReconException, ReconResult, SubjectType
from recon.models import BankRow, GroundTruth, InvoiceRow, PaymentView, SourceBundle
from recon.money import format_rupees


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover
        return "unknown"


CSS = """
:root {
  --bg: #f6f7f9; --panel: #ffffff; --ink: #14161a; --muted: #5c6370;
  --line: #e2e5ea; --accent: #0b6bcb; --ok: #1a7f4b; --bad: #b3261e;
  --warn: #8a6100; --chip: #eef1f5;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 14px/1.5 ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif;
}
header {
  background: var(--panel); border-bottom: 1px solid var(--line);
  padding: 18px 28px; position: sticky; top: 0; z-index: 10;
}
h1 { margin: 0 0 4px; font-size: 19px; letter-spacing: -0.01em; }
.sub { color: var(--muted); font-size: 12.5px; }
.sub code { background: var(--chip); padding: 1px 5px; border-radius: 4px; }
main { padding: 24px 28px 64px; max-width: 1240px; margin: 0 auto; }
h2 { font-size: 15px; margin: 34px 0 12px; letter-spacing: -0.01em; }
h2:first-child { margin-top: 0; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
.tile {
  background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px;
}
.tile .v { font-size: 24px; font-weight: 600; letter-spacing: -0.02em; }
.tile .k { color: var(--muted); font-size: 11.5px; text-transform: uppercase;
  letter-spacing: .06em; margin-top: 3px; }
.tile .ci { color: var(--muted); font-size: 11.5px; }
.tile.good .v { color: var(--ok); } .tile.bad .v { color: var(--bad); }
.controls { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; margin: 0 0 12px; }
input[type=search] {
  flex: 1 1 280px; padding: 8px 11px; border: 1px solid var(--line);
  border-radius: 8px; background: var(--panel); font: inherit; color: inherit;
}
.chip {
  border: 1px solid var(--line); background: var(--panel); border-radius: 999px;
  padding: 5px 12px; font-size: 12.5px; cursor: pointer; color: var(--muted);
}
.chip[aria-pressed=true] { background: var(--accent); border-color: var(--accent); color: #fff; }
.row {
  background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
  margin-bottom: 8px; overflow: hidden;
}
.row[hidden] { display: none; }
summary {
  display: grid; grid-template-columns: 190px 96px 1fr 150px 78px 74px;
  gap: 12px; align-items: center; padding: 11px 15px; cursor: pointer; list-style: none;
}
summary::-webkit-details-marker { display: none; }
summary:hover { background: #fafbfc; }
.mono { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 12.5px; }
.leg { color: var(--muted); font-size: 12px; }
.rule { color: var(--muted); font-size: 12px; }
.conf { text-align: right; font-variant-numeric: tabular-nums; }
.badge {
  display: inline-block; padding: 2px 8px; border-radius: 5px; font-size: 11px;
  font-weight: 600; text-align: center; white-space: nowrap;
}
.b-l1 { background: #e6f0fb; color: #0b4f9c; }
.b-l2 { background: #eae7fb; color: #4b3ba8; }
.b-l3 { background: #fdeee0; color: #9a5410; }
.b-baseline { background: var(--chip); color: var(--muted); }
.v-ok { background: #e3f5ea; color: var(--ok); }
.v-part { background: #fdf3e0; color: var(--warn); }
.v-bad { background: #fdeaea; color: var(--bad); }
.v-na { background: var(--chip); color: var(--muted); }
.detail { border-top: 1px solid var(--line); padding: 16px 18px; background: #fbfcfd; }
.detail h4 {
  margin: 0 0 8px; font-size: 11.5px; text-transform: uppercase;
  letter-spacing: .06em; color: var(--muted);
}
.cols { display: grid; grid-template-columns: repeat(auto-fit, minmax(270px, 1fr)); gap: 20px; }
dl { margin: 0; display: grid; grid-template-columns: auto 1fr; gap: 3px 14px; }
dt { color: var(--muted); font-size: 12.5px; }
dd { margin: 0; font-size: 12.5px; }
ul.evidence { margin: 0; padding-left: 17px; }
ul.evidence li { margin-bottom: 5px; font-size: 12.5px; }
table { width: 100%; border-collapse: collapse; background: var(--panel);
  border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
th, td { text-align: left; padding: 9px 13px; border-bottom: 1px solid var(--line);
  font-size: 12.5px; }
th { color: var(--muted); font-weight: 600; font-size: 11.5px;
  text-transform: uppercase; letter-spacing: .05em; }
tr:last-child td { border-bottom: none; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.note { color: var(--muted); font-size: 12.5px; margin: 8px 0 0; }
.empty { color: var(--muted); padding: 18px; text-align: center; }
@media (max-width: 860px) { summary { grid-template-columns: 1fr; gap: 4px; } }
"""

JS = """
const q = document.getElementById('q');
const onlyWrong = document.getElementById('only-wrong');
const chips = [...document.querySelectorAll('.chip[data-layer]')];

function apply() {
  const term = q.value.trim().toLowerCase();
  const active = chips.filter(c => c.getAttribute('aria-pressed') === 'true')
                      .map(c => c.dataset.layer);
  const wrongOnly = onlyWrong.getAttribute('aria-pressed') === 'true';
  let shown = 0;
  for (const row of document.querySelectorAll('.row[data-searchable]')) {
    const okLayer = active.length === 0 || active.includes(row.dataset.layer);
    const okTerm = !term || row.dataset.searchable.includes(term);
    const okWrong = !wrongOnly || row.dataset.verdict === 'incorrect';
    const visible = okLayer && okTerm && okWrong;
    row.hidden = !visible;
    if (visible) shown++;
  }
  document.getElementById('count').textContent = shown;
}
function toggle(el) {
  el.setAttribute('aria-pressed', el.getAttribute('aria-pressed') === 'true' ? 'false' : 'true');
  apply();
}
q.addEventListener('input', apply);
onlyWrong.addEventListener('click', () => toggle(onlyWrong));
chips.forEach(c => c.addEventListener('click', () => toggle(c)));
apply();
"""


def _layer_class(layer: str) -> str:
    return {"l1_exact": "b-l1", "l2_fuzzy": "b-l2", "l3_llm": "b-l3"}.get(layer, "b-baseline")


def _expected(truth: GroundTruth, payment_id: str, link_type: LinkType) -> set[str]:
    link = next((x for x in truth.links if x.payment_id == payment_id), None)
    if link is None:
        return set()
    if link_type is LinkType.PAYMENT_TO_BANK:
        return set(link.bank_txn_ids)
    return {link.invoice_id} if link.invoice_id else set()


def _payment_dl(payment: PaymentView | None) -> str:
    if payment is None:
        return "<p class='note'>payment record not in this split</p>"
    captured = payment.captured_at.isoformat() if payment.captured_at else "—"
    rows = {
        "payment_id": f"<span class='mono'>{_esc(payment.payment_id)}</span>",
        "order_id": f"<span class='mono'>{_esc(payment.order_id or '— absent —')}</span>",
        "receipt": f"<span class='mono'>{_esc(payment.order_receipt or '—')}</span>",
        "gross": format_rupees(payment.gross_paise),
        "fee + GST + TDS": (
            f"{format_rupees(payment.fee_paise)} + {format_rupees(payment.tax_paise)}"
            f" + {format_rupees(payment.tds_paise)}"
        ),
        "net settled": format_rupees(payment.net_paise),
        "captured": captured,
        "settlement": ", ".join(payment.settlement_ids) or "—",
    }
    return "<dl>" + "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in rows.items()) + "</dl>"


def _bank_dl(row: BankRow) -> str:
    rows = {
        "txn_id": f"<span class='mono'>{_esc(row.txn_id)}</span>",
        "credit": format_rupees(row.credit_paise),
        "value date": row.value_date.isoformat(),
        "posted": row.posted_date.isoformat(),
        "utr column": f"<span class='mono'>{_esc(row.utr or '— null —')}</span>",
        "narration": f"<span class='mono'>{_esc(row.narration)}</span>",
    }
    return "<dl>" + "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in rows.items()) + "</dl>"


def _invoice_dl(row: InvoiceRow) -> str:
    rows = {
        "invoice_id": f"<span class='mono'>{_esc(row.invoice_id)}</span>",
        "amount": format_rupees(row.invoice_amount_paise),
        "customer": _esc(row.customer_name),
        "issued": row.issue_date.isoformat(),
        "due": row.due_date.isoformat(),
        "order_id": f"<span class='mono'>{_esc(row.order_id or '— absent —')}</span>",
        "status": _esc(row.status),
    }
    return "<dl>" + "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in rows.items()) + "</dl>"


def _match_row(
    match: Match,
    sources: SourceBundle,
    truth: GroundTruth,
    *,
    with_truth: bool,
) -> str:
    payments = {p.payment_id: p for p in sources.payments}
    bank = {r.txn_id: r for r in sources.bank_rows}
    invoices = {i.invoice_id: i for i in sources.invoices}

    verdict, verdict_class, verdict_label = "na", "v-na", "—"
    if with_truth:
        want = _expected(truth, match.payment_id, match.link_type)
        got = set(match.counterpart_ids)
        if got == want:
            verdict, verdict_class, verdict_label = "correct", "v-ok", "correct"
        elif got and got < want:
            # Every edge asserted is right, but the set is incomplete -- a split
            # settlement with one leg tied. Calling that "wrong" would contradict
            # the edge-level precision the scorecard reports, and would overstate
            # the error: nothing false was asserted.
            verdict, verdict_class, verdict_label = "partial", "v-part", "partial"
        else:
            verdict, verdict_class, verdict_label = "incorrect", "v-bad", "wrong"

    counterpart_cells = []
    for cid in match.counterpart_ids:
        if cid in bank:
            counterpart_cells.append(f"<div><h4>Bank credit</h4>{_bank_dl(bank[cid])}</div>")
        elif cid in invoices:
            counterpart_cells.append(f"<div><h4>Invoice</h4>{_invoice_dl(invoices[cid])}</div>")

    expected_block = ""
    if with_truth and verdict in {"incorrect", "partial"}:
        want = _expected(truth, match.payment_id, match.link_type)
        missing = sorted(want - set(match.counterpart_ids))
        expected_block = (
            "<div><h4>Ground truth</h4><p class='note mono'>expected "
            f"{_esc(', '.join(sorted(want)) or 'no counterpart at all')}"
            + (f"<br>still untied: {_esc(', '.join(missing))}" if missing else "")
            + "</p></div>"
        )

    evidence = "".join(f"<li>{_esc(line)}</li>" for line in match.evidence)
    sources_listed = "".join(
        f"<dt>{_esc(k)}</dt><dd class='mono'>{_esc(', '.join(v))}</dd>"
        for k, v in match.source_records.items()
    )
    leg = match.link_type.value.replace("payment_to_", "→ ")
    searchable = " ".join(
        [match.payment_id, *match.counterpart_ids, match.rule, match.match_id]
    ).lower()

    return f"""
<details class="row" data-layer="{_esc(match.layer.value)}" data-verdict="{verdict}"
         data-searchable="{_esc(searchable)}">
  <summary>
    <span class="mono">{_esc(match.payment_id)}</span>
    <span class="leg">{_esc(leg)}</span>
    <span class="mono">{_esc(', '.join(match.counterpart_ids))}</span>
    <span class="badge {_layer_class(match.layer.value)}">{_esc(match.layer.value)}</span>
    <span class="conf">{match.confidence:.2f}</span>
    <span class="badge {verdict_class}">{verdict_label}</span>
  </summary>
  <div class="detail">
    <div class="cols">
      <div><h4>Payment</h4>{_payment_dl(payments.get(match.payment_id))}</div>
      {''.join(counterpart_cells)}
      {expected_block}
      <div>
        <h4>How it was matched</h4>
        <dl>
          <dt>match_id</dt><dd class="mono">{_esc(match.match_id)}</dd>
          <dt>layer</dt><dd>{_esc(match.layer.value)}</dd>
          <dt>rule</dt><dd class="mono">{_esc(match.rule)}</dd>
          <dt>confidence</dt><dd>{match.confidence:.2f}</dd>
          {sources_listed}
        </dl>
      </div>
    </div>
    <h4 style="margin-top:16px">Evidence</h4>
    <ul class="evidence">{evidence}</ul>
  </div>
</details>"""


def _exception_row(
    exception: ReconException, truth: GroundTruth, *, with_truth: bool
) -> str:
    verdict, verdict_class, verdict_label = "na", "v-na", "—"
    if with_truth:
        correct = False
        if exception.subject_type is SubjectType.PAYMENT:
            link_type = exception.link_type or LinkType.PAYMENT_TO_BANK
            correct = not _expected(truth, exception.subject_id, link_type)
        elif exception.subject_type is SubjectType.BANK_TXN:
            correct = exception.subject_id in truth.orphan_bank_txn_ids
        else:
            correct = exception.subject_id in truth.orphan_invoice_ids
        verdict = "correct" if correct else "incorrect"
        verdict_class = "v-ok" if correct else "v-bad"
        verdict_label = "rightly open" if correct else "missed"

    evidence = "".join(f"<li>{_esc(line)}</li>" for line in exception.evidence) or (
        "<li class='note'>no evidence recorded</li>"
    )
    leg = exception.link_type.value.replace("payment_to_", "→ ") if exception.link_type else "—"
    searchable = " ".join([exception.subject_id, exception.reason.value]).lower()
    return f"""
<details class="row" data-layer="exception" data-verdict="{verdict}"
         data-searchable="{_esc(searchable)}">
  <summary>
    <span class="mono">{_esc(exception.subject_id)}</span>
    <span class="leg">{_esc(leg)}</span>
    <span class="rule">{_esc(exception.detail)}</span>
    <span class="badge b-baseline">{_esc(exception.reason.value)}</span>
    <span class="conf">{exception.candidates_considered or ''}</span>
    <span class="badge {verdict_class}">{verdict_label}</span>
  </summary>
  <div class="detail">
    <dl>
      <dt>subject</dt><dd class="mono">{_esc(exception.subject_id)}
        ({_esc(exception.subject_type.value)})</dd>
      <dt>reason</dt><dd class="mono">{_esc(exception.reason.value)}</dd>
      <dt>reached</dt><dd>{_esc(exception.layer_reached.value)}</dd>
      <dt>candidates</dt><dd>{exception.candidates_considered}</dd>
    </dl>
    <h4 style="margin-top:14px">What was considered</h4>
    <ul class="evidence">{evidence}</ul>
  </div>
</details>"""


def _tiles(card: ScoreCard) -> str:
    bank = card.link(LinkType.PAYMENT_TO_BANK)
    invoice = card.link(LinkType.PAYMENT_TO_INVOICE)
    false_matches = sum(h.falsely_matched for h in card.honesty)
    lo, hi = card.fully_reconciled_ci
    tiles = [
        (
            "good",
            f"{card.fully_reconciled_rate:.1%}",
            "fully reconciled",
            f"{card.fully_reconciled}/{card.n_payments} · 95% CI {lo:.0%}–{hi:.0%}",
        ),
        ("", f"{bank.precision:.1%}", "bank precision", f"recall {bank.recall:.1%}"),
        ("", f"{invoice.precision:.1%}", "invoice precision", f"recall {invoice.recall:.1%}"),
        (
            "bad" if false_matches else "good",
            str(false_matches),
            "false matches",
            "links asserted where none exists",
        ),
    ]
    return "".join(
        f"<div class='tile {cls}'><div class='v'>{_esc(value)}</div>"
        f"<div class='k'>{_esc(key)}</div><div class='ci'>{_esc(note)}</div></div>"
        for cls, value, key, note in tiles
    )


def _layer_table(breakdown: Breakdown) -> str:
    rows = "".join(
        f"<tr><td><span class='badge {_layer_class(r.layer)}'>{_esc(r.layer)}</span></td>"
        f"<td class='mono'>{_esc(r.rule)}</td>"
        f"<td>{_esc(r.link_type.replace('payment_to_', ''))}</td>"
        f"<td class='num'>{r.matches}</td><td class='num'>{r.correct}</td>"
        f"<td class='num'>{r.precision:.1%}</td></tr>"
        for r in breakdown.layers
    )
    return (
        "<table><thead><tr><th>layer</th><th>rule</th><th>leg</th>"
        "<th class='num'>edges</th><th class='num'>correct</th>"
        f"<th class='num'>precision</th></tr></thead><tbody>{rows}</tbody></table>"
    )


def render_audit(
    sources: SourceBundle,
    result: ReconResult,
    truth: GroundTruth,
    card: ScoreCard,
    breakdown: Breakdown,
    *,
    split: str,
    with_truth: bool = True,
    config_hash: str = "",
) -> str:
    matches = sorted(result.matches, key=lambda m: (m.layer.value, m.payment_id))
    exceptions = sorted(result.exceptions, key=lambda x: (x.reason.value, x.subject_id))

    truth_note = (
        "Rendered in <strong>evaluation mode</strong>: every row is checked against "
        "ground truth, so wrong matches are visible rather than implied. In production "
        "there is no ground truth and the verdict column would be absent."
        if with_truth
        else "Rendered without ground truth; no verdicts are shown."
    )
    layers_present = sorted({m.layer.value for m in matches})
    chips = "".join(
        f"<button class='chip' data-layer='{_esc(layer)}' aria-pressed='false'>"
        f"{_esc(layer)}</button>"
        for layer in layers_present
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reconciliation audit — {_esc(split)}</title>
<style>{CSS}</style></head>
<body>
<header>
  <h1>Reconciliation audit trail — {_esc(split)}</h1>
  <div class="sub">
    matcher <code>{_esc(result.matcher)}</code> ·
    {len(sources.payments)} payments · {len(sources.batches)} settlement batches ·
    corpus <code>{_esc(config_hash or 'unknown')}</code> ·
    commit <code>{_esc(_commit())}</code> ·
    generated {_esc(datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC'))}
  </div>
  <div class="sub" style="margin-top:6px">{truth_note}</div>
</header>
<main>
  <h2>Headline</h2>
  <div class="tiles">{_tiles(card)}</div>

  <h2>Which layer earned its keep</h2>
  {_layer_table(breakdown)}
  <p class="note">Precision is measured against ground truth, not asserted. A rule
  that fires often and is wrong shows up here.</p>

  <h2>Matches <span class="note">(<span id="count">0</span> shown)</span></h2>
  <p class="note"><span class="badge v-ok">correct</span> the asserted set equals
  ground truth &nbsp;·&nbsp; <span class="badge v-part">partial</span> every asserted
  edge is right but the set is incomplete, which happens when one leg of a split
  settlement ties and the other does not &nbsp;·&nbsp;
  <span class="badge v-bad">wrong</span> at least one asserted edge is not in
  ground truth.</p>
  <div class="controls">
    <input type="search" id="q" placeholder="Filter by payment, counterpart, or rule…">
    {chips}
    <button class="chip" id="only-wrong" aria-pressed="false">only wrong</button>
  </div>
  {''.join(_match_row(m, sources, truth, with_truth=with_truth) for m in matches)
   or "<p class='empty'>no matches</p>"}

  <h2>Exception queue</h2>
  <p class="note">An exception is the correct answer when the record genuinely has no
  counterpart. Rows marked <span class="badge v-bad">missed</span> are ones where an
  answer existed and was not found — they are shown here rather than hidden among the
  genuine orphans.</p>
  {''.join(_exception_row(x, truth, with_truth=with_truth) for x in exceptions)
   or "<p class='empty'>no exceptions</p>"}
</main>
<script>{JS}</script>
</body></html>
"""
