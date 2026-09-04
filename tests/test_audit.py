"""The HTML audit view.

The view is the thing a reviewer actually looks at, so the tests are about it
telling the truth: correct verdicts, honest treatment of partial matches, no
unescaped record content, and no phoning home.
"""

from __future__ import annotations

import re

from conftest import make_bank, make_invoice, make_link, make_payment, make_sources, make_truth

from recon.eval.audit import render_audit
from recon.eval.breakdown import build_breakdown
from recon.eval.scoring import score
from recon.matcher.pipeline import run_layered
from recon.matcher.types import (
    ExceptionReason,
    LinkType,
    Match,
    MatchLayer,
    ReconException,
    ReconResult,
    SubjectType,
)


def _render(result: ReconResult, truth, sources=None, **kwargs) -> str:
    sources = sources or make_sources()
    card = score(result, truth)
    return render_audit(
        sources,
        result,
        truth,
        card,
        build_breakdown(result, truth),
        split="test",
        config_hash="deadbeef",
        **kwargs,
    )


def _match(payment_id, counterparts, link_type=LinkType.PAYMENT_TO_BANK, rule="r"):
    return Match.build(
        link_type=link_type,
        payment_id=payment_id,
        counterpart_ids=counterparts,
        layer=MatchLayer.L2_FUZZY,
        rule=rule,
        confidence=0.95,
        evidence=["because the narration carried a prefix of the UTR"],
        source_records={"gateway": [payment_id], "bank": counterparts},
    )


def _verdicts(html: str) -> list[tuple[str, str]]:
    return re.findall(r'data-layer="([a-z0-9_]+)" data-verdict="(\w+)"', html)


# --- structure --------------------------------------------------------------
def test_the_page_is_a_complete_standalone_document() -> None:
    html = _render(ReconResult(matcher="m", split="test", matches=[], exceptions=[]),
                   make_truth([make_link()]))
    assert html.startswith("<!doctype html>")
    assert "<title>" in html and html.rstrip().endswith("</html>")


def test_the_page_makes_no_network_requests() -> None:
    """No CDN, no fonts, no analytics -- it must open from the filesystem."""
    html = _render(
        run_layered(make_sources(), "test"), make_truth([make_link()])
    )
    assert "http://" not in html
    assert "https://" not in html
    assert "<script src" not in html and "<link rel=\"stylesheet\"" not in html


def test_the_header_states_the_corpus_it_scored() -> None:
    html = _render(run_layered(make_sources(), "test"), make_truth([make_link()]))
    assert "deadbeef" in html


# --- verdicts ---------------------------------------------------------------
def test_a_match_equalling_ground_truth_reads_correct() -> None:
    truth = make_truth([make_link(bank_txn_ids=["bank_1"])])
    html = _render(
        ReconResult(matcher="m", split="t", matches=[_match("pay_1", ["bank_1"])], exceptions=[]),
        truth,
    )
    assert ("l2_fuzzy", "correct") in _verdicts(html)


def test_an_incomplete_but_sound_match_reads_partial_not_wrong() -> None:
    """One leg of a split settlement tied. Nothing false was asserted."""
    truth = make_truth([make_link(bank_txn_ids=["bank_1", "bank_2"])])
    html = _render(
        ReconResult(matcher="m", split="t", matches=[_match("pay_1", ["bank_1"])], exceptions=[]),
        truth,
    )
    assert ("l2_fuzzy", "partial") in _verdicts(html)
    assert "still untied: bank_2" in html


def test_a_match_containing_a_false_edge_reads_wrong() -> None:
    truth = make_truth([make_link(bank_txn_ids=["bank_1"])])
    html = _render(
        ReconResult(
            matcher="m", split="t", matches=[_match("pay_1", ["bank_9"])], exceptions=[]
        ),
        truth,
    )
    assert ("l2_fuzzy", "incorrect") in _verdicts(html)


def test_matching_a_payment_with_no_counterpart_reads_wrong() -> None:
    truth = make_truth([make_link(bank_txn_ids=[], unresolvable="never_credited")])
    html = _render(
        ReconResult(matcher="m", split="t", matches=[_match("pay_1", ["bank_1"])], exceptions=[]),
        truth,
    )
    assert ("l2_fuzzy", "incorrect") in _verdicts(html)
    assert "no counterpart at all" in html


def test_verdicts_are_absent_without_ground_truth() -> None:
    truth = make_truth([make_link()])
    html = _render(
        ReconResult(matcher="m", split="t", matches=[_match("pay_1", ["bank_1"])], exceptions=[]),
        truth,
        with_truth=False,
    )
    assert all(v == "na" for _, v in _verdicts(html))
    assert "without ground truth" in html


# --- exceptions -------------------------------------------------------------
def _exception(subject_id, reason=ExceptionReason.NO_CANDIDATE, subject=SubjectType.PAYMENT):
    return ReconException.build(
        subject_type=subject,
        subject_id=subject_id,
        link_type=LinkType.PAYMENT_TO_INVOICE,
        reason=reason,
        detail="nothing matched",
        layer_reached=MatchLayer.L2_FUZZY,
        evidence=["considered 0 candidates"],
    )


def test_an_exception_on_an_unresolvable_record_reads_rightly_open() -> None:
    truth = make_truth([make_link(invoice_id=None, unresolvable="no_erp")])
    html = _render(
        ReconResult(matcher="m", split="t", matches=[], exceptions=[_exception("pay_1")]), truth
    )
    assert ("exception", "correct") in _verdicts(html)
    assert "rightly open" in html


def test_an_exception_hiding_a_findable_answer_reads_missed() -> None:
    truth = make_truth([make_link()])
    html = _render(
        ReconResult(matcher="m", split="t", matches=[], exceptions=[_exception("pay_1")]), truth
    )
    assert ("exception", "incorrect") in _verdicts(html)
    assert "missed" in html


# --- escaping ---------------------------------------------------------------
def test_record_content_cannot_inject_markup() -> None:
    """Narrations are untrusted text as far as the page is concerned."""
    nasty = "<script>alert('x')</script>"
    bank = make_bank().model_copy(update={"narration": nasty})
    sources = make_sources(bank=[bank])
    truth = make_truth([make_link(bank_txn_ids=["bank_1"])])
    html = _render(
        ReconResult(matcher="m", split="t", matches=[_match("pay_1", ["bank_1"])], exceptions=[]),
        truth,
        sources=sources,
    )
    assert nasty not in html
    assert "&lt;script&gt;" in html


def test_a_customer_name_with_an_ampersand_survives_intact() -> None:
    invoice = make_invoice().model_copy(update={"customer_name": "Ganga & Co"})
    sources = make_sources(invoices=[invoice])
    truth = make_truth([make_link()])
    html = _render(
        ReconResult(
            matcher="m",
            split="t",
            matches=[_match("pay_1", ["INV-2026-00001"], LinkType.PAYMENT_TO_INVOICE)],
            exceptions=[],
        ),
        truth,
        sources=sources,
    )
    assert "Ganga &amp; Co" in html


# --- end to end -------------------------------------------------------------
def test_the_view_renders_a_real_pipeline_run() -> None:
    payments = [make_payment(f"pay_{i}", order_id=f"order_{i}") for i in range(3)]
    invoices = [
        make_invoice(f"INV-2026-0000{i}", order_id=f"order_{i}") for i in range(3)
    ]
    sources = make_sources(payments=payments, invoices=invoices)
    truth = make_truth(
        [make_link(f"pay_{i}", invoice_id=f"INV-2026-0000{i}") for i in range(3)]
    )
    html = _render(run_layered(sources, "test"), truth, sources=sources)
    for payment in payments:
        assert payment.payment_id in html
    assert "Evidence" in html and "Exception queue" in html
