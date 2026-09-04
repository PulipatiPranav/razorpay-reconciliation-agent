"""Post-generation invariants.

The whole project rests on the ground-truth file being right.  These checks run
on every generation (``make data`` fails if any of them trips) so that a scoring
number can never be built on an incoherent universe.
"""

from __future__ import annotations

import json
from pathlib import Path

from recon.generator.generate import Universe
from recon.models import DefectTag, GroundTruth
from recon.money import net_of


def check_universe(universe: Universe) -> list[str]:
    """Internal coherence of the generated universe."""
    problems: list[str] = []
    by_id = {p.payment_id: p for p in universe.payments}

    for pay in universe.payments:
        if sum(amount for _, amount in pay.portions) != pay.credit:
            problems.append(f"{pay.payment_id}: settlement portions do not sum to credit")
        if net_of(pay.gross, pay.fee, pay.tax, pay.tds) != pay.credit:
            problems.append(f"{pay.payment_id}: gross - fee - tax - tds != credit")
        if pay.gross <= 0:
            problems.append(f"{pay.payment_id}: non-positive gross")

    for bundle in universe.bundles:
        expected = 0
        for pid in set(bundle.payment_ids):
            expected += sum(a for b, a in by_id[pid].portions if b == bundle.bundle_id)
        for refund in universe.refunds:
            if refund.bundle_id == bundle.bundle_id and not refund.is_phantom_duplicate:
                expected -= refund.amount
        for adj in universe.adjustments:
            if adj.bundle_id == bundle.bundle_id:
                expected -= adj.amount
        drift = bundle.credit_total - expected
        allowed = 5 if DefectTag.PAISE_DRIFT_BUNDLE in bundle.tags else 0
        if abs(drift) > allowed:
            problems.append(
                f"{bundle.bundle_id}: credit {bundle.credit_total} != expected {expected}"
            )
        if bundle.settled_date < bundle.last_capture:
            problems.append(f"{bundle.bundle_id}: settled before last capture")

    for bundle_id, row in universe.bank_rows.items():
        bundle = next(b for b in universe.bundles if b.bundle_id == bundle_id)
        if row.credit_paise != max(0, bundle.credit_total):
            problems.append(f"{bundle_id}: bank credit disagrees with settlement total")

    invoice_ids = {inv.invoice_id for inv in universe.invoices}
    for pay in universe.payments:
        if pay.invoice_id and pay.invoice_id not in invoice_ids:
            problems.append(f"{pay.payment_id}: points at a missing invoice")
        if DefectTag.NO_INVOICE in pay.tags and pay.invoice_id is not None:
            problems.append(f"{pay.payment_id}: tagged no-invoice but has one")
    return problems


def check_split_files(root: Path, splits: tuple[str, ...] = ("dev", "holdout")) -> list[str]:
    """Cross-file coherence of what was actually written to disk."""
    import csv

    problems: list[str] = []
    seen_payments: dict[str, str] = {}

    for split in splits:
        directory = root / split
        gt = GroundTruth.model_validate_json((directory / "ground_truth.json").read_text())

        gateway_payment_ids = set()
        with (directory / "gateway_settlements.csv").open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row["entity_type"] == "payment":
                    gateway_payment_ids.add(row["payment_id"])
        invoice_ids = set()
        with (directory / "erp_invoices.csv").open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                invoice_ids.add(row["invoice_id"])
        bank_txn_ids = set()
        with (directory / "bank_statement.csv").open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                bank_txn_ids.add(row["txn_id"])

        link_ids = {link.payment_id for link in gt.links}
        if link_ids != gateway_payment_ids:
            problems.append(
                f"{split}: ground truth covers {len(link_ids)} payments, "
                f"gateway file has {len(gateway_payment_ids)}"
            )
        for payment_id in link_ids:
            if payment_id in seen_payments:
                problems.append(
                    f"{payment_id} in both {seen_payments[payment_id]} and {split}"
                )
            seen_payments[payment_id] = split

        for link in gt.links:
            if link.invoice_id and link.invoice_id not in invoice_ids:
                problems.append(
                    f"{split}: {link.payment_id} -> missing invoice {link.invoice_id}"
                )
            for txn in link.bank_txn_ids:
                if txn not in bank_txn_ids:
                    problems.append(
                        f"{split}: {link.payment_id} -> missing bank txn {txn}"
                    )
        for txn in gt.orphan_bank_txn_ids:
            if txn not in bank_txn_ids:
                problems.append(f"{split}: orphan bank txn {txn} not in statement")
        for invoice_id in gt.orphan_invoice_ids:
            if invoice_id not in invoice_ids:
                problems.append(f"{split}: orphan invoice {invoice_id} not in ledger")
    return problems


def check_defect_coverage(root: Path, enabled: set[DefectTag]) -> list[str]:
    """Every enabled defect must actually appear in the dev split.

    A silently-zero defect class would make the per-mess-type breakdown lie by
    omission, which is worse than a defect that is simply hard.
    """
    problems: list[str] = []
    gt = GroundTruth.model_validate_json((root / "dev" / "ground_truth.json").read_text())
    present = (
        set(gt.manifest.defect_counts)
        | set(gt.manifest.bundle_defect_counts)
        | set(gt.manifest.payments_affected_by_bundle_defect)
    )
    for tag in enabled:
        if str(tag) not in present:
            problems.append(f"defect {tag} is enabled but never appears in dev")
    return problems


def summarise(root: Path, split: str) -> dict[str, object]:
    data: dict[str, dict[str, object]] = json.loads(
        (root / split / "manifest.json").read_text()
    )
    return data["manifest"]
