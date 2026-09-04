"""Emit one split of a universe as CSV sources plus a JSON ground-truth file.

Column order is fixed and explicit: these files are meant to look like exports
a finance team would actually receive, and a stable column order keeps diffs
between regenerations readable.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from dataclasses import field as dc_field
from datetime import date, datetime, timedelta
from pathlib import Path

from recon.generator.config import MessConfig
from recon.generator.generate import (
    GENERATOR_VERSION,
    IST,
    PROPAGATED_BUNDLE_TAGS,
    Universe,
    _Bundle,
)
from recon.models import (
    BankRow,
    DefectTag,
    EntityType,
    GatewayRow,
    GroundTruth,
    GroundTruthBundle,
    GroundTruthLink,
    InvoiceRow,
    Manifest,
)
from recon.money import format_rupees

GATEWAY_COLUMNS = [
    "entity_type", "entity_id", "payment_id", "order_id", "order_receipt",
    "method", "card_network", "amount", "currency", "fee", "tax", "tds",
    "credit", "debit", "settlement_id", "settlement_utr", "created_at",
    "captured_at", "settled_at", "description",
]

BANK_COLUMNS = [
    "txn_id", "value_date", "posted_date", "narration", "utr",
    "credit_amount", "debit_amount", "balance", "ref_no",
]

INVOICE_COLUMNS = [
    "invoice_id", "customer_id", "customer_name", "invoice_amount",
    "tax_amount", "currency", "issue_date", "due_date", "order_id",
    "po_number", "status", "notes",
]

OPENING_BALANCE_PAISE = 42_18_650_00


def _settlement_ts(day: date) -> datetime:
    """Payouts land in the 11:30 IST cycle; every row in a batch shares it."""
    return datetime.combine(day, datetime.min.time(), tzinfo=IST) + timedelta(hours=11, minutes=30)


def _iso(value: datetime | None) -> str:
    return value.isoformat() if value else ""


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    import csv

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------
def _gateway_rows_for(universe: Universe, bundle_ids: set[str]) -> list[GatewayRow]:
    by_bundle = {b.bundle_id: b for b in universe.bundles}
    rows: list[GatewayRow] = []

    for pay in universe.payments:
        portions = [(bid, amt) for bid, amt in pay.portions if bid in bundle_ids]
        if not portions:
            continue
        for index, (bid, portion) in enumerate(pay.portions):
            if bid not in bundle_ids:
                continue
            bundle = by_bundle[bid]
            settled_at = _settlement_ts(bundle.settled_date)
            if index == 0:
                rows.append(
                    GatewayRow(
                        entity_type=EntityType.PAYMENT,
                        entity_id=pay.payment_id,
                        payment_id=pay.payment_id,
                        order_id=pay.reported_order_id,
                        order_receipt=pay.receipt,
                        method=pay.method,
                        card_network=("VISA" if pay.method.value == "card" else None),
                        amount_paise=pay.gross,
                        fee_paise=pay.fee,
                        tax_paise=pay.tax,
                        tds_paise=pay.tds,
                        credit_paise=portion,
                        debit_paise=0,
                        settlement_id=bundle.bundle_id,
                        settlement_utr=bundle.utr,
                        created_at=pay.created_at,
                        captured_at=pay.captured_at,
                        settled_at=settled_at,
                        description=pay.description,
                    )
                )
            else:
                # Deferred balance release: the remainder of a split settlement
                # arrives on a later payout with no fresh fee/tax of its own.
                rows.append(
                    GatewayRow(
                        entity_type=EntityType.PAYMENT,
                        entity_id=pay.payment_id,
                        payment_id=pay.payment_id,
                        order_id=pay.reported_order_id,
                        order_receipt=pay.receipt,
                        method=pay.method,
                        card_network=None,
                        amount_paise=0,
                        fee_paise=0,
                        tax_paise=0,
                        tds_paise=0,
                        credit_paise=portion,
                        debit_paise=0,
                        settlement_id=bundle.bundle_id,
                        settlement_utr=bundle.utr,
                        created_at=pay.created_at,
                        captured_at=pay.captured_at,
                        settled_at=settled_at,
                        description="Deferred settlement balance",
                    )
                )

    for refund in universe.refunds:
        if refund.bundle_id not in bundle_ids:
            continue
        bundle = by_bundle[refund.bundle_id]
        rows.append(
            GatewayRow(
                entity_type=EntityType.REFUND,
                entity_id=refund.refund_id,
                payment_id=refund.payment_id,
                order_id=refund.order_id,
                order_receipt="",
                method=None,
                card_network=None,
                amount_paise=refund.amount,
                fee_paise=0,
                tax_paise=0,
                tds_paise=0,
                credit_paise=0,
                debit_paise=refund.amount,
                settlement_id=bundle.bundle_id,
                settlement_utr=bundle.utr,
                created_at=refund.created_at,
                captured_at=None,
                settled_at=_settlement_ts(bundle.settled_date),
                description="Refund",
            )
        )

    for adj in universe.adjustments:
        if adj.bundle_id not in bundle_ids:
            continue
        bundle = by_bundle[adj.bundle_id]
        rows.append(
            GatewayRow(
                entity_type=EntityType.ADJUSTMENT,
                entity_id=adj.adjustment_id,
                payment_id=None,
                order_id=None,
                order_receipt="",
                method=None,
                card_network=None,
                amount_paise=adj.amount,
                fee_paise=0,
                tax_paise=0,
                tds_paise=0,
                credit_paise=0,
                debit_paise=adj.amount,
                settlement_id=bundle.bundle_id,
                settlement_utr=bundle.utr,
                created_at=adj.created_at,
                captured_at=None,
                settled_at=adj.created_at,
                description=adj.description,
            )
        )

    rows.sort(key=lambda r: (r.settled_at or r.created_at, r.entity_id))
    return rows


def _bundle_tags_for_payment(bundle: _Bundle) -> set[DefectTag]:
    """Batch-level defects a payment inside the batch genuinely suffers from.

    ``NO_BANK_CREDIT`` is excluded: it is recorded as the payment's own
    unresolvable reason, so counting it twice would double-report it.
    ``SPLIT_SETTLEMENT`` is excluded because the deferred-balance row is present
    in the report, so the batch still ties -- the split hurts the split payment
    itself, not the payments riding alongside it.  Left in, it read as "88% of
    payments are affected by split settlements", which is not true.
    """
    return bundle.tags & PROPAGATED_BUNDLE_TAGS


def build_ground_truth(
    universe: Universe,
    bundle_ids: set[str],
    split_name: str,
    noise_txn_ids: list[str],
    orphan_invoice_ids: list[str],
    counts: dict[str, int],
) -> GroundTruth:
    by_bundle = {b.bundle_id: b for b in universe.bundles}
    links: list[GroundTruthLink] = []

    for pay in universe.payments:
        portions = [(bid, amt) for bid, amt in pay.portions if bid in bundle_ids]
        if not portions:
            continue
        own_tags = set(pay.tags)
        bundle_tags: set[DefectTag] = set()
        for bid, _ in pay.portions:
            bundle_tags |= _bundle_tags_for_payment(by_bundle[bid])
        own_tags -= bundle_tags
        bank_ids = [
            by_bundle[bid].bank_txn_id
            for bid, _ in pay.portions
            if by_bundle[bid].bank_txn_id is not None
        ]
        links.append(
            GroundTruthLink(
                payment_id=pay.payment_id,
                order_id=pay.true_order_id if pay.reported_order_id else None,
                invoice_id=pay.invoice_id,
                settlement_ids=[bid for bid, _ in pay.portions],
                utrs=[by_bundle[bid].utr for bid, _ in pay.portions],
                bank_txn_ids=[b for b in bank_ids if b],
                gross_paise=pay.gross,
                net_paise=pay.credit,
                defect_tags=sorted(own_tags, key=str),
                bundle_defect_tags=sorted(bundle_tags, key=str),
                unresolvable_reason=pay.unresolvable_reason,
            )
        )

    gt_bundles: list[GroundTruthBundle] = []
    for bid in sorted(bundle_ids):
        bundle = by_bundle[bid]
        gt_bundles.append(
            GroundTruthBundle(
                utr=bundle.utr,
                settlement_id=bundle.bundle_id,
                bank_txn_id=bundle.bank_txn_id,
                expected_credit_paise=bundle.credit_total,
                payment_ids=sorted(set(bundle.payment_ids)),
                refund_entity_ids=[r.refund_id for r in universe.refunds if r.bundle_id == bid],
                adjustment_entity_ids=[
                    a.adjustment_id for a in universe.adjustments if a.bundle_id == bid
                ],
                defect_tags=sorted(bundle.tags & PROPAGATED_BUNDLE_TAGS, key=str),
            )
        )

    defect_counts: dict[str, int] = {}
    affected: dict[str, int] = {}
    for link in links:
        for tag in link.defect_tags:
            defect_counts[str(tag)] = defect_counts.get(str(tag), 0) + 1
        for tag in link.bundle_defect_tags:
            affected[str(tag)] = affected.get(str(tag), 0) + 1
    bundle_counts: dict[str, int] = {}
    for gtb in gt_bundles:
        for tag in gtb.defect_tags:
            bundle_counts[str(tag)] = bundle_counts.get(str(tag), 0) + 1

    manifest = Manifest(
        split=split_name,
        seed=universe.config.seed,
        generator_version=GENERATOR_VERSION,
        config_hash=universe.config.config_hash(),
        generated_at=datetime(2026, 1, 1, tzinfo=IST),  # fixed: keeps files byte-stable
        n_payments=len(links),
        n_bundles=len(gt_bundles),
        n_gateway_rows=counts["gateway"],
        n_bank_rows=counts["bank"],
        n_invoice_rows=counts["invoice"],
        defect_counts=dict(sorted(defect_counts.items())),
        bundle_defect_counts=dict(sorted(bundle_counts.items())),
        payments_affected_by_bundle_defect=dict(sorted(affected.items())),
    )
    return GroundTruth(
        manifest=manifest,
        bundles=gt_bundles,
        links=links,
        orphan_invoice_ids=sorted(orphan_invoice_ids),
        orphan_bank_txn_ids=sorted(noise_txn_ids),
    )


# --------------------------------------------------------------------------
@dataclass
class _Extras:
    """Records that belong to no settlement batch, dealt out per split."""

    bank: list[BankRow] = dc_field(default_factory=list)
    invoices: list[str] = dc_field(default_factory=list)


def _bank_rows_for(
    universe: Universe, bundle_ids: set[str], noise: list[BankRow]
) -> list[BankRow]:
    rows = [row for bid, row in universe.bank_rows.items() if bid in bundle_ids]
    rows.extend(noise)
    rows.sort(key=lambda r: (r.value_date, r.txn_id))
    balanced: list[BankRow] = []
    balance = OPENING_BALANCE_PAISE
    for row in rows:
        balance += row.credit_paise - row.debit_paise
        balanced.append(row.model_copy(update={"balance_paise": balance}))
    return balanced


def _invoice_rows_for(
    universe: Universe, payment_ids: set[str], orphan_ids: list[str]
) -> list[InvoiceRow]:
    wanted = {
        p.invoice_id for p in universe.payments if p.payment_id in payment_ids and p.invoice_id
    }
    wanted |= set(orphan_ids)
    rows = [inv for inv in universe.invoices if inv.invoice_id in wanted]
    rows.sort(key=lambda r: r.invoice_id)
    return rows


def allocate_extras(
    universe: Universe, splits: dict[str, set[str]]
) -> dict[str, _Extras]:
    """Deal noise bank credits and orphan invoices out in proportion to split size.

    They are unattached to any bundle, so nothing forces them into one split.
    Dealing them proportionally keeps the exception base rate comparable across
    dev and held-out, which matters because exception precision is a headline
    number.
    """
    rng = random.Random(universe.config.seed + 4_517)
    sizes = {b.bundle_id: len(set(b.payment_ids)) for b in universe.bundles}
    weights = {name: sum(sizes[b] for b in bids) for name, bids in splits.items()}
    total = sum(weights.values()) or 1
    names = list(splits)

    out: dict[str, _Extras] = {n: _Extras() for n in names}
    for row in universe.noise_bank_rows:
        pick = rng.choices(names, weights=[weights[n] / total for n in names], k=1)[0]
        out[pick].bank.append(row)
    for invoice_id in universe.orphan_invoice_ids:
        pick = rng.choices(names, weights=[weights[n] / total for n in names], k=1)[0]
        out[pick].invoices.append(invoice_id)
    return out


def write_split(
    universe: Universe,
    split_name: str,
    bundle_ids: set[str],
    out_dir: Path,
    noise_bank: list[BankRow],
    orphan_invoice_ids: list[str],
) -> Manifest:
    out_dir.mkdir(parents=True, exist_ok=True)
    payment_ids = {
        p.payment_id
        for p in universe.payments
        if any(bid in bundle_ids for bid, _ in p.portions)
    }

    gateway = _gateway_rows_for(universe, bundle_ids)
    bank = _bank_rows_for(universe, bundle_ids, noise_bank)
    invoices = _invoice_rows_for(universe, payment_ids, orphan_invoice_ids)

    _write_csv(
        out_dir / "gateway_settlements.csv",
        GATEWAY_COLUMNS,
        [
            {
                "entity_type": r.entity_type.value,
                "entity_id": r.entity_id,
                "payment_id": r.payment_id or "",
                "order_id": r.order_id or "",
                "order_receipt": r.order_receipt,
                "method": r.method.value if r.method else "",
                "card_network": r.card_network or "",
                "amount": format_rupees(r.amount_paise),
                "currency": r.currency,
                "fee": format_rupees(r.fee_paise),
                "tax": format_rupees(r.tax_paise),
                "tds": format_rupees(r.tds_paise),
                "credit": format_rupees(r.credit_paise),
                "debit": format_rupees(r.debit_paise),
                "settlement_id": r.settlement_id or "",
                "settlement_utr": r.settlement_utr or "",
                "created_at": _iso(r.created_at),
                "captured_at": _iso(r.captured_at),
                "settled_at": _iso(r.settled_at),
                "description": r.description,
            }
            for r in gateway
        ],
    )

    _write_csv(
        out_dir / "bank_statement.csv",
        BANK_COLUMNS,
        [
            {
                "txn_id": r.txn_id,
                "value_date": r.value_date.isoformat(),
                "posted_date": r.posted_date.isoformat(),
                "narration": r.narration,
                "utr": r.utr or "",
                "credit_amount": format_rupees(r.credit_paise),
                "debit_amount": format_rupees(r.debit_paise),
                "balance": format_rupees(r.balance_paise),
                "ref_no": r.ref_no,
            }
            for r in bank
        ],
    )

    _write_csv(
        out_dir / "erp_invoices.csv",
        INVOICE_COLUMNS,
        [
            {
                "invoice_id": r.invoice_id,
                "customer_id": r.customer_id,
                "customer_name": r.customer_name,
                "invoice_amount": format_rupees(r.invoice_amount_paise),
                "tax_amount": format_rupees(r.tax_amount_paise),
                "currency": r.currency,
                "issue_date": r.issue_date.isoformat(),
                "due_date": r.due_date.isoformat(),
                "order_id": r.order_id or "",
                "po_number": r.po_number,
                "status": r.status,
                "notes": r.notes,
            }
            for r in invoices
        ],
    )

    ground_truth = build_ground_truth(
        universe,
        bundle_ids,
        split_name,
        noise_txn_ids=[r.txn_id for r in noise_bank],
        orphan_invoice_ids=orphan_invoice_ids,
        counts={"gateway": len(gateway), "bank": len(bank), "invoice": len(invoices)},
    )
    (out_dir / "ground_truth.json").write_text(
        ground_truth.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "manifest": json.loads(ground_truth.manifest.model_dump_json()),
                "config": universe.config.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return ground_truth.manifest


def write_universe(cfg: MessConfig, universe: Universe, root: Path) -> dict[str, Manifest]:
    from recon.generator.generate import split_universe

    splits = split_universe(universe)
    extras = allocate_extras(universe, splits)
    manifests: dict[str, Manifest] = {}
    for name, bundle_ids in splits.items():
        manifests[name] = write_split(
            universe,
            name,
            bundle_ids,
            root / name,
            noise_bank=extras[name].bank,
            orphan_invoice_ids=extras[name].invoices,
        )
    return manifests
