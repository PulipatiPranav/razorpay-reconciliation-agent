"""Generator properties: reproducibility, independent toggles, coherence."""

from __future__ import annotations

from pathlib import Path

import pytest

from recon.generator.config import MessConfig
from recon.generator.generate import generate_universe, split_universe
from recon.generator.validate import check_split_files, check_universe
from recon.generator.writers import write_universe
from recon.models import DefectTag, GroundTruth


@pytest.fixture(scope="module")
def universe():
    return generate_universe(MessConfig())


def test_seed_reproduces_the_universe_exactly() -> None:
    a = generate_universe(MessConfig(seed=7))
    b = generate_universe(MessConfig(seed=7))
    assert [p.payment_id for p in a.payments] == [p.payment_id for p in b.payments]
    assert [p.gross for p in a.payments] == [p.gross for p in b.payments]
    assert [b_.credit_total for b_ in a.bundles] == [b_.credit_total for b_ in b.bundles]


def test_different_seeds_produce_different_universes() -> None:
    a = generate_universe(MessConfig(seed=7))
    b = generate_universe(MessConfig(seed=8))
    assert [p.payment_id for p in a.payments] != [p.payment_id for p in b.payments]


def test_invariants_hold(universe) -> None:
    assert check_universe(universe) == []


def test_requested_volume_is_produced(universe) -> None:
    assert len(universe.payments) == 600


def test_config_hash_changes_when_a_toggle_changes() -> None:
    base = MessConfig()
    off = MessConfig(disabled=frozenset({DefectTag.TDS}))
    assert base.config_hash() != off.config_hash()


# --- independent toggles ---------------------------------------------------
@pytest.mark.parametrize(
    ("tag", "probe"),
    [
        (DefectTag.TDS, lambda u: sum(p.tds for p in u.payments)),
        (DefectTag.SPLIT_SETTLEMENT, lambda u: sum(len(p.portions) - 1 for p in u.payments)),
        (DefectTag.REFUND_FULL, lambda u: len(u.refunds)),
        (
            DefectTag.CHARGEBACK_ADJUSTMENT,
            lambda u: len(u.adjustments),
        ),
        (
            DefectTag.NO_INVOICE,
            lambda u: sum(1 for p in u.payments if DefectTag.NO_INVOICE in p.tags),
        ),
        (
            DefectTag.PAISE_DRIFT_ROW,
            lambda u: sum(1 for p in u.payments if DefectTag.PAISE_DRIFT_ROW in p.tags),
        ),
    ],
)
def test_disabling_a_defect_removes_it_entirely(tag: DefectTag, probe) -> None:
    on = generate_universe(MessConfig())
    off = generate_universe(MessConfig(disabled=frozenset({tag})))
    assert probe(on) > 0, "defect should be present when enabled"
    assert probe(off) == 0, "defect should vanish when disabled"


def test_disabling_one_defect_leaves_the_others_present() -> None:
    off = generate_universe(MessConfig(disabled=frozenset({DefectTag.TDS})))
    assert sum(off_p.tds for off_p in off.payments) == 0
    assert len(off.refunds) > 0
    assert any(DefectTag.PAISE_DRIFT_ROW in p.tags for p in off.payments)


def test_defect_rates_land_near_their_configured_proportions(universe) -> None:
    cfg = universe.config
    n = len(universe.payments)
    observed = {
        DefectTag.TDS: sum(1 for p in universe.payments if DefectTag.TDS in p.tags) / n,
        DefectTag.PAISE_DRIFT_ROW: sum(
            1 for p in universe.payments if DefectTag.PAISE_DRIFT_ROW in p.tags
        )
        / n,
        DefectTag.SPLIT_SETTLEMENT: sum(
            1 for p in universe.payments if DefectTag.SPLIT_SETTLEMENT in p.tags
        )
        / n,
    }
    for tag, rate in observed.items():
        target = cfg.rate_for(tag)
        assert abs(rate - target) < 0.05, f"{tag}: {rate:.3f} vs target {target:.3f}"


# --- unresolvable records --------------------------------------------------
def test_unresolvable_records_exist_and_are_labelled(universe) -> None:
    unresolvable = [p for p in universe.payments if p.unresolvable_reason]
    assert 15 <= len(unresolvable) <= 60
    assert {p.unresolvable_reason for p in unresolvable} == {
        "no_erp_counterpart",
        "no_order_id_and_no_erp_counterpart",
        "settlement_never_credited",
    }


def test_orphan_invoices_point_at_no_payment(universe) -> None:
    orders = {p.true_order_id for p in universe.payments}
    orphan_ids = set(universe.orphan_invoice_ids)
    orphans = {i.invoice_id for i in universe.invoices if i.invoice_id in orphan_ids}
    assert orphans
    for invoice in universe.invoices:
        if invoice.invoice_id in orphans:
            assert invoice.order_id not in orders


def test_noise_bank_credits_match_no_settlement(universe) -> None:
    settlement_utrs = {b.utr for b in universe.bundles}
    assert universe.noise_bank_rows
    for row in universe.noise_bank_rows:
        assert row.utr not in settlement_utrs


# --- splitting -------------------------------------------------------------
def test_splits_are_disjoint_and_cover_everything(universe) -> None:
    splits = split_universe(universe)
    dev, holdout = splits["dev"], splits["holdout"]
    assert dev & holdout == set()
    assert dev | holdout == {b.bundle_id for b in universe.bundles}


def test_no_payment_appears_in_both_splits(universe) -> None:
    splits = split_universe(universe)
    by_split = {}
    for name, bundle_ids in splits.items():
        payments = {
            p.payment_id
            for p in universe.payments
            if any(bid in bundle_ids for bid, _ in p.portions)
        }
        by_split[name] = payments
    assert by_split["dev"] & by_split["holdout"] == set()
    assert len(by_split["dev"]) + len(by_split["holdout"]) == len(universe.payments)


def test_split_settlements_never_straddle_the_split(universe) -> None:
    splits = split_universe(universe)
    owner = {bid: name for name, bids in splits.items() for bid in bids}
    for payment in universe.payments:
        owners = {owner[bid] for bid, _ in payment.portions}
        assert len(owners) == 1, f"{payment.payment_id} straddles the dev/holdout split"


def test_written_files_are_coherent(tmp_path: Path, universe) -> None:
    write_universe(universe.config, universe, tmp_path)
    assert check_split_files(tmp_path) == []


def test_ground_truth_separates_own_and_batch_defects(tmp_path: Path, universe) -> None:
    write_universe(universe.config, universe, tmp_path)
    gt = GroundTruth.model_validate_json((tmp_path / "dev" / "ground_truth.json").read_text())
    for link in gt.links:
        assert not (set(link.defect_tags) & set(link.bundle_defect_tags))


def test_generation_is_byte_identical_across_runs(tmp_path: Path) -> None:
    cfg = MessConfig(seed=11)
    first, second = tmp_path / "a", tmp_path / "b"
    for target in (first, second):
        write_universe(cfg, generate_universe(cfg), target)
    for split in ("dev", "holdout"):
        for name in (
            "gateway_settlements.csv",
            "bank_statement.csv",
            "erp_invoices.csv",
            "ground_truth.json",
        ):
            assert (first / split / name).read_bytes() == (second / split / name).read_bytes()
