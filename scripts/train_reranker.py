from __future__ import annotations

import argparse

from index.reranker import TrainingConfig, RerankerConfig, TrainingRun


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train phase-A reranker with pool resampling, hybrid losses, and early stopping.")

    p.add_argument("--hits-csv", type=str, default="out/reranker_dataset/eval_hits.csv")
    p.add_argument("--dataset-csv", type=str, default="out/reranker_dataset/dataset.csv")
    p.add_argument("--patch-vectors", type=str, default="out/index/patch_vectors.npy")
    p.add_argument("--patch-manifest", type=str, default="out/index/patch_manifest.parquet")
    p.add_argument("--query-cache", type=str, default="out/reranker/query_vectors.npy")
    p.add_argument("--out", type=str, default="out/reranker/reranker_listwise.pt")

    p.add_argument("--train-topk", type=int, default=100)
    p.add_argument("--list-k", type=int, default=64)
    p.add_argument("--eval-use-full-list", action="store_true")

    p.add_argument("--sampling-mode", type=str, default="stratified")
    p.add_argument("--sample-top-n", type=int, default=16)
    p.add_argument("--sample-mid-n", type=int, default=24)
    p.add_argument("--sample-tail-n", type=int, default=24)
    p.add_argument("--band-top-end", type=int, default=10)
    p.add_argument("--band-mid-end", type=int, default=40)
    p.add_argument("--no-shuffle-candidates", action="store_true")

    p.add_argument("--tau-q-lo", type=float, default=0.10)
    p.add_argument("--tau-q-hi", type=float, default=0.90)
    p.add_argument("--tau-div", type=float, default=4.0)
    p.add_argument("--tau-min", type=float, default=1.0)
    p.add_argument("--tau-max", type=float, default=256.0)

    p.add_argument("--distance-loss-weight", type=float, default=0.25)
    p.add_argument("--distance-loss-type", type=str, default="huber")
    p.add_argument("--distance-target", type=str, default="log1p")
    p.add_argument("--distance-clip-max", type=float, default=256.0)
    p.add_argument("--no-distance-loss", action="store_true")

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--early-stopping-patience", type=int, default=6)
    p.add_argument("--early-stopping-min-delta", type=float, default=0.001)
    p.add_argument("--no-plateau-scheduler", action="store_true")
    p.add_argument("--plateau-patience", type=int, default=2)
    p.add_argument("--plateau-factor", type=float, default=0.5)
    p.add_argument("--plateau-min-lr", type=float, default=1e-6)

    p.add_argument("--embed-dim", type=int, default=768)
    p.add_argument("--hidden", type=int, nargs=2, default=[512, 256])
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--no-pair-features", action="store_true")

    return p.parse_args()


def main() -> None:
    a = parse_args()

    tcfg = TrainingConfig(
        hits_csv=a.hits_csv,
        dataset_csv=a.dataset_csv,
        patch_vectors_path=a.patch_vectors,
        patch_manifest_path=a.patch_manifest,
        query_vectors_cache=a.query_cache,
        out_path=a.out,
        train_topk=a.train_topk,
        list_k=a.list_k,
        eval_use_full_list=a.eval_use_full_list,
        sampling_mode=a.sampling_mode,
        shuffle_candidates=(not a.no_shuffle_candidates),
        sample_top_n=a.sample_top_n,
        sample_mid_n=a.sample_mid_n,
        sample_tail_n=a.sample_tail_n,
        band_top_end=a.band_top_end,
        band_mid_end=a.band_mid_end,
        tau_q_lo=a.tau_q_lo,
        tau_q_hi=a.tau_q_hi,
        tau_div=a.tau_div,
        tau_min=a.tau_min,
        tau_max=a.tau_max,
        use_distance_loss=(not a.no_distance_loss),
        distance_loss_weight=a.distance_loss_weight,
        distance_loss_type=a.distance_loss_type,
        distance_target=a.distance_target,
        distance_clip_max=a.distance_clip_max,
        batch_size=a.batch_size,
        num_epochs=a.epochs,
        lr=a.lr,
        weight_decay=a.weight_decay,
        device=a.device,
        num_workers=a.num_workers,
        seed=a.seed,
        early_stopping_patience=a.early_stopping_patience,
        early_stopping_min_delta=a.early_stopping_min_delta,
        use_plateau_scheduler=(not a.no_plateau_scheduler),
        plateau_patience=a.plateau_patience,
        plateau_factor=a.plateau_factor,
        plateau_min_lr=a.plateau_min_lr,
    )

    mcfg = RerankerConfig(
        embed_dim=a.embed_dim,
        hidden_dims=(int(a.hidden[0]), int(a.hidden[1])),
        dropout=float(a.dropout),
        device=a.device,
        use_pair_features=not a.no_pair_features,
        use_distance_head=True,
    )

    TrainingRun(tcfg, mcfg).run()


if __name__ == "__main__":
    main()
