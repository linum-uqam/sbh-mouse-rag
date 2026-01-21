from __future__ import annotations

import argparse
from pathlib import Path

from index.reranker import TrainingConfig, RerankerConfig, TrainingRun


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train listwise reranker from eval_hits.csv soft labels (gt_prob).")

    p.add_argument("--hits-csv", type=str, default="out/reranker_dataset/eval_hits.csv")
    p.add_argument("--dataset-csv", type=str, default="out/reranker_dataset/dataset.csv")
    p.add_argument("--patch-vectors", type=str, default="out/index/patch_vectors.npy")
    p.add_argument("--patch-manifest", type=str, default="out/index/patch_manifest.parquet")
    p.add_argument("--query-cache", type=str, default="out/reranker/query_vectors.npy")
    p.add_argument("--out", type=str, default="out/reranker/reranker_listwise.pt")

    p.add_argument("--train-topk", type=int, default=100)
    p.add_argument("--list-k", type=int, default=100)

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    # model
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
        batch_size=a.batch_size,
        num_epochs=a.epochs,
        lr=a.lr,
        weight_decay=a.weight_decay,
        device=a.device,
        num_workers=a.num_workers,
        seed=a.seed,
    )

    mcfg = RerankerConfig(
        embed_dim=a.embed_dim,
        hidden_dims=(int(a.hidden[0]), int(a.hidden[1])),
        dropout=float(a.dropout),
        device=a.device,
        use_pair_features=not a.no_pair_features,
    )

    TrainingRun(tcfg, mcfg).run()


if __name__ == "__main__":
    main()
