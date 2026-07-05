"""
evaluation.py
=============
Métriques d'évaluation des recommandations :
  - Precision@K
  - Recall@K
  - NDCG@K
"""

import numpy as np
import pandas as pd
from typing import Union


# ---------------------------------------------------------------------------
# Métriques individuelles
# ---------------------------------------------------------------------------

def precision_at_k(recommended: list[str], relevant: list[str], k: int) -> float:
    """
    Precision@K = |recommandé[:K] ∩ pertinent| / K
    """
    if k <= 0:
        return 0.0
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / k


def recall_at_k(recommended: list[str], relevant: list[str], k: int) -> float:
    """
    Recall@K = |recommandé[:K] ∩ pertinent| / |pertinent|
    """
    if not relevant:
        return 0.0
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / len(relevant)


def ndcg_at_k(recommended: list[str], relevant: list[str], k: int) -> float:
    """
    NDCG@K — Normalized Discounted Cumulative Gain.
    Tient compte de l'ordre : les résultats pertinents en tête rapportent plus.
    """
    if not relevant or k <= 0:
        return 0.0

    recommended_k = recommended[:k]
    relevant_set  = set(relevant)

    # DCG : gain actualisé
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, item in enumerate(recommended_k)
        if item in relevant_set
    )

    # IDCG : gain idéal (si tous les pertinents étaient en tête)
    n_relevant_in_k = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant_in_k))

    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Évaluation globale
# ---------------------------------------------------------------------------

def evaluate_all(
    predictions: dict[str, list[str]],
    ground_truth: dict[str, list[str]],
    k_values: list[int] = [5, 10],
) -> dict:
    """
    Évalue les recommandations sur l'ensemble des candidats.

    Parameters
    ----------
    predictions : {candidate_id: [job_id_rank1, job_id_rank2, ...]}
    ground_truth: {candidate_id: [relevant_job_id1, relevant_job_id2, ...]}
    k_values    : liste des valeurs de K à évaluer

    Returns
    -------
    dict : {k: {Precision@k, Recall@k, NDCG@k}}
    """
    results = {}
    for k in k_values:
        p_scores, r_scores, n_scores = [], [], []
        for cid, recs in predictions.items():
            relevant = ground_truth.get(cid, [])
            p_scores.append(precision_at_k(recs, relevant, k))
            r_scores.append(recall_at_k(recs, relevant, k))
            n_scores.append(ndcg_at_k(recs, relevant, k))

        results[k] = {
            f"Precision@{k}": round(float(np.mean(p_scores)), 4),
            f"Recall@{k}":    round(float(np.mean(r_scores)), 4),
            f"NDCG@{k}":      round(float(np.mean(n_scores)), 4),
        }

    return results


def evaluate_from_dataframe(
    recs_df: pd.DataFrame,
    ground_truth: dict[str, list[str]],
    k_values: list[int] = [5, 10],
) -> dict:
    """
    Évalue à partir d'un DataFrame au format (candidate_id, rank, job_id, score).
    """
    # Construire predictions: {candidate_id: [job_id en ordre de rank]}
    predictions = {}
    for cid, group in recs_df.groupby("candidate_id"):
        group_sorted = group.sort_values("rank")
        predictions[str(cid)] = group_sorted["job_id"].astype(str).tolist()

    return evaluate_all(predictions, ground_truth, k_values)


def print_evaluation_report(results: dict) -> None:
    """Affiche un rapport lisible des métriques."""
    print("\n" + "=" * 50)
    print("         RAPPORT D'ÉVALUATION")
    print("=" * 50)
    for k, metrics in sorted(results.items()):
        print(f"\n  @ K = {k}")
        for metric, value in metrics.items():
            print(f"    {metric:<15} : {value:.4f}  ({value*100:.2f}%)")
    print("=" * 50 + "\n")
