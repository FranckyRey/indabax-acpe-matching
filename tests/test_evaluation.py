"""
Tests des métriques d'évaluation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.evaluation import precision_at_k, recall_at_k, ndcg_at_k, evaluate_all


class TestPrecisionAtK:
    def test_perfect(self):
        assert precision_at_k(["A", "B", "C"], ["A", "B", "C"], k=3) == 1.0

    def test_zero(self):
        assert precision_at_k(["X", "Y", "Z"], ["A", "B", "C"], k=3) == 0.0

    def test_partial(self):
        # 1 hit sur 3
        result = precision_at_k(["A", "X", "Y"], ["A", "B", "C"], k=3)
        assert abs(result - 1/3) < 1e-9

    def test_k_larger_than_recommended(self):
        # K=5, seulement 3 recommandations : 2 hits / 5
        result = precision_at_k(["A", "B", "X"], ["A", "B", "C"], k=5)
        assert abs(result - 2/5) < 1e-9

    def test_k_zero(self):
        assert precision_at_k(["A"], ["A"], k=0) == 0.0


class TestRecallAtK:
    def test_perfect(self):
        assert recall_at_k(["A", "B", "C"], ["A", "B", "C"], k=3) == 1.0

    def test_zero(self):
        assert recall_at_k(["X", "Y"], ["A", "B"], k=2) == 0.0

    def test_partial(self):
        # 1 hit sur 3 pertinents
        result = recall_at_k(["A", "X", "Y"], ["A", "B", "C"], k=3)
        assert abs(result - 1/3) < 1e-9

    def test_empty_relevant(self):
        assert recall_at_k(["A", "B"], [], k=2) == 0.0

    def test_k5_with_3_relevant(self):
        # 2 hits sur 3 pertinents
        result = recall_at_k(["A", "B", "X", "Y", "Z"], ["A", "B", "C"], k=5)
        assert abs(result - 2/3) < 1e-9


class TestNdcgAtK:
    def test_perfect(self):
        # Tous les pertinents en tête dans le bon ordre
        assert ndcg_at_k(["A", "B", "C"], ["A", "B", "C"], k=3) == 1.0

    def test_zero(self):
        assert ndcg_at_k(["X", "Y", "Z"], ["A", "B", "C"], k=3) == 0.0

    def test_order_matters(self):
        # Résultat pertinent en position 1 vs position 2
        ndcg_first  = ndcg_at_k(["A", "X"], ["A"], k=2)
        ndcg_second = ndcg_at_k(["X", "A"], ["A"], k=2)
        assert ndcg_first > ndcg_second

    def test_empty_relevant(self):
        assert ndcg_at_k(["A", "B"], [], k=2) == 0.0

    def test_k_zero(self):
        assert ndcg_at_k(["A"], ["A"], k=0) == 0.0


class TestEvaluateAll:
    def test_basic_evaluation(self):
        predictions = {
            "C1": ["J1", "J2", "J3", "J4", "J5"],
            "C2": ["J4", "J5", "J6", "J7", "J8"],
        }
        ground_truth = {
            "C1": ["J1", "J2", "J3"],
            "C2": ["J4", "J5", "J6"],
        }
        results = evaluate_all(predictions, ground_truth, k_values=[5])
        assert results[5]["Precision@5"] == 0.6    # (3/5 + 3/5) / 2
        assert results[5]["Recall@5"]    == 1.0    # (3/3 + 3/3) / 2

    def test_multiple_k_values(self):
        predictions  = {"C1": ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8", "J9", "J10"]}
        ground_truth = {"C1": ["J1", "J2", "J3"]}
        results = evaluate_all(predictions, ground_truth, k_values=[5, 10])
        assert 5 in results
        assert 10 in results
        assert "Precision@5" in results[5]
        assert "NDCG@10" in results[10]

    def test_missing_candidate_in_ground_truth(self):
        predictions  = {"C1": ["J1", "J2", "J3"]}
        ground_truth = {}   # Pas de ground truth
        results = evaluate_all(predictions, ground_truth, k_values=[3])
        # Tous les scores doivent être 0
        assert results[3]["Precision@3"] == 0.0
        assert results[3]["Recall@3"]    == 0.0
        assert results[3]["NDCG@3"]      == 0.0
