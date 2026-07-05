"""
Tests du moteur d'appariement.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest
from src.matching_engine import MatchingEngine


def make_sample_jobs() -> pd.DataFrame:
    """Crée un DataFrame minimal d'offres pour les tests."""
    return pd.DataFrame([
        {
            "job_id": "JOB001",
            "titre": "Data Analyst",
            "secteur": "Technologie",
            "localisation": "Brazzaville",
            "localisation_norm": "brazzaville",
            "type_contrat": "CDI",
            "groupe_contrat": "Emploi",
            "entreprise": "TechCo",
            "type_entreprise": "Privée",
            "poste": "1",
            "date_publication": "2025-01-01",
            "description": "Analyse de données avec Python et SQL",
            "profil": "Bac+3 en statistiques",
            "competences": "Python SQL Excel",
            "text_profile": "Data Analyst Technologie Analyse de données Python SQL statistiques",
        },
        {
            "job_id": "JOB002",
            "titre": "Comptable",
            "secteur": "Finance",
            "localisation": "Pointe-Noire",
            "localisation_norm": "pointe-noire",
            "type_contrat": "CDD",
            "groupe_contrat": "Emploi",
            "entreprise": "FinanceCo",
            "type_entreprise": "Privée",
            "poste": "1",
            "date_publication": "2025-01-02",
            "description": "Gestion de la comptabilité générale",
            "profil": "Bac+2 en comptabilité",
            "competences": "Sage Comptabilité",
            "text_profile": "Comptable Finance comptabilité générale Sage Bac+2",
        },
        {
            "job_id": "JOB003",
            "titre": "Ingénieur Génie Civil",
            "secteur": "BTP",
            "localisation": "Dolisie",
            "localisation_norm": "dolisie",
            "type_contrat": "CDI",
            "groupe_contrat": "Emploi",
            "entreprise": "BuildCo",
            "type_entreprise": "Privée",
            "poste": "1",
            "date_publication": "2025-01-03",
            "description": "Supervision de chantiers de construction",
            "profil": "Bac+5 en génie civil",
            "competences": "AutoCAD BTP",
            "text_profile": "Ingénieur Génie Civil BTP construction AutoCAD supervision",
        },
    ])


def make_sample_candidate() -> pd.Series:
    return pd.Series({
        "Matricule": "CAND001",
        "Métier visé / Qualification visée": "Data Analyst",
        "Secteur demandé": "Technologie",
        "Mobilité géographique": "Brazzaville",
        "mobilite_norm": "brazzaville",
        "text_profile": "Data Analyst Technologie Python SQL statistiques Bac+3",
    })


class TestMatchingEngine:
    def setup_method(self):
        self.engine = MatchingEngine(alpha=0.70, beta=0.20, gamma=0.10)
        self.jobs   = make_sample_jobs()
        self.engine.fit(self.jobs)

    def test_fit_sets_job_matrix(self):
        assert self.engine.job_matrix is not None
        assert self.engine.job_matrix.shape[0] == 3

    def test_score_all_returns_correct_length(self):
        candidate = make_sample_candidate()
        scores = self.engine.score_all(candidate)
        assert len(scores) == 3

    def test_score_all_values_in_range(self):
        candidate = make_sample_candidate()
        scores = self.engine.score_all(candidate)
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_recommend_top_k_returns_k_results(self):
        candidate = make_sample_candidate()
        recs = self.engine.recommend_top_k(candidate, k=2)
        assert len(recs) == 2

    def test_recommend_top_k_ordered_by_score(self):
        candidate = make_sample_candidate()
        recs = self.engine.recommend_top_k(candidate, k=3)
        scores = [r["score"] for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_recommend_top_k_has_required_fields(self):
        candidate = make_sample_candidate()
        recs = self.engine.recommend_top_k(candidate, k=1)
        rec = recs[0]
        assert "rank" in rec
        assert "job_id" in rec
        assert "titre" in rec
        assert "score" in rec

    def test_data_analyst_ranks_first(self):
        """Le candidat Data Analyst doit obtenir JOB001 (Data Analyst) en #1."""
        candidate = make_sample_candidate()
        recs = self.engine.recommend_top_k(candidate, k=3)
        assert recs[0]["job_id"] == "JOB001"

    def test_recommend_all_returns_dataframe(self):
        candidates_df = pd.DataFrame([make_sample_candidate()])
        result = self.engine.recommend_all(candidates_df, k=3, id_col="Matricule")
        assert isinstance(result, pd.DataFrame)
        assert set(["candidate_id", "rank", "job_id", "score"]).issubset(result.columns)

    def test_explain_returns_contributions(self):
        candidate = make_sample_candidate()
        exp = self.engine.explain(candidate, "JOB001")
        assert "contributions" in exp
        assert "similarite_textuelle" in exp["contributions"]

    def test_sector_bonus_exact_match(self):
        bonus = self.engine._sector_bonus("Technologie")
        # JOB001 a secteur Technologie → bonus = 1.0
        idx = self.jobs[self.jobs["job_id"] == "JOB001"].index[0]
        assert bonus[idx] == 1.0

    def test_sector_bonus_no_match(self):
        bonus = self.engine._sector_bonus("Agriculture")
        assert all(b == 0.0 for b in bonus)

    def test_geo_bonus_national_mobility(self):
        bonuses = self.engine._geo_bonus("Mobilité nationale")
        assert all(b == 1.0 for b in bonuses)

    def test_save_and_load(self, tmp_path):
        save_path = tmp_path / "engine.pkl"
        self.engine.save(save_path)
        loaded = MatchingEngine.load(save_path)
        assert loaded.job_matrix.shape == self.engine.job_matrix.shape
