"""
search.py
=========
Moteur de recherche en langage naturel.
Permet de retrouver des offres ou des candidats à partir d'une requête libre.

Exemples :
  "Je recherche un développeur Python à Brazzaville"
  "Candidat en comptabilité avec mobilité nationale"
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class NaturalLanguageSearch:
    """
    Recherche sémantique par requête en langage naturel.
    Réutilise le vectoriseur TF-IDF du MatchingEngine.
    """

    def __init__(self, matching_engine):
        """
        Parameters
        ----------
        matching_engine : instance de MatchingEngine déjà entraîné (fit)
        """
        self.engine = matching_engine

    # ------------------------------------------------------------------
    # Recherche d'offres
    # ------------------------------------------------------------------

    def search_jobs(self, query: str, n: int = 5) -> list[dict]:
        """
        Recherche les offres les plus pertinentes pour une requête libre.

        Parameters
        ----------
        query : requête en langage naturel
                Ex: "Je recherche un ingénieur en génie civil à Brazzaville"
        n     : nombre de résultats à retourner

        Returns
        -------
        list of dict: rank, job_id, titre, secteur, localisation, score
        """
        if not query.strip():
            return []

        query_vec = self.engine.vectorizer.transform([query.lower()])
        scores    = cosine_similarity(query_vec, self.engine.job_matrix).flatten()
        top_idx   = np.argsort(scores)[::-1][:n]

        results = []
        for rank, idx in enumerate(top_idx, start=1):
            row = self.engine.jobs_df.iloc[idx]
            results.append({
                "rank":         rank,
                "job_id":       row["job_id"],
                "titre":        row["titre"],
                "secteur":      row["secteur"],
                "localisation": row["localisation"],
                "entreprise":   row.get("entreprise", ""),
                "type_contrat": row.get("type_contrat", ""),
                "description":  row.get("description", "")[:200],
                "score":        round(float(scores[idx]), 4),
            })
        return results

    # ------------------------------------------------------------------
    # Recherche de candidats
    # ------------------------------------------------------------------

    def search_candidates(
        self,
        query: str,
        candidates_df: pd.DataFrame,
        candidate_matrix,
        n: int = 5,
        id_col: str = "Matricule",
    ) -> list[dict]:
        """
        Recherche les candidats les plus pertinents pour une requête libre.

        Parameters
        ----------
        query            : requête en langage naturel
                           Ex: "candidat en comptabilité avec mobilité nationale"
        candidates_df    : DataFrame nettoyé des demandeurs
        candidate_matrix : matrice TF-IDF des candidats (sparse)
        n                : nombre de résultats

        Returns
        -------
        list of dict: rank, candidate_id, metier, secteur, diplome, score
        """
        if not query.strip():
            return []

        query_vec = self.engine.vectorizer.transform([query.lower()])
        scores    = cosine_similarity(query_vec, candidate_matrix).flatten()
        top_idx   = np.argsort(scores)[::-1][:n]

        results = []
        for rank, idx in enumerate(top_idx, start=1):
            row = candidates_df.iloc[idx]
            results.append({
                "rank":           rank,
                "candidate_id":   str(row.get(id_col, "")),
                "metier_vise":    str(row.get("Métier visé / Qualification visée", "")),
                "secteur":        str(row.get("Secteur demandé", "")),
                "diplome":        str(row.get("Diplome", "")),
                "niveau_etude":   str(row.get("niveau_etude", "")),
                "mobilite":       str(row.get("Mobilité géographique", "")),
                "score":          round(float(scores[idx]), 4),
            })
        return results

    def build_candidate_matrix(self, candidates_df: pd.DataFrame):
        """Vectorise les profils textuels des candidats avec le même vectoriseur."""
        texts = candidates_df["text_profile"].fillna("").tolist()
        return self.engine.vectorizer.transform(texts)
