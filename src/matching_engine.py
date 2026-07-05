"""
matching_engine.py
==================
Moteur d'appariement hybride :
  - TF-IDF + cosine similarity (signal principal)
  - Bonus sectoriel (correspondance candidat ↔ offre)
  - Bonus géographique (mobilité candidat ↔ localisation offre)

Score final = α × sim_tfidf + β × bonus_secteur + γ × bonus_geo
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Poids du score hybride (doivent sommer à 1)
# ---------------------------------------------------------------------------
ALPHA = 0.70   # poids similarité textuelle TF-IDF
BETA  = 0.20   # poids correspondance secteur
GAMMA = 0.10   # poids correspondance géographique


class MatchingEngine:
    """
    Moteur d'appariement candidats ↔ offres.

    Utilisation typique :
        engine = MatchingEngine()
        engine.fit(offres_df)
        recommendations = engine.recommend_top_k(candidate_row, k=10)
    """

    def __init__(
        self,
        alpha: float = ALPHA,
        beta: float  = BETA,
        gamma: float = GAMMA,
        max_features: int = 15_000,
        ngram_range: tuple = (1, 2),
    ):
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            min_df=1,
        )
        self.job_matrix   = None   # (n_jobs, n_features) sparse
        self.jobs_df      = None   # DataFrame des offres (nettoyées)
        self.job_sectors  = None   # liste des secteurs normalisés
        self.job_locs     = None   # liste des localisations normalisées

    # ------------------------------------------------------------------
    # Entraînement / indexation des offres
    # ------------------------------------------------------------------

    def fit(self, jobs_df: pd.DataFrame) -> "MatchingEngine":
        """
        Indexe toutes les offres d'emploi.

        Parameters
        ----------
        jobs_df : DataFrame nettoyé (issu de clean_offres)
                  Doit contenir les colonnes : job_id, text_profile,
                  secteur, localisation_norm
        """
        self.jobs_df = jobs_df.reset_index(drop=True)

        # Vectoriser les profils textuels des offres
        job_texts = self.jobs_df["text_profile"].fillna("").tolist()
        self.job_matrix = self.vectorizer.fit_transform(job_texts)

        # Caches pour les bonus catégoriels
        self.job_sectors = self.jobs_df["secteur"].fillna("").str.lower().tolist()
        self.job_locs    = self.jobs_df["localisation_norm"].fillna("").str.lower().tolist()

        return self

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _semantic_scores(self, candidate_text: str) -> np.ndarray:
        """Retourne le vecteur de similarité cosinus candidat ↔ toutes les offres."""
        cand_vec = self.vectorizer.transform([candidate_text])
        sims = cosine_similarity(cand_vec, self.job_matrix).flatten()
        return sims

    def _sector_bonus(self, candidate_sector: str) -> np.ndarray:
        """
        Score de correspondance sectorielle (0.0 à 1.0) pour chaque offre.
        - 1.0 : correspondance exacte
        - 0.5 : correspondance partielle (un mot en commun)
        - 0.0 : aucune correspondance
        """
        cs = candidate_sector.lower().strip()
        cs_words = set(cs.split())

        bonuses = np.zeros(len(self.job_sectors))
        for i, js in enumerate(self.job_sectors):
            if not js:
                continue
            if cs == js:
                bonuses[i] = 1.0
            elif cs_words & set(js.split()):
                bonuses[i] = 0.5
        return bonuses

    def _geo_bonus(self, candidate_mobility: str) -> np.ndarray:
        """
        Score de correspondance géographique (0.0 à 1.0) pour chaque offre.
        - 1.0 : mobilité nationale ou internationale
        - 0.8 : même ville déclarée
        - 0.4 : même région/pays
        - 0.0 : aucune correspondance
        """
        mob = candidate_mobility.lower().strip()
        national_keywords = {"national", "tout", "toute", "partout", "international", "oui"}

        bonuses = np.zeros(len(self.job_locs))
        is_national = any(kw in mob for kw in national_keywords)

        for i, jloc in enumerate(self.job_locs):
            if not jloc or jloc == "non déclaré":
                bonuses[i] = 0.2
                continue
            if is_national:
                bonuses[i] = 1.0
            elif jloc in mob or mob in jloc:
                bonuses[i] = 0.8
            elif "congo" in jloc or "congo" in mob:
                bonuses[i] = 0.3
        return bonuses

    def score_all(self, candidate_row: pd.Series) -> np.ndarray:
        """
        Calcule le score hybride pour un candidat contre toutes les offres.

        Returns
        -------
        scores : np.ndarray de forme (n_jobs,), valeurs dans [0, 1]
        """
        text     = str(candidate_row.get("text_profile", ""))
        sector   = str(candidate_row.get("Secteur demandé", ""))
        mobility = str(candidate_row.get("mobilite_norm", ""))

        sem  = self._semantic_scores(text)
        sec  = self._sector_bonus(sector)
        geo  = self._geo_bonus(mobility)

        scores = self.alpha * sem + self.beta * sec + self.gamma * geo
        # Normaliser dans [0, 1]
        max_score = scores.max()
        if max_score > 0:
            scores = scores / max_score
        return scores

    # ------------------------------------------------------------------
    # Recommandation Top-K
    # ------------------------------------------------------------------

    def recommend_top_k(
        self,
        candidate_row: pd.Series,
        k: int = 10,
        include_details: bool = True,
    ) -> list[dict]:
        """
        Retourne les k meilleures offres pour un candidat.

        Returns
        -------
        list of dict avec keys: rank, job_id, titre, secteur,
                                localisation, type_contrat, score,
                                score_semantic, score_secteur, score_geo
        """
        scores   = self.score_all(candidate_row)
        top_idx  = np.argsort(scores)[::-1][:k]

        text   = str(candidate_row.get("text_profile", ""))
        sector = str(candidate_row.get("Secteur demandé", ""))
        mob    = str(candidate_row.get("mobilite_norm", ""))

        sem_scores = self._semantic_scores(text)
        sec_scores = self._sector_bonus(sector)
        geo_scores = self._geo_bonus(mob)

        results = []
        for rank, idx in enumerate(top_idx, start=1):
            row = self.jobs_df.iloc[idx]
            entry = {
                "rank":           rank,
                "job_id":         row["job_id"],
                "titre":          row["titre"],
                "secteur":        row["secteur"],
                "localisation":   row["localisation"],
                "type_contrat":   row.get("type_contrat", ""),
                "entreprise":     row.get("entreprise", ""),
                "score":          round(float(scores[idx]), 4),
            }
            if include_details:
                entry["score_semantique"] = round(float(sem_scores[idx]), 4)
                entry["score_secteur"]    = round(float(sec_scores[idx]), 4)
                entry["score_geo"]        = round(float(geo_scores[idx]), 4)
            results.append(entry)

        return results

    # ------------------------------------------------------------------
    # Génération en batch (tous les candidats)
    # ------------------------------------------------------------------

    def recommend_all(
        self,
        candidates_df: pd.DataFrame,
        k: int = 10,
        id_col: str = "Matricule",
    ) -> pd.DataFrame:
        """
        Génère les recommandations Top-K pour tous les candidats.

        Returns
        -------
        DataFrame avec colonnes : candidate_id, rank, job_id, score
        """
        records = []
        for _, row in tqdm(candidates_df.iterrows(), total=len(candidates_df),
                           desc=f"Génération recommandations Top-{k}"):
            cid  = str(row[id_col])
            recs = self.recommend_top_k(row, k=k, include_details=False)
            for rec in recs:
                records.append({
                    "candidate_id": cid,
                    "rank":         rec["rank"],
                    "job_id":       rec["job_id"],
                    "score":        rec["score"],
                })
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Persistance
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Sauvegarde le moteur entraîné."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "MatchingEngine":
        """Charge un moteur sauvegardé."""
        with open(Path(path), "rb") as f:
            return pickle.load(f)

    # ------------------------------------------------------------------
    # Explication des scores
    # ------------------------------------------------------------------

    def explain(self, candidate_row: pd.Series, job_id: str) -> dict:
        """
        Explique le score d'une paire candidat-offre spécifique.
        """
        job_rows = self.jobs_df[self.jobs_df["job_id"] == job_id]
        if job_rows.empty:
            return {"error": f"job_id '{job_id}' introuvable"}

        job_idx  = job_rows.index[0]
        text     = str(candidate_row.get("text_profile", ""))
        sector   = str(candidate_row.get("Secteur demandé", ""))
        mobility = str(candidate_row.get("mobilite_norm", ""))

        sem = float(self._semantic_scores(text)[job_idx])
        sec = float(self._sector_bonus(sector)[job_idx])
        geo = float(self._geo_bonus(mobility)[job_idx])
        total = self.alpha * sem + self.beta * sec + self.gamma * geo

        return {
            "job_id":             job_id,
            "titre":              self.jobs_df.iloc[job_idx]["titre"],
            "score_final":        round(total, 4),
            "contributions": {
                "similarite_textuelle": {
                    "valeur": round(sem, 4),
                    "poids":  self.alpha,
                    "impact": round(self.alpha * sem, 4),
                },
                "correspondance_secteur": {
                    "valeur": round(sec, 4),
                    "poids":  self.beta,
                    "impact": round(self.beta * sec, 4),
                },
                "correspondance_geo": {
                    "valeur": round(geo, 4),
                    "poids":  self.gamma,
                    "impact": round(self.gamma * geo, 4),
                },
            },
            "profil_candidat": text[:300],
            "profil_offre":    self.jobs_df.iloc[job_idx]["text_profile"][:300],
        }
