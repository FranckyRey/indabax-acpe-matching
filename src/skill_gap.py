"""
skill_gap.py
============
Analyse des écarts de compétences (Skill Gap) entre un candidat et une offre.
Identifie les compétences ou qualifications manquantes.
"""

import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


# ---------------------------------------------------------------------------
# Dictionnaire de compétences connues (extensible)
# ---------------------------------------------------------------------------
KNOWN_SKILLS = {
    # Informatique / Tech
    "python", "java", "javascript", "sql", "r", "matlab", "c++", "php",
    "html", "css", "excel", "word", "powerpoint", "power bi", "tableau",
    "sas", "spss", "stata", "tensorflow", "keras", "pytorch", "scikit-learn",
    "pandas", "numpy", "git", "github", "docker", "linux", "windows",
    "aws", "azure", "gcp", "mongodb", "postgresql", "mysql",
    # Gestion / Finance
    "comptabilité", "audit", "fiscalité", "budget", "trésorerie",
    "contrôle de gestion", "analyse financière", "sage", "sap",
    "gestion de projet", "ms project", "prince2", "pmp",
    # Langues
    "anglais", "français", "espagnol", "portugais", "lingala", "kikongo",
    # Soft skills
    "communication", "leadership", "travail en équipe", "autonomie",
    "rigueur", "organisation", "adaptabilité", "créativité",
    # Secteurs métier
    "logistique", "supply chain", "rh", "recrutement", "marketing",
    "vente", "commercial", "juridique", "droit", "maintenance",
    "electricité", "mécanique", "hydraulique", "hse", "qualité",
    "génie civil", "btp", "architecture", "topographie",
    "infirmier", "médecine", "pharmacie", "nursing",
    "enseignement", "formation", "pédagogie",
}


def extract_skills(text: str, custom_skills: set[str] | None = None) -> set[str]:
    """
    Extrait les compétences mentionnées dans un texte.

    Parameters
    ----------
    text          : texte brut (profil candidat ou description offre)
    custom_skills : ensemble de compétences supplémentaires à rechercher

    Returns
    -------
    set de compétences trouvées (en minuscules)
    """
    if not text or pd.isna(text):
        return set()

    text_lower = str(text).lower()
    skills_to_check = KNOWN_SKILLS.copy()
    if custom_skills:
        skills_to_check = skills_to_check | {s.lower() for s in custom_skills}

    found = set()
    for skill in skills_to_check:
        # Correspondance sur mot entier ou expression
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found.add(skill)

    return found


def compute_skill_gap(
    candidate_text: str,
    job_text: str,
    custom_skills: set[str] | None = None,
) -> dict:
    """
    Calcule l'écart de compétences entre un candidat et une offre.

    Returns
    -------
    dict avec:
      - skills_candidate  : compétences du candidat
      - skills_job        : compétences requises par l'offre
      - skills_matched    : compétences en commun
      - skills_missing    : compétences manquantes chez le candidat
      - skills_extra      : compétences du candidat non requises par l'offre
      - coverage_rate     : taux de couverture (0.0 à 1.0)
      - compatibility_pct : pourcentage de compatibilité
    """
    cand_skills = extract_skills(candidate_text, custom_skills)
    job_skills  = extract_skills(job_text, custom_skills)

    matched = cand_skills & job_skills
    missing = job_skills - cand_skills
    extra   = cand_skills - job_skills

    coverage = len(matched) / len(job_skills) if job_skills else 1.0

    return {
        "skills_candidate":  sorted(cand_skills),
        "skills_job":        sorted(job_skills),
        "skills_matched":    sorted(matched),
        "skills_missing":    sorted(missing),
        "skills_extra":      sorted(extra),
        "coverage_rate":     round(coverage, 4),
        "compatibility_pct": round(coverage * 100, 1),
    }


def enrich_recommendations_with_gap(
    recommendations: list[dict],
    candidate_text: str,
    jobs_df: pd.DataFrame,
) -> list[dict]:
    """
    Enrichit chaque recommandation avec l'analyse skill gap.

    Parameters
    ----------
    recommendations : liste de dicts retournée par MatchingEngine.recommend_top_k
    candidate_text  : profil textuel du candidat
    jobs_df         : DataFrame des offres (avec colonnes text_profile, job_id)

    Returns
    -------
    Liste enrichie avec la clé 'skill_gap' dans chaque dict
    """
    job_text_map = dict(zip(
        jobs_df["job_id"].astype(str),
        jobs_df["text_profile"].fillna("").astype(str),
    ))

    enriched = []
    for rec in recommendations:
        job_text = job_text_map.get(str(rec["job_id"]), "")
        gap = compute_skill_gap(candidate_text, job_text)
        enriched.append({**rec, "skill_gap": gap})

    return enriched
