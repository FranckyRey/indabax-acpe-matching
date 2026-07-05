"""
preprocessing.py
================
Chargement, nettoyage et préparation des données brutes pour le moteur d'appariement.
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Chemins par défaut
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------

def load_data(data_dir: str | Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Charge les 4 fichiers Excel et retourne (offres, demandeurs, appariements).
    Les deux fichiers d'offres sont fusionnés en un seul DataFrame.
    """
    data_dir = Path(data_dir)

    offres_main = pd.read_excel(data_dir / "Offres_ACPE.xlsx", sheet_name="Feuil1")
    offres_ext  = pd.read_excel(data_dir / "Offres_ACPE_Extensions.xlsx", sheet_name="Offres Avril 2026")
    demandeurs  = pd.read_excel(data_dir / "Demandeurs .xlsx", sheet_name="Feuil1")
    appariements = pd.read_excel(data_dir / "Appariement_Demandeurs_Offres.xlsx", sheet_name="Sheet1")

    offres = _merge_offres(offres_main, offres_ext)

    return offres, demandeurs, appariements


def _merge_offres(offres_main: pd.DataFrame, offres_ext: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise et fusionne les deux fichiers d'offres.
    Les offres étendues (avec description) enrichissent les offres principales.
    """
    # --- Normaliser offres_main ---
    offres_main = offres_main.copy()
    offres_main.columns = [_normalize_col(c) for c in offres_main.columns]
    offres_main = offres_main.rename(columns={
        "reference_offre": "job_id",
        "intitule":        "titre",
        "secteur_activite": "secteur",
        "lieu":            "localisation",
        "type_contrat":    "type_contrat",
        "groupe_de_contrat": "groupe_contrat",
        "date_de_publication": "date_publication",
        "type_dentreprise": "type_entreprise",
    })
    offres_main["description"] = ""
    offres_main["profil"]      = ""
    offres_main["competences"] = ""

    # --- Normaliser offres_ext ---
    offres_ext = offres_ext.copy()
    offres_ext.columns = [_normalize_col(c) for c in offres_ext.columns]
    offres_ext = offres_ext.rename(columns={
        "reference":  "job_id",
        "intitule":   "titre",
        "competences": "competences",
    })
    offres_ext["secteur"]       = offres_ext.get("secteur", "")
    offres_ext["localisation"]  = offres_ext.get("localisation", "")
    offres_ext["type_contrat"]  = offres_ext.get("type_contrat", "")
    offres_ext["groupe_contrat"] = ""
    offres_ext["date_publication"] = pd.NaT
    offres_ext["type_entreprise"]  = ""
    offres_ext["poste"]            = ""

    cols = ["job_id", "titre", "secteur", "localisation", "type_contrat",
            "groupe_contrat", "entreprise", "type_entreprise", "poste",
            "date_publication", "description", "profil", "competences"]

    for df in [offres_main, offres_ext]:
        for c in cols:
            if c not in df.columns:
                df[c] = ""

    merged = pd.concat([offres_main[cols], offres_ext[cols]], ignore_index=True)
    # Dédoublonner — les offres ext peuvent avoir des ID en commun avec main
    merged = merged.drop_duplicates(subset="job_id", keep="last")
    merged = merged.reset_index(drop=True)
    return merged


# ---------------------------------------------------------------------------
# Nettoyage textuel
# ---------------------------------------------------------------------------

def clean_text(text) -> str:
    """Normalise une chaîne : lowercase, supprime caractères spéciaux excessifs."""
    if pd.isna(text) or text is None:
        return ""
    text = str(text).strip()
    # Supprimer caractères de contrôle
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    # Réduire espaces multiples
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_location(loc) -> str:
    """Normalise les noms de villes congolaises."""
    if pd.isna(loc):
        return "non déclaré"
    loc = str(loc).strip().lower()
    mapping = {
        "brazza": "brazzaville",
        "bvl":    "brazzaville",
        "pnr":    "pointe-noire",
        "point noire": "pointe-noire",
        "dolisie": "dolisie",
    }
    for key, val in mapping.items():
        if key in loc:
            return val
    return loc if loc else "non déclaré"


# ---------------------------------------------------------------------------
# Construction des profils textuels
# ---------------------------------------------------------------------------

def build_candidate_text(row: pd.Series) -> str:
    """
    Construit un profil textuel synthétique à partir des champs du demandeur.
    Utilisé pour le calcul de similarité.
    """
    fields = [
        row.get("Métier visé / Qualification visée", ""),
        row.get("qualification_metier", ""),
        row.get("Filière / Spécialité", ""),
        row.get("Secteur demandé", ""),
        row.get("Objectif", ""),
        row.get("Qualification", ""),
        row.get("Diplome", ""),
        row.get("niveau_etude", ""),
    ]
    return " ".join([clean_text(f) for f in fields if not _is_empty(f)])


def build_job_text(row: pd.Series) -> str:
    """
    Construit un profil textuel synthétique à partir des champs de l'offre.
    """
    fields = [
        row.get("titre", ""),
        row.get("secteur", ""),
        row.get("description", ""),
        row.get("profil", ""),
        row.get("competences", ""),
    ]
    return " ".join([clean_text(f) for f in fields if not _is_empty(f)])


# ---------------------------------------------------------------------------
# Nettoyage des DataFrames
# ---------------------------------------------------------------------------

def clean_demandeurs(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie le DataFrame des demandeurs."""
    df = df.copy()
    # Supprimer la colonne vide
    if "offre_pertinente " in df.columns:
        df = df.drop(columns=["offre_pertinente "])

    # Normaliser les colonnes textuelles
    text_cols = [
        "Qualification", "Secteur d'activité", "Objectif", "Diplome",
        "niveau_etude", "qualification_metier", "secteur_metier",
        "Filière / Spécialité", "Secteur demandé",
        "Métier visé / Qualification visée", "Mobilité géographique"
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    # Normaliser la mobilité géographique
    if "Mobilité géographique" in df.columns:
        df["mobilite_norm"] = df["Mobilité géographique"].apply(normalize_location)
    else:
        df["mobilite_norm"] = "non déclaré"

    # Construire le profil textuel
    df["text_profile"] = df.apply(build_candidate_text, axis=1)

    return df


def clean_offres(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie le DataFrame des offres."""
    df = df.copy()
    text_cols = ["titre", "secteur", "localisation", "description", "profil", "competences"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    if "localisation" in df.columns:
        df["localisation_norm"] = df["localisation"].apply(normalize_location)
    else:
        df["localisation_norm"] = "non déclaré"

    # Construire le texte de l'offre
    df["text_profile"] = df.apply(build_job_text, axis=1)

    return df


def prepare_ground_truth(appariements: pd.DataFrame) -> dict[str, list[str]]:
    """
    Transforme le fichier d'appariements en dict {id_demandeur: [job_id1, job_id2, job_id3]}.
    """
    gt = {}
    for _, row in appariements.iterrows():
        cid = str(row["id_demandeur"])
        jobs = [str(row["id_offre1"]), str(row["id_offre2"]), str(row["id_offre3"])]
        gt[cid] = [j for j in jobs if j and j != "nan"]
    return gt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_col(col: str) -> str:
    """Normalise un nom de colonne : lowercase, underscores, sans accents."""
    col = col.strip().lower()
    col = re.sub(r"[éèêë]", "e", col)
    col = re.sub(r"[àâä]", "a", col)
    col = re.sub(r"[ùûü]", "u", col)
    col = re.sub(r"[îï]", "i", col)
    col = re.sub(r"[ôö]", "o", col)
    col = re.sub(r"[çc]", "c", col)
    col = re.sub(r"[^a-z0-9_]", "_", col)
    col = re.sub(r"_+", "_", col).strip("_")
    return col


def _is_empty(val) -> bool:
    if val is None:
        return True
    if pd.isna(val):
        return True
    s = str(val).strip().lower()
    return s in ("", "nan", "none", "non déclaré", "non declare", "non declaré")
