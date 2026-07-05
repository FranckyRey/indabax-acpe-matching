# 🎯 Système d'Appariement Emploi — ACPE
### IndabaX Congo 2026 — Hackathon IA & Emploi

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

Prototype fonctionnel de mise en relation intelligente entre **demandeurs d'emploi** et **offres d'emploi** pour l'Agence Congolaise pour l'Emploi (ACPE), développé dans le cadre du Hackathon IndabaX Congo 2026.

---

## 🚀 Fonctionnalités

| Fonctionnalité | Description |
|---|---|
| 🔍 **Recommandations Top-K** | Top-5 et Top-10 offres pour chaque candidat avec score de compatibilité |
| 📊 **Tableau de bord** | KPIs interactifs : candidats, offres, secteurs, géographie |
| 🔎 **Recherche NLP** | Requêtes en langage naturel pour offres et candidats |
| 📈 **Skill Gap** | Identification des compétences manquantes |
| 📋 **Export CSV** | Recommandations au format attendu par le jury |

---

## 🧠 Architecture du Moteur

Le moteur d'appariement utilise un **score hybride** :

```
Score = α × Similarité_TF-IDF + β × Bonus_Secteur + γ × Bonus_Géo
      = 70% × sim_cosinus + 20% × correspondance_secteur + 10% × correspondance_geo
```

**Vectorisation TF-IDF** sur les profils textuels construits à partir :
- **Candidat** : Métier visé + Qualification + Filière + Secteur demandé + Diplôme
- **Offre** : Titre + Secteur + Description + Profil recherché + Compétences

---

## 📦 Installation

```bash
# Cloner le dépôt
git clone https://github.com/votre-equipe/indabax-acpe-matching.git
cd indabax-acpe-matching

# Créer un environnement virtuel
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Installer les dépendances
pip install -r requirements.txt
```

---

## 📂 Structure du Projet

```
InDaBaX/
├── data/
│   ├── raw/                    # Données brutes Excel (non versionnées)
│   ├── processed/              # Données nettoyées CSV
│   └── embeddings/             # Modèle TF-IDF sérialisé
├── src/
│   ├── preprocessing.py        # Chargement et nettoyage des données
│   ├── matching_engine.py      # Moteur d'appariement hybride
│   ├── evaluation.py           # Métriques P@K, R@K, NDCG@K
│   ├── skill_gap.py            # Analyse des écarts de compétences
│   └── search.py               # Recherche en langage naturel
├── app/
│   └── streamlit_app.py        # Interface Streamlit multi-pages
├── scripts/
│   └── generate_recommendations.py  # Script de génération des recommandations
├── tests/                      # Tests unitaires
├── outputs/                    # Recommandations générées (CSV)
└── requirements.txt
```

---

## 🗂️ Données

| Fichier | Description | Lignes |
|---|---|---|
| `Offres_ACPE.xlsx` | Offres d'emploi principales | 2 535 |
| `Offres_ACPE_Extensions.xlsx` | Offres avec description complète | 143 |
| `Demandeurs .xlsx` | Profils des demandeurs d'emploi | 41 298 |
| `Appariement_Demandeurs_Offres.xlsx` | Ground truth (3 offres/candidat) | 41 298 |

> ⚠️ Les fichiers de données ne sont pas versionnés (`.gitignore`). Placer les fichiers Excel dans `data/raw/`.

---

## ⚡ Utilisation

### 1. Générer les recommandations

```bash
python scripts/generate_recommendations.py
```

Produit :
- `outputs/recommendations.csv` — recommandations au format `(candidate_id, rank, job_id, score)`
- `outputs/evaluation_report.csv` — rapport Precision/Recall/NDCG@5 et @10

### 2. Lancer l'interface

```bash
streamlit run app/streamlit_app.py
```

### 3. Lancer les tests

```bash
python -m pytest tests/ -v
```

---

## 📊 Résultats d'Évaluation

| Métrique | Valeur |
|---|---|
| Precision@5 | **27.65%** |
| Recall@5 | **46.08%** |
| NDCG@5 | **43.61%** |
| Precision@10 | **17.28%** |
| NDCG@10 | **48.84%** |

---

## 👥 Équipe

| Membre | Rôle |
|---|---|
| — | — |

---

## 📝 Licence

MIT — IndabaX Congo 2026
