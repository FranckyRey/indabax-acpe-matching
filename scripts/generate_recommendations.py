"""
generate_recommendations.py
===========================
Script principal : entraîne le moteur et génère les recommandations
Top-5 et Top-10 pour tous les candidats, puis évalue les performances.

Usage :
    python scripts/generate_recommendations.py
"""

import sys
import time
import pandas as pd
from pathlib import Path

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import load_data, clean_demandeurs, clean_offres, prepare_ground_truth
from src.matching_engine import MatchingEngine
from src.evaluation import evaluate_from_dataframe, print_evaluation_report

DATA_DIR    = Path(__file__).parent.parent / "data" / "raw"
OUTPUT_DIR  = Path(__file__).parent.parent / "outputs"
MODEL_PATH  = Path(__file__).parent.parent / "data" / "embeddings" / "engine.pkl"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print("  SYSTÈME D'APPARIEMENT ACPE — Génération Recommandations")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Chargement des données
    # ------------------------------------------------------------------
    print("\n[1/5] Chargement des données...")
    t0 = time.time()
    offres_raw, demandeurs_raw, appariements = load_data(DATA_DIR)
    print(f"  [OK] {len(offres_raw)} offres | {len(demandeurs_raw)} candidats "
          f"| {len(appariements)} appariements  ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------
    # 2. Nettoyage
    # ------------------------------------------------------------------
    print("\n[2/5] Nettoyage et préparation...")
    t0 = time.time()
    offres     = clean_offres(offres_raw)
    demandeurs = clean_demandeurs(demandeurs_raw)
    gt         = prepare_ground_truth(appariements)
    print(f"  [OK] Données nettoyées  ({time.time()-t0:.1f}s)")

    # Sauvegarder les données nettoyées
    processed_dir = Path(__file__).parent.parent / "data" / "processed"
    processed_dir.mkdir(exist_ok=True)
    offres.to_csv(processed_dir / "offres_clean.csv", index=False)
    demandeurs.to_csv(processed_dir / "demandeurs_clean.csv", index=False)
    print(f"  [OK] Données nettoyées sauvegardées dans data/processed/")

    # ------------------------------------------------------------------
    # 3. Entraînement du moteur
    # ------------------------------------------------------------------
    print("\n[3/5] Entraînement du moteur d'appariement...")
    t0 = time.time()
    engine = MatchingEngine(alpha=0.70, beta=0.20, gamma=0.10)
    engine.fit(offres)
    engine.save(MODEL_PATH)
    print(f"  [OK] Moteur TF-IDF entraîné sur {len(offres)} offres  ({time.time()-t0:.1f}s)")
    print(f"  [OK] Modèle sauvegardé : {MODEL_PATH}")

    # ------------------------------------------------------------------
    # 4. Génération des recommandations (Top-10, contient Top-5)
    # ------------------------------------------------------------------
    print("\n[4/5] Génération des recommandations Top-10...")
    t0 = time.time()
    recs_df = engine.recommend_all(demandeurs, k=10, id_col="Matricule")
    recs_df.to_csv(OUTPUT_DIR / "recommendations.csv", index=False)
    print(f"  [OK] {len(recs_df)} lignes générées  ({time.time()-t0:.1f}s)")
    print(f"  [OK] Sauvegardé : outputs/recommendations.csv")

    # ------------------------------------------------------------------
    # 5. Évaluation
    # ------------------------------------------------------------------
    print("\n[5/5] Évaluation des performances...")
    results = evaluate_from_dataframe(recs_df, gt, k_values=[5, 10])
    print_evaluation_report(results)

    # Sauvegarder le rapport d'évaluation
    report_rows = []
    for k, metrics in results.items():
        for name, value in metrics.items():
            report_rows.append({"metrique": name, "valeur": value, "pourcentage": f"{value*100:.2f}%"})
    pd.DataFrame(report_rows).to_csv(OUTPUT_DIR / "evaluation_report.csv", index=False)
    print(f"  [OK] Rapport sauvegardé : outputs/evaluation_report.csv")

    print("\n[OK] Terminé avec succès !")


if __name__ == "__main__":
    main()
