"""
streamlit_app.py
================
Interface principale du Système d'Appariement ACPE.
Pages : Accueil | Recommandations | Tableau de Bord | Recherche NLP | Skill Gap
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import pickle
import time

# Ajouter le répertoire racine
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import load_data, clean_demandeurs, clean_offres, prepare_ground_truth
from src.matching_engine import MatchingEngine
from src.skill_gap import compute_skill_gap
from src.search import NaturalLanguageSearch

# ---------------------------------------------------------------------------
# Configuration de la page
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ACPE — Système d'Appariement Emploi",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS personnalisé
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px;
    color: white !important;
}
div[data-testid="metric-container"] label {
    color: #94a3b8 !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #38bdf8 !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
}

/* Titres */
h1 { color: #f1f5f9 !important; }
h2, h3 { color: #e2e8f0 !important; }

/* Score badge */
.score-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.85rem;
}
.score-high   { background: #064e3b; color: #6ee7b7; }
.score-medium { background: #78350f; color: #fde68a; }
.score-low    { background: #7f1d1d; color: #fca5a5; }

/* Cards offres */
.job-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    transition: border-color 0.2s;
}
.job-card:hover { border-color: #38bdf8; }
.job-rank {
    font-size: 1.4rem;
    font-weight: 700;
    color: #38bdf8;
}

/* Streamlit overrides */
.stSelectbox label, .stTextInput label, .stTextArea label {
    color: #94a3b8 !important;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Chargement des données (mis en cache)
# ---------------------------------------------------------------------------
DATA_DIR   = ROOT / "data" / "raw"
MODEL_PATH = ROOT / "data" / "embeddings" / "engine.pkl"


@st.cache_resource(show_spinner="Chargement des données...")
def load_all_data():
    offres_raw, demandeurs_raw, appariements = load_data(DATA_DIR)
    offres     = clean_offres(offres_raw)
    demandeurs = clean_demandeurs(demandeurs_raw)
    gt         = prepare_ground_truth(appariements)
    return offres, demandeurs, gt


@st.cache_resource(show_spinner="Chargement du moteur d'appariement...")
def load_engine(offres: pd.DataFrame):
    """Charge ou entraîne le moteur."""
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            engine = pickle.load(f)
    else:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        engine = MatchingEngine(alpha=0.70, beta=0.20, gamma=0.10)
        engine.fit(offres)
        engine.save(MODEL_PATH)
    return engine


# ---------------------------------------------------------------------------
# Sidebar — Navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎯 ACPE")
    st.markdown("**Système d'Appariement Emploi**")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        options=[
            "🏠 Accueil",
            "🔍 Recommandations",
            "📊 Tableau de Bord",
            "🔎 Recherche Intelligente",
            "📈 Analyse Skill Gap",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<small>IndabaX Congo 2026<br>Hackathon IA & Emploi</small>",
        unsafe_allow_html=True,
    )


# Chargement global
try:
    offres, demandeurs, ground_truth = load_all_data()
    engine = load_engine(offres)
    nlp_search = NaturalLanguageSearch(engine)
    data_loaded = True
except Exception as e:
    data_loaded = False
    st.error(f"❌ Erreur de chargement : {e}\n\nVérifiez que les données sont dans `data/raw/`")


# ===========================================================================
# PAGE 1 — ACCUEIL
# ===========================================================================
if page == "🏠 Accueil":
    st.markdown("# 🎯 Système d'Appariement Emploi — ACPE")
    st.markdown(
        "Bienvenue sur le prototype de mise en relation intelligente entre **demandeurs d'emploi** "
        "et **offres d'emploi** de l'Agence Congolaise pour l'Emploi."
    )

    if data_loaded:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("👤 Candidats", f"{len(demandeurs):,}")
        with col2:
            st.metric("💼 Offres d'emploi", f"{len(offres):,}")
        with col3:
            n_sectors = offres["secteur"].nunique()
            st.metric("🏭 Secteurs", n_sectors)
        with col4:
            n_cities = offres["localisation_norm"].nunique()
            st.metric("📍 Villes", n_cities)

    st.markdown("---")
    st.markdown("### 🚀 Fonctionnalités disponibles")
    cols = st.columns(2)
    features = [
        ("🔍 Recommandations", "Top-5 et Top-10 offres pour chaque candidat avec score de compatibilité"),
        ("📊 Tableau de Bord", "KPIs, statistiques et visualisations interactives pour les conseillers"),
        ("🔎 Recherche Intelligente", "Retrouvez offres et candidats par requête en langage naturel"),
        ("📈 Analyse Skill Gap", "Identifiez les compétences manquantes pour chaque recommandation"),
    ]
    for i, (title, desc) in enumerate(features):
        with cols[i % 2]:
            st.info(f"**{title}**\n\n{desc}")

    st.markdown("---")
    st.markdown("### 🧠 Architecture du moteur")
    st.markdown("""
    Le moteur d'appariement utilise un **score hybride** :

    ```
    Score = 70% × Similarité TF-IDF (profil candidat ↔ offre)
           + 20% × Correspondance sectorielle
           + 10% × Correspondance géographique
    ```

    **Méthode TF-IDF** : les profils textuels des candidats (métier visé, qualifications, filière...)
    et des offres (titre, secteur, description) sont vectorisés et comparés par similarité cosinus.
    """)


# ===========================================================================
# PAGE 2 — RECOMMANDATIONS
# ===========================================================================
elif page == "🔍 Recommandations":
    st.markdown("# 🔍 Recommandations Personnalisées")

    if not data_loaded:
        st.stop()

    tab1, tab2 = st.tabs(["Par ID Candidat", "Profil Libre"])

    # --- Onglet 1 : par ID candidat ---
    with tab1:
        col_search, col_k = st.columns([3, 1])
        with col_search:
            candidate_id = st.selectbox(
                "Sélectionner un candidat",
                options=demandeurs["Matricule"].astype(str).tolist(),
                index=0,
            )
        with col_k:
            k = st.selectbox("Top-K", options=[5, 10], index=0)

        if candidate_id:
            candidate_row = demandeurs[demandeurs["Matricule"].astype(str) == candidate_id].iloc[0]
            _show_candidate_and_recs(candidate_row, engine, offres, ground_truth, k, candidate_id)

    # --- Onglet 2 : profil libre ---
    with tab2:
        st.markdown("#### Saisir un profil manuellement")
        col1, col2 = st.columns(2)
        with col1:
            metier_vise  = st.text_input("Métier visé", placeholder="Ex: Data Analyst")
            secteur      = st.text_input("Secteur demandé", placeholder="Ex: Technologie")
            specialite   = st.text_input("Filière / Spécialité", placeholder="Ex: Informatique")
        with col2:
            diplome      = st.text_input("Diplôme", placeholder="Ex: Licence en Informatique")
            niveau       = st.selectbox("Niveau d'étude", ["Bac", "Bac+2", "Bac+3", "Bac+5", "Doctorat"])
            mobilite     = st.text_input("Mobilité géographique", placeholder="Ex: Brazzaville")
        k_libre = st.selectbox("Top-K ", options=[5, 10], index=0)

        if st.button("🔍 Trouver les meilleures offres", type="primary"):
            from src.preprocessing import build_candidate_text, normalize_location, clean_text
            row = pd.Series({
                "Métier visé / Qualification visée": metier_vise,
                "Secteur demandé": secteur,
                "Filière / Spécialité": specialite,
                "Diplome": diplome,
                "niveau_etude": niveau,
                "Qualification": "",
                "Objectif": "Emploi",
                "mobilite_norm": normalize_location(mobilite),
                "Mobilité géographique": mobilite,
            })
            row["text_profile"] = build_candidate_text(row)
            _show_recs_only(row, engine, offres, k_libre)


def _show_candidate_and_recs(candidate_row, engine, offres, ground_truth, k, candidate_id):
    """Affiche le profil candidat et ses recommandations."""
    st.markdown("---")
    # Profil candidat
    with st.expander("👤 Profil du candidat", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"**Matricule :** `{candidate_row.get('Matricule', '')}`")
            st.markdown(f"**Âge :** {candidate_row.get('Age', 'N/A')}")
            st.markdown(f"**Genre :** {candidate_row.get('Genre', 'N/A')}")
        with c2:
            st.markdown(f"**Niveau :** {candidate_row.get('niveau_etude', 'N/A')}")
            st.markdown(f"**Diplôme :** {candidate_row.get('Diplome', 'N/A')}")
            st.markdown(f"**Filière :** {candidate_row.get('Filière / Spécialité', 'N/A')}")
        with c3:
            st.markdown(f"**Métier visé :** {candidate_row.get('Métier visé / Qualification visée', 'N/A')}")
            st.markdown(f"**Secteur :** {candidate_row.get('Secteur demandé', 'N/A')}")
            st.markdown(f"**Mobilité :** {candidate_row.get('Mobilité géographique', 'N/A')}")

    # Offres pertinentes du ground truth
    relevant_ids = ground_truth.get(str(candidate_id), [])
    if relevant_ids:
        st.info(f"✅ **Offres de référence (ground truth) :** {', '.join(relevant_ids)}")

    # Recommandations
    st.markdown(f"### 🏆 Top-{k} Recommandations")
    recs = engine.recommend_top_k(candidate_row, k=k, include_details=True)
    _render_recommendations(recs, relevant_ids)


def _show_recs_only(candidate_row, engine, offres, k):
    """Affiche uniquement les recommandations (sans ground truth)."""
    st.markdown(f"### 🏆 Top-{k} Recommandations")
    recs = engine.recommend_top_k(candidate_row, k=k, include_details=True)
    _render_recommendations(recs, [])


def _render_recommendations(recs, relevant_ids):
    """Affiche les recommandations sous forme de tableau + cartes."""
    # Tableau synthétique
    table_data = []
    for r in recs:
        in_gt = "✅" if r["job_id"] in relevant_ids else ""
        table_data.append({
            "Rang": r["rank"],
            "Référence": r["job_id"],
            "Poste": r["titre"],
            "Secteur": r["secteur"],
            "Lieu": r["localisation"],
            "Score": f"{r['score']:.2%}",
            "Réf. ✓": in_gt,
        })
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    # Graphique en barres des scores
    fig = px.bar(
        pd.DataFrame({"Poste": [r["titre"] for r in recs], "Score": [r["score"] for r in recs]}),
        x="Score", y="Poste", orientation="h",
        color="Score", color_continuous_scale="Blues",
        title="Scores de compatibilité",
        template="plotly_dark",
    )
    fig.update_layout(height=max(300, len(recs) * 45), yaxis={"autorange": "reversed"})
    st.plotly_chart(fig, use_container_width=True)

    # Détail d'une offre
    selected = st.selectbox(
        "Voir le détail d'une offre",
        options=[f"#{r['rank']} — {r['titre']}" for r in recs],
    )
    idx = int(selected.split("—")[0].replace("#", "").strip()) - 1
    rec = recs[idx]
    with st.expander(f"💼 Détail : {rec['titre']}", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Référence :** `{rec['job_id']}`")
            st.markdown(f"**Entreprise :** {rec.get('entreprise', 'N/A')}")
            st.markdown(f"**Secteur :** {rec['secteur']}")
            st.markdown(f"**Localisation :** {rec['localisation']}")
            st.markdown(f"**Type contrat :** {rec.get('type_contrat', 'N/A')}")
        with c2:
            st.markdown("**Décomposition du score :**")
            if "score_semantique" in rec:
                fig2 = go.Figure(go.Bar(
                    x=[rec["score_semantique"], rec["score_secteur"], rec["score_geo"]],
                    y=["Similarité textuelle", "Secteur", "Géographie"],
                    orientation="h",
                    marker_color=["#38bdf8", "#818cf8", "#34d399"],
                ))
                fig2.update_layout(template="plotly_dark", height=200, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig2, use_container_width=True)


# ===========================================================================
# PAGE 3 — TABLEAU DE BORD
# ===========================================================================
elif page == "📊 Tableau de Bord":
    st.markdown("# 📊 Tableau de Bord Décisionnel")

    if not data_loaded:
        st.stop()

    # KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("👤 Candidats", f"{len(demandeurs):,}")
    with col2:
        st.metric("💼 Offres", f"{len(offres):,}")
    with col3:
        st.metric("🏭 Secteurs offres", offres["secteur"].nunique())
    with col4:
        st.metric("🎓 Niveaux d'étude", demandeurs["niveau_etude"].nunique())
    with col5:
        avg_age = demandeurs["Age"].mean()
        st.metric("📅 Âge moyen", f"{avg_age:.0f} ans")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    # Top secteurs (offres)
    with col_a:
        st.markdown("### 🏭 Top 10 Secteurs (Offres)")
        top_sectors = offres["secteur"].value_counts().head(10).reset_index()
        top_sectors.columns = ["Secteur", "Nombre"]
        fig = px.bar(top_sectors, x="Nombre", y="Secteur", orientation="h",
                     color="Nombre", color_continuous_scale="Blues",
                     template="plotly_dark")
        fig.update_layout(yaxis={"autorange": "reversed"}, height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Top métiers visés (candidats)
    with col_b:
        st.markdown("### 🎯 Top 10 Métiers Visés (Candidats)")
        top_metiers = demandeurs["Métier visé / Qualification visée"].value_counts().head(10).reset_index()
        top_metiers.columns = ["Métier", "Nombre"]
        fig = px.bar(top_metiers, x="Nombre", y="Métier", orientation="h",
                     color="Nombre", color_continuous_scale="Purples",
                     template="plotly_dark")
        fig.update_layout(yaxis={"autorange": "reversed"}, height=400)
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)

    # Distribution géographique des offres
    with col_c:
        st.markdown("### 📍 Répartition Géographique (Offres)")
        geo_offres = offres["localisation_norm"].value_counts().head(15).reset_index()
        geo_offres.columns = ["Ville", "Offres"]
        fig = px.pie(geo_offres, values="Offres", names="Ville",
                     template="plotly_dark", hole=0.4)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    # Distribution niveaux d'étude (candidats)
    with col_d:
        st.markdown("### 🎓 Niveaux d'Étude (Candidats)")
        niveaux = demandeurs["niveau_etude"].value_counts().reset_index()
        niveaux.columns = ["Niveau", "Candidats"]
        fig = px.bar(niveaux, x="Niveau", y="Candidats",
                     color="Candidats", color_continuous_scale="Teal",
                     template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    col_e, col_f = st.columns(2)

    # Genre
    with col_e:
        st.markdown("### 👥 Répartition par Genre")
        genre = demandeurs["Genre"].value_counts().reset_index()
        genre.columns = ["Genre", "Nombre"]
        fig = px.pie(genre, values="Nombre", names="Genre",
                     color_discrete_sequence=["#38bdf8", "#f472b6"],
                     template="plotly_dark", hole=0.3)
        st.plotly_chart(fig, use_container_width=True)

    # Distribution des âges
    with col_f:
        st.markdown("### 📅 Distribution des Âges")
        fig = px.histogram(demandeurs, x="Age", nbins=30,
                           color_discrete_sequence=["#818cf8"],
                           template="plotly_dark")
        fig.update_layout(bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)

    # Type de contrat
    st.markdown("### 📋 Types de Contrat (Offres)")
    contrats = offres["type_contrat"].value_counts().reset_index()
    contrats.columns = ["Type", "Nombre"]
    fig = px.bar(contrats, x="Type", y="Nombre",
                 color="Nombre", color_continuous_scale="Oranges",
                 template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# PAGE 4 — RECHERCHE INTELLIGENTE
# ===========================================================================
elif page == "🔎 Recherche Intelligente":
    st.markdown("# 🔎 Recherche en Langage Naturel")
    st.markdown(
        "Recherchez des **offres** ou des **candidats** à partir d'une requête libre, "
        "sans avoir besoin de correspondance exacte de mots-clés."
    )

    if not data_loaded:
        st.stop()

    search_type = st.radio(
        "Type de recherche",
        options=["🔍 Offres d'emploi", "👤 Candidats"],
        horizontal=True,
    )

    query = st.text_area(
        "Votre requête",
        placeholder=(
            "Ex: 'Je recherche un développeur Python à Brazzaville'\n"
            "Ex: 'Candidat en comptabilité avec mobilité nationale'"
        ),
        height=80,
    )

    col_n, _ = st.columns([1, 3])
    with col_n:
        n_results = st.slider("Nombre de résultats", 3, 20, 5)

    if st.button("🔍 Rechercher", type="primary") and query.strip():
        with st.spinner("Recherche en cours..."):
            if search_type == "🔍 Offres d'emploi":
                results = nlp_search.search_jobs(query, n=n_results)
                if results:
                    st.markdown(f"### ✅ {len(results)} offres trouvées")
                    for r in results:
                        with st.container():
                            st.markdown(
                                f"**#{r['rank']} — {r['titre']}** "
                                f"| `{r['job_id']}` "
                                f"| Score: **{r['score']:.2%}**\n\n"
                                f"🏭 {r['secteur']} | 📍 {r['localisation']} "
                                f"| 🏢 {r['entreprise']} | 📄 {r['type_contrat']}"
                            )
                            if r.get("description"):
                                st.caption(r["description"])
                            st.divider()
                else:
                    st.warning("Aucun résultat trouvé.")

            else:  # Candidats
                cand_matrix = nlp_search.build_candidate_matrix(demandeurs)
                results = nlp_search.search_candidates(query, demandeurs, cand_matrix, n=n_results)
                if results:
                    st.markdown(f"### ✅ {len(results)} candidats trouvés")
                    df = pd.DataFrame(results)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.warning("Aucun candidat trouvé.")


# ===========================================================================
# PAGE 5 — SKILL GAP
# ===========================================================================
elif page == "📈 Analyse Skill Gap":
    st.markdown("# 📈 Analyse des Écarts de Compétences")
    st.markdown(
        "Identifiez les **compétences manquantes** chez un candidat par rapport aux exigences d'une offre."
    )

    if not data_loaded:
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        candidate_id_sg = st.selectbox(
            "Candidat",
            options=demandeurs["Matricule"].astype(str).tolist(),
            key="sg_candidate",
        )
    with col2:
        job_id_sg = st.selectbox(
            "Offre d'emploi",
            options=offres["job_id"].astype(str).tolist(),
            key="sg_job",
        )

    if st.button("📈 Analyser l'écart", type="primary"):
        cand = demandeurs[demandeurs["Matricule"].astype(str) == candidate_id_sg].iloc[0]
        job  = offres[offres["job_id"].astype(str) == job_id_sg].iloc[0]

        gap = compute_skill_gap(cand["text_profile"], job["text_profile"])

        # Score de couverture
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("✅ Compétences communes", len(gap["skills_matched"]))
        with col_s2:
            st.metric("❌ Compétences manquantes", len(gap["skills_missing"]))
        with col_s3:
            st.metric("📊 Taux de couverture", f"{gap['compatibility_pct']}%")

        # Graphique gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=gap["compatibility_pct"],
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Compatibilité compétences", "font": {"color": "white"}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#38bdf8"},
                "steps": [
                    {"range": [0, 40],  "color": "#7f1d1d"},
                    {"range": [40, 70], "color": "#78350f"},
                    {"range": [70, 100],"color": "#064e3b"},
                ],
            },
            number={"suffix": "%", "font": {"color": "white"}},
        ))
        fig.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig, use_container_width=True)

        # Détails
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("### ✅ Compétences correspondantes")
            if gap["skills_matched"]:
                for s in gap["skills_matched"]:
                    st.success(f"✓ {s}")
            else:
                st.info("Aucune compétence commune détectée")

        with col_b:
            st.markdown("### ❌ Compétences manquantes")
            if gap["skills_missing"]:
                for s in gap["skills_missing"]:
                    st.error(f"✗ {s}")
            else:
                st.success("Aucune compétence manquante !")

        with col_c:
            st.markdown("### ➕ Compétences supplémentaires")
            if gap["skills_extra"]:
                for s in gap["skills_extra"]:
                    st.info(f"+ {s}")
            else:
                st.caption("Aucune compétence supplémentaire")

        # Recommandation formation
        if gap["skills_missing"]:
            st.markdown("---")
            st.markdown("### 💡 Recommandations de formation")
            st.markdown(
                "Pour améliorer votre profil, nous vous recommandons de développer "
                "les compétences suivantes :"
            )
            for s in gap["skills_missing"]:
                st.markdown(f"- **{s.capitalize()}** — Formation recommandée")
