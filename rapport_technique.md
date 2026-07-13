# Rapport Technique
## Système Intelligent d'Appariement Demandeurs d'Emploi — Offres d'Emploi
### Hackathon IndabaX Congo 2026 — Agence Congolaise pour l'Emploi (ACPE)

---

> **Résumé exécutif**
> Ce rapport présente l'architecture, les choix techniques et les résultats d'un prototype de système d'appariement intelligent entre demandeurs d'emploi et offres d'emploi, développé dans le cadre du Hackathon IndabaX Congo 2026. Le système repose sur un moteur hybride combinant la similarité sémantique TF-IDF avec des bonifications catégorielles (secteur, géographie). Évalué sur 41 298 candidats réels et 2 678 offres d'emploi, il atteint un **NDCG@10 de 48,84 %** et un **Recall@10 de 57,58 %**, démontrant sa capacité à identifier les meilleures correspondances dans un contexte de données hétérogènes et incomplètes.

---

## Table des Matières

1. [Introduction et Contexte](#1-introduction-et-contexte)
2. [Exploration et Préparation des Données](#2-exploration-et-préparation-des-données)
3. [Architecture du Moteur d'Appariement](#3-architecture-du-moteur-dappariement)
4. [Résultats d'Évaluation](#4-résultats-dévaluation)
5. [Interface et Tableau de Bord](#5-interface-et-tableau-de-bord)
6. [Perspectives et Améliorations](#6-perspectives-et-améliorations)

---

## 1. Introduction et Contexte

### 1.1 Problématique

L'Agence Congolaise pour l'Emploi (ACPE) joue un rôle pivot dans la mise en relation entre demandeurs d'emploi et entreprises au Congo-Brazzaville. Chaque année, plusieurs dizaines de milliers de demandeurs s'enregistrent auprès de l'agence, tandis que des milliers d'offres d'emploi sont collectées.

Le processus de mise en correspondance est aujourd'hui **essentiellement manuel** : un conseiller examine manuellement les profils de candidats et les compare aux offres disponibles pour suggérer des appariements. Ce processus est :

- **Chronophage** : un conseiller ne peut traiter qu'un nombre limité de dossiers par jour.
- **Subjectif** : les critères de correspondance varient d'un conseiller à l'autre.
- **Peu scalable** : il ne peut pas s'adapter à la croissance du volume de données.
- **Sous-optimal** : des correspondances pertinentes peuvent être manquées, notamment entre secteurs voisins.

### 1.2 Objectif du Hackathon

Le défi consiste à développer un **système intelligent d'appariement** capable de :

1. Analyser automatiquement les profils des candidats et les caractéristiques des offres.
2. Produire une liste ordonnée des **Top-K offres les plus compatibles** pour chaque candidat (K = 5 et K = 10).
3. Expliquer les recommandations de manière intelligible pour les conseillers.
4. Fournir un **tableau de bord analytique** pour le pilotage des activités de placement.
5. *(Bonus)* Proposer une recherche en langage naturel et une analyse des écarts de compétences (*Skill Gap*).

### 1.3 Contraintes Techniques

Le système a été développé sous les contraintes suivantes :

- **Volume de données** : 41 298 demandeurs, 2 678 offres, soit plus de 110 millions de paires candidat-offre potentielles.
- **Qualité des données** : descriptions textuelles absentes pour la majorité des offres principales, valeurs manquantes abondantes, encodages hétérogènes.
- **Déployabilité** : le prototype doit être accessible en ligne et utilisable par des non-techniciens.
- **Reproductibilité** : le code source est versionné sous Git et hébergé sur GitHub.

---

## 2. Exploration et Préparation des Données

### 2.1 Sources de Données

Le système s'appuie sur **quatre fichiers Excel** fournis par l'ACPE :

| Fichier | Description | Lignes | Taille |
|---|---|---|---|
| `Offres_ACPE.xlsx` | Offres d'emploi principales | 2 535 | 0,14 Mo |
| `Offres_ACPE_Extensions.xlsx` | Offres enrichies avec descriptions | 143 | 0,03 Mo |
| `Demandeurs .xlsx` | Profils des demandeurs d'emploi | 41 298 | 2,61 Mo |
| `Appariement_Demandeurs_Offres.xlsx` | Vérité terrain (3 offres/candidat) | 41 298 | 0,96 Mo |

### 2.2 Analyse Exploratoire

L'analyse exploratoire a révélé plusieurs caractéristiques structurantes des données :

#### Structure des offres d'emploi

Les offres principales (`Offres_ACPE.xlsx`) contiennent les colonnes : `Référence Offre`, `Intitulé`, `Secteur d'Activité`, `Lieu`, `Type Contrat`, `Groupe de Contrat`, `Date de Publication`, `Type d'Entreprise`.

**Problème identifié** : les colonnes `description`, `profil` et `compétences` sont **entièrement vides** dans les offres principales. Seules les 143 offres du fichier d'extension disposent de descriptions textuelles. Cela constitue la principale contrainte du projet.

#### Structure des demandeurs

Les profils candidats contiennent : `Matricule`, `Genre`, `Âge`, `Niveau d'étude`, `Qualification`, `Diplôme`, `Filière / Spécialité`, `Métier visé / Qualification visée`, `Secteur demandé`, `Mobilité géographique`, `Objectif`.

**Observations** :
- La colonne `offre_pertinente` est 100 % nulle → ignorée.
- Les champs textuels contiennent des abréviations et variantes orthographiques (ex: "Bvl", "Brazza" pour Brazzaville).
- Taux de valeurs manquantes variable selon les colonnes (entre 5 % et 45 %).

#### Distribution géographique

La majorité des offres se concentre sur deux grandes villes :
- **Brazzaville** : ~60 % des offres
- **Pointe-Noire** : ~35 % des offres
- Autres villes : ~5 % (Dolisie, Ouesso, etc.)

#### Secteurs d'activité les plus représentés

Les secteurs les plus fréquents dans les offres incluent les BTP/Génie Civil, le Commerce, les Services, les Télécommunications et le Secteur Bancaire/Finance.

### 2.3 Pipeline de Préparation des Données

La préparation des données est implémentée dans `src/preprocessing.py` et suit les étapes suivantes :

#### Étape 1 : Chargement et fusion des offres

Les deux fichiers d'offres sont normalisés (renommage des colonnes, harmonisation des types) puis fusionnés. La déduplication est effectuée sur la clé `job_id` en conservant la version enrichie lorsqu'elle existe :

```python
merged = pd.concat([offres_main[cols], offres_ext[cols]], ignore_index=True)
merged = merged.drop_duplicates(subset="job_id", keep="last")
```

Résultat : **2 678 offres uniques** après fusion.

#### Étape 2 : Nettoyage textuel

La fonction `clean_text()` applique les transformations suivantes :
- Conversion en chaîne de caractères et suppression des espaces en début/fin.
- Suppression des caractères de contrôle (`\x00`–`\x1f`).
- Réduction des espaces multiples (`\s+` → espace unique).

#### Étape 3 : Normalisation géographique

La fonction `normalize_location()` mappe les abréviations courantes vers les noms normalisés :

```
"Bvl", "Brazza" → "Brazzaville"
"Pnr", "Point Noire" → "Pointe-Noire"
```

Toute valeur absente ou non reconnue est remplacée par `"non déclaré"`.

#### Étape 4 : Construction des profils textuels

C'est l'étape la plus critique du pipeline. Chaque candidat et chaque offre sont représentés par un **profil textuel synthétique**, obtenu en concaténant les champs disponibles :

**Profil candidat** (`build_candidate_text`):
```
[Métier visé] + [Qualification métier] + [Filière/Spécialité]
+ [Secteur demandé] + [Objectif] + [Qualification] + [Diplôme] + [Niveau étude]
```

**Profil offre** (`build_job_text`):
```
[Titre] + [Secteur] + [Description] + [Profil recherché] + [Compétences]
```

Les valeurs nulles, `"nan"`, `"non déclaré"` sont filtrées par la fonction `_is_empty()` avant concaténation, garantissant que le profil final ne contient que de l'information réelle.

#### Étape 5 : Préparation de la vérité terrain

Le fichier d'appariements est transformé en dictionnaire Python `{id_demandeur: [id_offre1, id_offre2, id_offre3]}` pour l'évaluation :

```python
gt = {str(row["id_demandeur"]): [str(row["id_offre1"]), ...] for _, row in appariements.iterrows()}
```

---

## 3. Architecture du Moteur d'Appariement

### 3.1 Vue d'Ensemble

Le moteur d'appariement est implémenté dans `src/matching_engine.py` sous forme d'une classe `MatchingEngine`. Son architecture repose sur un **score hybride** combinant trois composantes :

```
Score_final(c, o) = α × Sim_TF-IDF(c, o) + β × Bonus_Secteur(c, o) + γ × Bonus_Géo(c, o)
```

Avec les pondérations calibrées par expérimentation :

| Composante | Symbole | Poids | Justification |
|---|---|---|---|
| Similarité sémantique TF-IDF | α | **0,70** | Signal principal, capture la cohérence du profil |
| Correspondance sectorielle | β | **0,20** | Fort indicateur de pertinence métier |
| Correspondance géographique | γ | **0,10** | Critère secondaire (mobilité variable) |

### 3.2 Composante 1 — Similarité Sémantique TF-IDF

#### Choix de l'algorithme

**TF-IDF (Term Frequency — Inverse Document Frequency)** a été retenu comme méthode de vectorisation pour les raisons suivantes :

1. **Robustesse aux données courtes** : les profils textuels (quelques dizaines de mots) sont trop courts pour que des modèles de plongements denses (Word2Vec, BERT) soient efficacement entraînés.
2. **Pas besoin de modèle pré-entraîné** : le TF-IDF s'entraîne directement sur le corpus des offres, sans nécessiter de corpus externe en français congolais.
3. **Interprétabilité** : les termes à fort TF-IDF peuvent être inspectés et expliqués aux conseillers.
4. **Efficacité computationnelle** : la représentation sparse permet de calculer les similarités pour 41 298 candidats × 2 678 offres en moins de 8 minutes.

#### Paramétrage du vectoriseur

```python
TfidfVectorizer(
    max_features=15_000,   # Vocabulaire limité aux 15 000 termes les plus discriminants
    ngram_range=(1, 2),    # Unigrammes + bigrammes (ex: "data analyst")
    sublinear_tf=True,     # TF logarithmique pour atténuer les termes très fréquents
    min_df=1,              # Conserver tous les termes (corpus spécialisé)
)
```

Le choix des **bigrammes** (`ngram_range=(1,2)`) est particulièrement important : il permet de capturer des expressions composées typiques des métiers (`"génie civil"`, `"data analyst"`, `"bac +5"`) que les unigrammes seuls ne peuvent pas distinguer.

#### Calcul de la similarité

La similarité entre un candidat `c` et une offre `o` est calculée par la **similarité cosinus** entre leurs représentations TF-IDF :

```
Sim_TF-IDF(c, o) = cos(v_c, v_o) = (v_c · v_o) / (||v_c|| × ||v_o||)
```

Cette mesure est normalisée dans [0, 1], indépendante de la longueur des profils, et s'implémente efficacement avec des matrices sparse :

```python
cand_vec = self.vectorizer.transform([candidate_text])
sims = cosine_similarity(cand_vec, self.job_matrix).flatten()
```

### 3.3 Composante 2 — Bonus Sectoriel

Le bonus sectoriel capture la **cohérence métier** entre le secteur demandé par le candidat et le secteur de l'offre. Il est calculé selon une logique à trois niveaux :

| Cas | Score | Exemple |
|---|---|---|
| Correspondance exacte | **1,0** | "Informatique" ↔ "Informatique" |
| Correspondance partielle (mot en commun) | **0,5** | "Télécommunications" ↔ "Informatique et Télécommunications" |
| Aucune correspondance | **0,0** | "Agriculture" ↔ "Finance" |

Cette approche à niveaux évite la pénalisation trop sévère des secteurs voisins (ex: BTP et Génie Civil) et récompense les concordances exactes.

### 3.4 Composante 3 — Bonus Géographique

Le bonus géographique modélise la **mobilité du candidat** face à la localisation de l'offre :

| Cas | Score | Exemple |
|---|---|---|
| Candidat à mobilité nationale/internationale | **1,0** | "Tout Congo" ou "Oui" |
| Ville correspondante | **0,8** | Candidat à Brazzaville ↔ Offre à Brazzaville |
| Même pays (mention "Congo") | **0,3** | Générique |
| Offre sans localisation déclarée | **0,2** | Valeur par défaut |
| Aucune correspondance | **0,0** | Candidat Brazzaville ↔ Offre Pointe-Noire |

### 3.5 Score Final et Normalisation

Le score final est calculé puis normalisé dans [0, 1] par rapport au score maximum du candidat courant :

```python
scores = alpha * sem + beta * sec + gamma * geo
scores = scores / scores.max()  # Normalisation
```

Cette normalisation garantit que le meilleur candidat obtient toujours un score de 1,0, facilitant l'interprétation par les conseillers.

### 3.6 Génération des Recommandations en Batch

Pour les 41 298 candidats, les recommandations sont générées séquentiellement via `recommend_all()`. Le processus :
1. Vectorise le profil textuel du candidat.
2. Calcule les scores hybrides contre les 2 678 offres.
3. Extrait les indices des K offres avec les scores les plus élevés (`np.argsort`).
4. Retourne les résultats au format `{candidate_id, rank, job_id, score}`.

**Volume généré** : 412 980 recommandations (10 par candidat) en environ 7,5 minutes.

### 3.7 Persistance du Modèle

Le moteur entraîné est sérialisé via `pickle` dans `data/embeddings/engine.pkl`. Le chargement en production évite de ré-entraîner le vectoriseur à chaque démarrage de l'application.

### 3.8 Justification des Choix par Rapport aux Alternatives

| Approche | Avantages | Inconvénients | Pourquoi non retenue |
|---|---|---|---|
| **TF-IDF + Cosinus** *(retenue)* | Rapide, interprétable, robuste aux données courtes | Pas de sémantique latente | — |
| BM25 | Meilleure pondération que TF-IDF | Plus complexe, peu de gain sur données courtes | Gain marginal attendu |
| Word2Vec / FastText | Capture la sémantique | Nécessite un grand corpus en français local | Pas de corpus disponible |
| BERT / CamemBERT | Représentations contextuelles de pointe | Très lent sur 110M paires, GPU requis | Contrainte computationnelle |
| Filtrage collaboratif | Apprend des comportements passés | Requiert de l'historique de placement | Aucun historique disponible |

---

## 4. Résultats d'Évaluation

### 4.1 Protocole d'Évaluation

**Vérité terrain** : Le fichier `Appariement_Demandeurs_Offres.xlsx` fournit, pour chaque candidat, les identifiants de **3 offres pertinentes** validées par les conseillers ACPE. Ces triplets constituent la référence pour l'évaluation.

**Métriques** : Trois métriques standard de systèmes de recommandation sont calculées pour K = 5 et K = 10 :

#### Precision@K
Proportion d'offres recommandées qui sont effectivement pertinentes parmi les K premières :

```
Precision@K = |recommandé[:K] ∩ pertinent| / K
```

#### Recall@K
Proportion des offres pertinentes retrouvées parmi les K premières recommandations :

```
Recall@K = |recommandé[:K] ∩ pertinent| / |pertinent|
```

#### NDCG@K — Normalized Discounted Cumulative Gain
Mesure tenant compte de l'ordre des résultats : une offre pertinente en position 1 rapporte plus qu'en position K.

```
DCG@K  = Σ_{i=1}^{K} rel_i / log₂(i + 1)
NDCG@K = DCG@K / IDCG@K
```

où `IDCG@K` est le score idéal (si toutes les offres pertinentes étaient placées en tête de liste).

### 4.2 Résultats Quantitatifs

| Métrique | Valeur | Interprétation |
|---|---|---|
| **Precision@5** | **27,65 %** | En moyenne, 1,38 offre sur 5 recommandées est pertinente |
| **Recall@5** | **46,08 %** | Le système retrouve 46 % des offres de référence dès les 5 premières |
| **NDCG@5** | **43,61 %** | Qualité de classement à 5 (offres pertinentes bien positionnées) |
| **Precision@10** | **17,28 %** | En moyenne, 1,73 offre sur 10 recommandées est pertinente |
| **Recall@10** | **57,58 %** | Le système retrouve 58 % des offres de référence dans les 10 premières |
| **NDCG@10** | **48,84 %** | Qualité de classement à 10 |

### 4.3 Analyse et Interprétation

#### Performance du Recall@10 : 57,58 %

Ce résultat signifie que, pour près de **6 candidats sur 10**, au moins l'une de leurs offres de référence figure dans les 10 premières recommandations du système. C'est particulièrement remarquable étant donné que :
- Les offres principales ne disposent d'aucune description textuelle (titre et secteur uniquement).
- Le système ne dispose d'aucun historique de placements réussis.

#### Precision@5 vs Recall@5 : le compromis classique

La Precision@5 (27,65 %) semble faible, mais ce résultat doit être contextualisé : la vérité terrain ne contient que **3 offres pertinentes** par candidat (sur 2 678 possibles). Un système aléatoire obtiendrait une Precision@5 de seulement ~0,11 %. Le moteur obtient donc **250 fois mieux que le hasard**.

#### NDCG@10 : 48,84 %

Le NDCG mesure non seulement si les offres pertinentes sont retrouvées, mais aussi si elles apparaissent en tête de liste. Un score de 48,84 % indique que le système **ordonne correctement** les recommandations dans près d'un cas sur deux.

#### Évolution K=5 → K=10

| Métrique | K=5 | K=10 | Gain |
|---|---|---|---|
| Recall | 46,08 % | 57,58 % | **+11,50 points** |
| NDCG | 43,61 % | 48,84 % | **+5,23 points** |

Le gain en Recall entre K=5 et K=10 (+11,5 points) confirme que le système place les offres pertinentes entre les rangs 6 et 10, justifiant l'utilisation du Top-10 comme liste de recommandation principale.

### 4.4 Limites de l'Évaluation

- La vérité terrain est limitée à 3 offres par candidat, sous-estimant potentiellement le nombre réel d'offres compatibles.
- L'évaluation est réalisée sur l'ensemble du dataset sans séparation train/test, car le système n'a pas de phase d'apprentissage supervisé au sens strict.

---

## 5. Interface et Tableau de Bord

### 5.1 Architecture Applicative

L'interface est développée avec **Streamlit** (Python), déployée sur **Streamlit Community Cloud** et accessible à l'adresse publique :
`https://indabax-acpe-matching-zre9mtcplujy2yvpaeywc9.streamlit.app`

L'application est structurée en **5 pages** accessibles via la barre latérale de navigation.

### 5.2 Description des Pages

#### Page 1 — Accueil

Tableau de bord de bienvenue affichant :
- Les **4 indicateurs clés** : nombre de candidats, d'offres, de secteurs et de villes.
- La description des fonctionnalités disponibles avec navigation rapide.
- L'explication pédagogique de la formule du score hybride.

#### Page 2 — Recommandations Personnalisées

Page principale à destination des conseillers ACPE. Elle offre deux modes d'utilisation :

**Mode 1 — Par ID candidat** :
1. Le conseiller sélectionne un matricule dans la liste déroulante.
2. Le système affiche le profil complet du candidat (âge, diplôme, métier visé, mobilité).
3. Les offres de référence issues de la vérité terrain sont affichées en vert.
4. Le Top-5 ou Top-10 est présenté sous forme de tableau interactif et de graphique à barres.
5. Le détail d'une offre sélectionnée décompose les trois composantes du score (TF-IDF, secteur, géographie).

**Mode 2 — Profil libre** :
Permet aux conseillers de tester l'algorithme avec un profil saisi manuellement (métier visé, secteur, diplôme, mobilité).

#### Page 3 — Tableau de Bord Décisionnel

Visualisations interactives (Plotly) pour le pilotage analytique :

- **Top 10 Secteurs par volume d'offres** (graphique en barres horizontal)
- **Top 10 Métiers Visés par les candidats** (graphique en barres horizontal)
- **Répartition Géographique des offres** (camembert / donut)
- **Distribution des Niveaux d'Étude** (histogramme)
- **Répartition par Genre** (camembert)
- **Distribution des Âges des candidats** (histogramme, 30 bins)
- **Types de Contrat proposés** (graphique en barres)

Toutes les visualisations utilisent le thème sombre `plotly_dark` pour une cohérence visuelle.

#### Page 4 — Recherche en Langage Naturel *(Défi Bonus 1)*

Module de recherche sémantique permettant de formuler des requêtes libres en français :

- **Recherche d'offres** : ex. *"Je cherche un poste de comptable à Brazzaville avec un contrat CDI"*
- **Recherche de candidats** : ex. *"Candidat en informatique avec mobilité nationale, niveau Bac+5"*

Le moteur utilise le même vectoriseur TF-IDF pour encoder la requête et la comparer aux profils en corpus. Les résultats sont affichés avec leur score de similarité.

#### Page 5 — Analyse Skill Gap *(Défi Bonus 2)*

Module d'analyse des écarts de compétences entre un candidat et une offre spécifique :

1. Le conseiller sélectionne un candidat et une offre.
2. Le système compare les termes des profils textuels de chacun.
3. Il identifie :
   - **Compétences communes** (présentes dans les deux profils)
   - **Compétences manquantes** (présentes dans l'offre, absentes chez le candidat)
   - **Compétences supplémentaires** (maîtrisées par le candidat mais non requises par l'offre)
4. Un graphique de type **jauge** (gauge chart Plotly) affiche le taux de couverture des compétences.
5. Une liste de **recommandations de formation** est générée pour les compétences manquantes.

### 5.3 Choix Techniques de l'Interface

| Choix | Justification |
|---|---|
| **Streamlit** | Développement rapide, intégration native Python/Pandas/Plotly |
| **Plotly** | Graphiques interactifs (zoom, filtres, survol) adaptés au pilotage |
| **@st.cache_resource** | Les données et le modèle sont chargés une seule fois, puis mis en cache |
| **Thème sombre** | Meilleure lisibilité des données et esthétique professionnelle |
| **Layout wide** | Exploitation maximale de l'espace écran pour les tableaux et graphiques |

---

## 6. Perspectives et Améliorations

### 6.1 Améliorations du Moteur d'Appariement

#### 6.1.1 Passage aux Plongements Denses (Embeddings)

La principale limite du TF-IDF est l'absence de compréhension sémantique latente : les synonymes ne sont pas reconnus comme tels (`"informaticien"` ≠ `"développeur"`). Une évolution naturelle serait d'utiliser :

- **CamemBERT** (BERT pré-entraîné sur le français) pour encoder les profils en vecteurs de 768 dimensions, capturant la sémantique contextuelle.
- **Sentence-BERT** pour obtenir des représentations de phrases entières optimisées pour la similarité.

L'indexation vectorielle via **FAISS** (Facebook AI Similarity Search) permettrait de maintenir des temps de réponse sub-secondes malgré la haute dimensionnalité.

#### 6.1.2 Apprentissage Supervisé

Avec suffisamment d'historique de placements réussis, il serait possible d'entraîner un modèle discriminatif (ex: **LightGBM**) sur des features extraites des paires candidat-offre pour prédire directement la probabilité de succès d'un appariement. Cela permettrait d'apprendre les interactions non linéaires entre les variables.

#### 6.1.3 Calibration des Poids

Les poids α = 0,70, β = 0,20, γ = 0,10 ont été définis heuristiquement. Une **optimisation bayésienne** (ex: bibliothèque Optuna) permettrait de trouver les pondérations maximisant le NDCG@5 sur un ensemble de validation.

### 6.2 Enrichissement des Données

#### 6.2.1 Description des Offres

La quasi-absence de descriptions textuelles est le principal facteur limitant les performances. Plusieurs approches pourraient combler ce manque :
- **Génération automatique** : utiliser un LLM (ex: GPT-4, Gemini) pour générer une description de poste à partir du titre, du secteur et du type de contrat.
- **Collecte web** : enrichir les offres ACPE avec des descriptions issues de portails d'emploi congolais ou régionaux.
- **Formulaires structurés** : inviter les entreprises à compléter un formulaire détaillé lors du dépôt d'offre.

#### 6.2.2 Normalisation du Référentiel Métiers

L'introduction d'un **référentiel de métiers structuré** (similaire au ROME en France) permettrait de standardiser les intitulés et de créer des hiérarchies métier/famille/domaine, améliorant les correspondances sectorielles.

### 6.3 Fonctionnalités Complémentaires

| Fonctionnalité | Valeur ajoutée |
|---|---|
| **Notifications automatiques** | Alerter le candidat par SMS/email quand une offre compatible est publiée |
| **Feedback conseiller** | Permettre aux conseillers de valider/rejeter les recommandations pour ré-entraîner le modèle |
| **Scoring des entreprises** | Intégrer un historique des recrutements réussis par entreprise |
| **Analyse de tendances** | Détecter les secteurs en croissance/décroissance pour anticiper les besoins en formation |
| **API REST** | Exposer le moteur via une API pour intégration aux systèmes d'information de l'ACPE |

### 6.4 Déploiement en Production

Pour un déploiement à l'échelle de l'ACPE, les évolutions infrastructurelles suivantes seraient nécessaires :

- **Base de données** : remplacer les fichiers Excel par une base de données PostgreSQL avec mise à jour incrémentale des offres et candidats.
- **Recalcul différentiel** : à l'arrivée d'une nouvelle offre, ne recalculer les scores que pour les candidats susceptibles d'être intéressés (filtrage préalable par secteur/localisation).
- **Monitoring** : surveiller la dérive des performances (data drift) au fil du temps et déclencher un ré-entraînement si le NDCG chute sous un seuil.
- **Authentification** : sécuriser l'accès à l'interface pour les conseillers via un système de login.

---

## Conclusion

Ce prototype démontre qu'un système d'appariement intelligent, même basé sur des techniques classiques de traitement du langage naturel (TF-IDF, similarité cosinus), peut apporter une valeur opérationnelle significative à l'ACPE :

- **57,58 % des offres de référence** sont retrouvées dans le Top-10, contre ~0,1 % pour une sélection aléatoire.
- **412 980 recommandations** sont générées en moins de 8 minutes, tâche impossible manuellement.
- L'interface rend les résultats accessibles et exploitables par des conseillers non techniques.

Les voies d'amélioration identifiées — enrichissement textuel des offres, embeddings denses, apprentissage supervisé — permettraient d'atteindre des performances encore plus élevées dans les prochaines itérations du système.

---

## Annexes

### A. Structure du Projet

```
InDaBaX/
├── data/
│   ├── raw/                          # Données brutes Excel
│   │   ├── Offres_ACPE.xlsx
│   │   ├── Offres_ACPE_Extensions.xlsx
│   │   ├── Demandeurs .xlsx
│   │   └── Appariement_Demandeurs_Offres.xlsx
│   ├── processed/                    # Données nettoyées (CSV)
│   └── embeddings/
│       └── engine.pkl                # Moteur TF-IDF sérialisé
├── src/
│   ├── preprocessing.py              # Nettoyage et préparation
│   ├── matching_engine.py            # Moteur d'appariement hybride
│   ├── evaluation.py                 # Métriques P@K, R@K, NDCG@K
│   ├── skill_gap.py                  # Analyse des écarts de compétences
│   └── search.py                     # Recherche NLP
├── app/
│   └── streamlit_app.py              # Interface 5 pages
├── scripts/
│   └── generate_recommendations.py  # Script d'entraînement et génération
├── tests/                            # 48 tests unitaires (pytest)
├── outputs/
│   ├── recommendations.csv           # 412 980 recommandations générées
│   └── evaluation_report.csv         # Rapport de performance
└── requirements.txt                  # Dépendances Python
```

### B. Dépendances Techniques

| Bibliothèque | Version | Usage |
|---|---|---|
| `scikit-learn` | ≥ 1.3 | TF-IDF, similarité cosinus |
| `pandas` | ≥ 2.0 | Manipulation des données |
| `numpy` | ≥ 1.24 | Calculs vectoriels |
| `streamlit` | ≥ 1.30 | Interface web |
| `plotly` | ≥ 5.15 | Visualisations interactives |
| `openpyxl` | ≥ 3.1 | Lecture des fichiers Excel |
| `tqdm` | ≥ 4.65 | Barre de progression |
| `pytest` | ≥ 7.4 | Tests unitaires |

### C. Résultats des Tests Unitaires

```
48 tests au total — 48 passed ✓

tests/test_evaluation.py    : 18 tests (P@K, R@K, NDCG@K, évaluation globale)
tests/test_matching.py      : 14 tests (score hybride, bonus, persistance)
tests/test_preprocessing.py : 16 tests (nettoyage, normalisation, profils)
```

---

*Rapport rédigé dans le cadre du Hackathon IndabaX Congo 2026 — Thème : Développement d'un Système Intelligent d'Appariement entre les Demandeurs d'Emploi et les Offres d'Emploi.*
