"""
Tests du module preprocessing.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pytest
from src.preprocessing import (
    clean_text, normalize_location, build_candidate_text,
    build_job_text, _is_empty, _normalize_col
)


class TestCleanText:
    def test_removes_extra_spaces(self):
        assert clean_text("  hello   world  ") == "hello world"

    def test_handles_nan(self):
        assert clean_text(float("nan")) == ""
        assert clean_text(None) == ""

    def test_handles_normal_string(self):
        assert clean_text("Data Analyst") == "Data Analyst"

    def test_handles_numeric(self):
        result = clean_text(42)
        assert result == "42"


class TestNormalizeLocation:
    def test_brazzaville_variants(self):
        assert normalize_location("Brazza") == "brazzaville"
        assert normalize_location("BVL") == "brazzaville"
        assert normalize_location("BRAZZAVILLE") == "brazzaville"

    def test_pointe_noire(self):
        assert normalize_location("PNR") == "pointe-noire"
        assert normalize_location("Point Noire") == "pointe-noire"

    def test_nan_returns_non_declare(self):
        assert normalize_location(None) == "non déclaré"
        assert normalize_location(float("nan")) == "non déclaré"


class TestBuildCandidateText:
    def test_concatenates_fields(self):
        row = pd.Series({
            "Métier visé / Qualification visée": "Data Analyst",
            "qualification_metier": "Analyste",
            "Filière / Spécialité": "Statistiques",
            "Secteur demandé": "Technologie",
            "Objectif": "Emploi",
            "Qualification": "Licence",
            "Diplome": "Licence en Informatique",
            "niveau_etude": "Bac +3",
        })
        result = build_candidate_text(row)
        assert "Data Analyst" in result
        assert "Statistiques" in result
        assert "Technologie" in result

    def test_ignores_nan_fields(self):
        row = pd.Series({
            "Métier visé / Qualification visée": "Comptable",
            "qualification_metier": float("nan"),
            "Filière / Spécialité": "Informatique",
            "Secteur demandé": "Technologie",
            "Objectif": "Emploi",
            "Qualification": "Master",
            "Diplome": "Master en Informatique",
            "niveau_etude": "Bac +5",
        })
        result = build_candidate_text(row)
        # Vérifier que 'nan' n'apparaît pas en tant que mot isolé
        import re
        assert not re.search(r'\bnan\b', result.lower())
        assert "Comptable" in result

    def test_ignores_non_declare(self):
        row = pd.Series({
            "Métier visé / Qualification visée": "Ingénieur",
            "qualification_metier": "Non déclaré",
            "Filière / Spécialité": "Génie Civil",
            "Secteur demandé": "BTP",
            "Objectif": "Emploi",
            "Qualification": "Master",
            "Diplome": "Master",
            "niveau_etude": "Bac +5",
        })
        result = build_candidate_text(row)
        assert "Non déclaré" not in result


class TestIsEmpty:
    def test_none_is_empty(self):
        from src.preprocessing import _is_empty
        assert _is_empty(None) is True

    def test_nan_is_empty(self):
        from src.preprocessing import _is_empty
        assert _is_empty(float("nan")) is True

    def test_empty_string_is_empty(self):
        from src.preprocessing import _is_empty
        assert _is_empty("") is True

    def test_non_declare_is_empty(self):
        from src.preprocessing import _is_empty
        assert _is_empty("Non déclaré") is True

    def test_real_value_not_empty(self):
        from src.preprocessing import _is_empty
        assert _is_empty("Data Analyst") is False


class TestNormalizeCol:
    def test_removes_accents(self):
        # La fonction remplace les caractères spéciaux par '_' puis collapse les '_' multiples
        result = _normalize_col("Filière / Spécialité")
        assert result.startswith("filiere")
        assert result.endswith("specialite")
        assert "_" in result

    def test_lowercase(self):
        assert _normalize_col("MATRICULE") == "matricule"
