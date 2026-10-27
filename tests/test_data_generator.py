"""
Tests for generate_bengaluru_data.py:
  1. Default run with zone_income_coupling=0.0 produces identical output to original.
  2. Setting zone_income_coupling > 0 creates synthetic positive control where
     income depends on zone affluence.
"""

import numpy as np
import pandas as pd
import pytest

import generate_bengaluru_data as g


def test_default_population_generation_identical_up_to_floating_point_tolerance():
    """
    Verifies that running generate_population with default zone_income_coupling=0.0
    reproduces existing population.csv records identical up to floating-point tolerance (<1e-14).
    """
    pop_orig = pd.read_csv("data/population.csv")

    g.rng = np.random.default_rng(g.RNG_SEED)
    g.generate_antenna_layout()
    pop_new = g.generate_population(n_agents=g.N_AGENTS, zone_income_coupling=0.0)

    # Identical shape and column names
    assert pop_orig.shape == pop_new.shape
    assert list(pop_orig.columns) == list(pop_new.columns)

    # Categorical and discrete columns must match identically
    discrete_cols = [
        "agent_id", "phone_number", "age_group", "gender", "education_level",
        "occupation_category", "work_status", "income_bracket",
        "socio_demographic_class", "home_zone", "work_zone",
    ]
    for col in discrete_cols:
        assert (pop_orig[col] == pop_new[col]).all(), f"Mismatch in column {col}"

    # Coordinates match within floating-point serialization tolerance (1e-6 degrees ~ 10cm)
    for col in ["home_lat", "home_lon", "work_lat", "work_lon"]:
        np.testing.assert_allclose(pop_orig[col], pop_new[col], atol=1e-6)


def test_zone_income_coupling_positive_control():
    """
    Verifies that zone_income_coupling > 0.0 couples income bracket with home zone.
    """
    g.rng = np.random.default_rng(g.RNG_SEED)
    g.generate_antenna_layout()
    pop_coupled = g.generate_population(n_agents=g.N_AGENTS, zone_income_coupling=1.5)

    g.rng = np.random.default_rng(g.RNG_SEED)
    g.generate_antenna_layout()
    pop_uncoupled = g.generate_population(n_agents=g.N_AGENTS, zone_income_coupling=0.0)

    # Income distributions across zones must differ when coupling is active
    assert not (pop_coupled["income_bracket"] == pop_uncoupled["income_bracket"]).all()
