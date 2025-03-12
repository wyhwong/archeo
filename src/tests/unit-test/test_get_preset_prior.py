import os

import pytest

import archeo
import archeo.schema


def test_get_preset_prior():
    """Test the get_preset_prior function."""

    for name in [
        "default",
        "agnostic_precessing_spin",
        "agnostic_aligned_spin",
        "precessing_spin",
        "aligned_spin",
        "positively_aligned_spin",
    ]:

        assert isinstance(archeo.get_prior_config(name), archeo.schema.PriorConfig)


def test_prior_config_methods():
    """Test the PriorConfig class methods."""

    filepath = "./default_prior.json"
    prior = archeo.get_prior_config("default")

    assert not os.path.exists(filepath)
    prior.to_json(filepath)
    assert os.path.exists(filepath)
    reloaded_prior = archeo.schema.PriorConfig.from_json(filepath)
    assert prior == reloaded_prior
    os.remove(filepath)
    assert not os.path.exists(filepath)


def test_get_invalid_prior():
    """Test the get_preset_prior function with invalid name."""

    pytest.raises(ValueError, archeo.get_prior_config, "invalid")
