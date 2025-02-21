import pytest

import archeo
import archeo.schema


def test_get_preset_prior():
    """Test the get_preset_prior function."""

    for name in ["default", "precessing", "aligned_spin", "positively_aligned_spin"]:

        assert isinstance(archeo.get_prior_config(name), archeo.schema.PriorConfig)


def test_get_invalid_prior():
    """Test the get_preset_prior function with invalid name."""

    pytest.raises(ValueError, archeo.get_prior_config, "invalid")
