import numpy as np
import pytest

from archeo.data_structures.distribution import PiecewiseUniform, Uniform


def test_piecewise_uniform_rejects_weights_not_summing_to_one():
    with pytest.raises(ValueError, match="Total weights must sum to 1"):
        PiecewiseUniform(
            uniforms={
                Uniform(low=0, high=1): 0.4,
                Uniform(low=1, high=2): 0.4,  # sum 0.8
            }
        )


def test_piecewise_uniform_draw_multiple_handles_rounding_remainder():
    dist = PiecewiseUniform(
        uniforms={
            Uniform(low=0, high=1): 0.34,
            Uniform(low=1, high=2): 0.33,
            Uniform(low=2, high=3): 0.33,
        }
    )
    samples = dist.draw(size=10)  # int truncation causes remainder branch
    assert len(samples) == 10
    assert ((samples >= 0) & (samples <= 3)).all()


def test_piecewise_uniform_draw_single_sample_path():
    dist = PiecewiseUniform(uniforms={Uniform(low=10, high=20): 1.0})
    x = dist.draw()
    assert 10 <= x <= 20
