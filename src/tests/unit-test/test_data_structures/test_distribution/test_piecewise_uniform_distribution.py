import numpy as np

from archeo.data_structures.distribution import PiecewiseUniform, Uniform


SAMPLE_SIZE = 100000


def test_piecewise_uniform_distribution():

    uniforms = {
        Uniform(low=0, high=3): 0.5,
        Uniform(low=3, high=7): 0.3,
        Uniform(low=7, high=10): 0.2,
    }
    dist = PiecewiseUniform(uniforms=uniforms)

    assert dist.min == 0
    assert dist.max == 10

    samples = dist.draw(size=SAMPLE_SIZE)
    assert len(samples) == SAMPLE_SIZE
    assert all(0 <= sample <= 10 for sample in samples)

    # Check expected mean
    expected_mean = sum((u.low + u.high) / 2 * p for u, p in uniforms.items())
    assert np.isclose(np.mean(samples), expected_mean, atol=expected_mean * 0.1)

    # Separate the samples into 10 bins
    # the expected count in each set of bins should be around 500, 300, and 200 respectively
    counts, _ = np.histogram(samples, bins=[0, 3, 7, 10])
    expected_counts = [len(samples) * p for p in uniforms.values()]
    assert all(
        np.isclose(count, expected_count, atol=expected_count * 0.1)
        for count, expected_count in zip(counts, expected_counts)
    )
