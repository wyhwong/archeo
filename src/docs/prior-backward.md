Priors refer to the prior probability distributions for each ancestral model parameter.

---

## Initializing Preset Priors

Initializing a preset prior in archeo is straightforward.
For example, to initialize a tiny aligned spin prior (for testing), you can use the following code:

```python
import archeo

prior = archeo.Prior.from_config("default")
```

Note that the prior initialization will take some time for:

1. Downloading the fits model if not already cached.
2. Generating samples from the fits model.

The wait time mainly depends on the number of samples specified in the prior configuration.
During the sample generation process, you will see progress bars indicating the status of sample generation.

If you have generated the prior samples once,
they could be cached locally for future use,
and subsequent initializations will be much faster.

```python
prior.to_parquet("path/to/save/prior.parquet")

# NOTE: this function also works for json/csv/feather format
prior = archeo.Prior.from_filepath("path/to/save/prior.parquet")
```

You may replace the string `"tiny_aligned_spin"` with any other supported preset prior name, here we list all available preset priors and corresponding configurations:

| Name | Samples  | Fits Model | Spin Aligned | Only Up-Aligned Spin | $\chi_1$ | $\chi_2$ | $\phi_1$ [rad] | $\phi_2$ [rad] | $\theta_1$ [rad] | $\theta_2$ [rad] | $m_1 [M_\odot]$ | $m_2 [M_\odot]$ | $q$ | Uniform in $q$ |
|------------------------------------|-----------|------------------|----|-----|-------|-------|------------|------------|-----------|-----------|---------|---------|-------|-----|
| default (tiny_aligned_spin)        | 5,000     | NRSur3dq8Remnant | ✅ | ❌ | 0 - 1 | 0 - 1 | 0 - $2\pi$ | 0 - $2\pi$ | 0 - $\pi$ | 0 - $\pi$ | 5 - 200 | 5 - 200 | 1 - 6 | ❌ |
| agnostic_precessing_spin           | 2,000,000 | NRSur7dq4Remnant | ❌ | ❌ | 0 - 1 | 0 - 1 | 0 - $2\pi$ | 0 - $2\pi$ | 0 - $\pi$ | 0 - $\pi$ | 5 - 200 | 5 - 200 | 1 - 6 | ❌ |
| agnostic_aligned_spin              | 2,000,000 | NRSur3dq8Remnant | ✅ | ❌ | 0 - 1 | 0 - 1 | 0 - $2\pi$ | 0 - $2\pi$ | 0 - $\pi$ | 0 - $\pi$ | 5 - 200 | 5 - 200 | 1 - 6 | ❌ |
| precessing_spin                    | 2,000,000 | NRSur7dq4Remnant | ❌ | ❌ | 0 - 1 | 0 - 1 | 0 - $2\pi$ | 0 - $2\pi$ | 0 - $\pi$ | 0 - $\pi$ | 5 - 65  | 5 - 65  | 1 - 6 | ❌ |
| aligned_spin                       | 2,000,000 | NRSur3dq8Remnant | ✅ | ❌ | 0 - 1 | 0 - 1 | 0 - $2\pi$ | 0 - $2\pi$ | 0 - $\pi$ | 0 - $\pi$ | 5 - 65  | 5 - 65  | 1 - 6 | ❌ |
| positively_aligned_spin            | 2,000,000 | NRSur3dq8Remnant | ✅ | ✅ | 0 - 1 | 0 - 1 | 0 - $2\pi$ | 0 - $2\pi$ | 0 - $\pi$ | 0 - $\pi$ | 5 - 65  | 5 - 65  | 1 - 6 | ❌ |

---

## Customizing Prior Configurations

You can also create a customized prior configuration using the data structure in archeo:

```python
from archeo.schema import Domain, PriorConfig

prior_config = PriorConfig(
    n_samples=<int>,
    is_spin_aligned=<True/False>,
    m_1=Domain(low=<float>, high=<float>),  # unit: solar mass
    m_2=Domain(low=<float>, high=<float>),  # unit: solar mass
    a_1=Domain(low=<float>, high=<float>),  # unit: dimensionless
    a_2=Domain(low=<float>, high=<float>),  # unit: dimensionless
)

prior = archeo.Prior.from_config(prior_config)
```

To have more specific control over the prior configuration,
you may want to look into [archeo.schema](https://github.com/wyhwong/archeo/blob/main/src/archeo/schema.py).
We provide some extra configuration options such as `is_only_up_aligned_spin` and `is_uniform_in_mass_ratio` to help you set up your desired prior easily.

We also implemented an astrophysical motivated mass distribution prior based on [Mahapatra et al., 2022](https://arxiv.org/abs/2209.05766). You can enable this mass function by setting the flag `is_mahapatra_mass_func=True` in the `PriorConfig`.
