Priors refer to the prior probability distributions for each ancestral model parameter.

## Initializing Preset Priors

Initializing a preset prior in archeo is straightforward. For example, to initialize a prior with precessing spins, you can use the following code:

```python
import archeo

prior = archeo.Prior.from_config("precessing_spin")
```

You may replace the string `"precessing_spin"` with any other supported preset prior name, here we list all available preset priors and corresponding configurations:

| Name | Samples  | Fits Model | Spin Aligned | Only Up-Aligned Spin | $\chi_1$ | $\chi_2$ | $\phi_1$ [rad] | $\phi_2$ [rad] | $\theta_1$ [rad] | $\theta_2$ [rad] | $m_1 [M_\odot]$ | $m_2 [M_\odot]$ | $q$ | Uniform in $q$ |
|------------------------------------|-----------|------------------|----|-----|-------|-------|------------|------------|-----------|-----------|---------|---------|-------|-----|
| default (tiny experimental)        | 5,000     | NRSur3dq8Remnant | ✅ | ❌ | 0 - 1 | 0 - 1 | 0 - $2\pi$ | 0 - $2\pi$ | 0 - $\pi$ | 0 - $\pi$ | 5 - 200 | 5 - 200 | 1 - 6 | ❌ |
| agnostic_precessing_spin           | 2,000,000 | NRSur7dq4Remnant | ❌ | ❌ | 0 - 1 | 0 - 1 | 0 - $2\pi$ | 0 - $2\pi$ | 0 - $\pi$ | 0 - $\pi$ | 5 - 200 | 5 - 200 | 1 - 6 | ❌ |
| agnostic_aligned_spin              | 2,000,000 | NRSur3dq8Remnant | ✅ | ❌ | 0 - 1 | 0 - 1 | 0 - $2\pi$ | 0 - $2\pi$ | 0 - $\pi$ | 0 - $\pi$ | 5 - 200 | 5 - 200 | 1 - 6 | ❌ |
| precessing_spin                    | 2,000,000 | NRSur7dq4Remnant | ❌ | ❌ | 0 - 1 | 0 - 1 | 0 - $2\pi$ | 0 - $2\pi$ | 0 - $\pi$ | 0 - $\pi$ | 5 - 65  | 5 - 65  | 1 - 6 | ❌ |
| aligned_spin                       | 2,000,000 | NRSur3dq8Remnant | ✅ | ❌ | 0 - 1 | 0 - 1 | 0 - $2\pi$ | 0 - $2\pi$ | 0 - $\pi$ | 0 - $\pi$ | 5 - 65  | 5 - 65  | 1 - 6 | ❌ |
| positively_aligned_spin            | 2,000,000 | NRSur3dq8Remnant | ✅ | ✅ | 0 - 1 | 0 - 1 | 0 - $2\pi$ | 0 - $2\pi$ | 0 - $\pi$ | 0 - $\pi$ | 5 - 65  | 5 - 65  | 1 - 6 | ❌ |

## Customizing Prior Configurations

You can also create a custom prior configuration using the data structure in archeo:
