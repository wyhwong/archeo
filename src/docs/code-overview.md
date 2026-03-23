## Conceptual workflow: From GW Observations to Ancestral Inference

This is the “mental model” of how the repository fits together: simulations generate populations, importance sampling connects priors/posteriors, and downstream utilities compute Bayes factors and plots.

```mermaid
flowchart TD
  %% Inputs
  PR["Priors on source / ancestral binaries\n(distributions, domains, configuration)"] --> SIM["Simulate black hole mergers\n(generate synthetic populations)"]
  PE["Parameter-estimation samples\n(observed GW events)"] --> AP["Infer ancestral posterior\n(condition on PE samples)"]

  %% Backward approach (default)
  SIM --> AP

  %% Forward / hybrid approach (optional)
  SIM --> IS{"Importance sampling?\n(optional reweighting)"}
  IS -->|Compute evidence| BFC["Compute Bayes factor curves\n(compare priors / models)"]
  IS -->|Reweight samples| AP

  %% Outputs
  SIM --> AW["Analysis outputs\n(visualization / postprocessing)"]
  AP --> AW
  BFC --> AW
```

**Notes**
- Archeo primarily implements the **backward modeling** workflow (conditioning on existing PE samples), and supports a **forward-like mode via importance sampling** (reweighting under new ancestral priors).

---

## Base Diagrams

In the following, we introduce the data classes used in archeo using UML-style diagrams.

The UML-style diagrams below are originally generated from the codebase using **Pyreverse** ([docs](https://pylint.pycqa.org/en/v2.14.5/pyreverse.html)).
Pyreverse analyzes your source code and generates package and class diagrams.
You can generate the base diagrams with:

```bash
pyreverse -o mmd -d . archeo
```

However, while Pyreverse is great for *complete* diagrams,
full-project graphs often become too dense to read in a browser.
So below we reorganize the output into **smaller, purpose-driven diagrams**:

- Core physical objects: `Binary` and `BlackHole`
- Core sampling primitives: `Distribution` and `Domain`
- Importance sampling: `ImportanceSamplingData` and related classes
- Bayes factor curves: `CandidatePrior`, `BayesFactorCurve`, and related classes

---

## Core Physical Objects: `Binary` and `BlackHole`

Here we introduce the core data models for the repository: `Binary` and `BlackHole`.
These are the fundamental building blocks for simulating populations and performing inference.

```mermaid
classDiagram
  class BlackHole {
    mass: PositiveFloat
    speed: NonNegativeFloat
    spin_magnitude: NonNegativeFloat
    spin_vector : tuple[float,float,float]
    horizontal_spin : NonNegativeFloat
    vertical_spin : NonNegativeFloat
  }

  class Binary {
    mass_ratio : PositiveFloat
    effective_spin : NonNegativeFloat
    precession_spin : NonNegativeFloat
    primary_black_hole : BlackHole
    secondary_black_hole : BlackHole
  }

  class BinaryGenerator {
    is_aligned_spin : bool
    enforce_source_binding : bool
    primary_black_hole_source : BlackHoleSource
    secondary_black_hole_source : BlackHoleSource
    mass_ratio_domain : Domain
    draw(size: int) list[Binary]
  }

  class BlackHoleSource {
    **parameters: Any
    draw(size: int) list[BlackHole]
  }

  class BlackHoleGenerator {
    mass_distribution : Distribution
    spin_magnitude_distribution : Distribution
    theta_distribution : Distribution
    phi_distribution : Distribution
    draw(size: int) list[BlackHole]
  }

  class BlackHolePopulation {
    black_holes : list[BlackHole]
    draw(size: int) list[BlackHole]
    from_simulation_results(df: pd.DataFrame, phi_distribution: Distribution, theta_distribution: Distribution) BlackHolePopulation
  }

  BlackHoleSource --> BlackHoleGenerator : Child class
  BlackHoleSource --> BlackHolePopulation : Child class

  Binary --> BlackHole : primary_black_hole
  Binary --> BlackHole : secondary_black_hole

  BinaryGenerator --> BlackHoleSource : primary_black_hole_source
  BinaryGenerator --> BlackHoleSource : secondary_black_hole_source

  BinaryGenerator --> Binary : generates
  BlackHoleGenerator --> BlackHole : generates
  BlackHolePopulation --> BlackHole : generates
```

For details on the `Distribution` and `Domain` classes, see the next section.

---

## Core Sampling Primitives: Distributions and Domains

This diagram captures the “lego bricks” used to build priors (for masses/spins/etc.) and constrain them, supplementing the `Binary` and `BlackHole` data models.

### Distributions

At the moment, we have implemented a few basic distributions (Uniform, Normal, PiecewiseUniform)
and a custom `MahapatraMassFunction` for modeling mass distributions.
All distributions inherit from a common `DistributionBase` class that defines the interface for sampling and parameter management.

```mermaid
classDiagram
  class DistributionBase {
    **parameters: Any
    min : float
    max : float
    draw(size: int) list[float]
  }

  class Uniform {
    low : float
    high : float
  }

  class Normal {
    mean : float
    std : float
  }

  class PiecewiseUniform {
    uniforms : dict[Uniform, PositiveFloat]
  }

  class MahapatraMassFunction {
    alpha : PositiveFloat
    dm : PositiveFloat
    mass : Domain
    resolution : PositiveFloat
    masses : list[float]
    probis : list[float]
  }

  DistributionBase <|-- MahapatraMassFunction : inherits
  DistributionBase <|-- Uniform : inherits
  DistributionBase <|-- Normal : inherits
  DistributionBase <|-- PiecewiseUniform : inherits
```

### Domains

Domains are used to define the ranges of parameters (e.g., mass ratio between 1 and 6) and to check whether samples fall within these ranges.

```mermaid
classDiagram
  class DomainBase {
    **parameters: Any
    contains(value: float)* bool
  }

  class Domain {
    low : float
    high : float
    to_tuple() tuple[float,float]
  }

  class PiecewiseDomain {
    domains : list[Domain]
  }

  DomainBase <|-- Domain : inherits
  DomainBase <|-- PiecewiseDomain : inherits
```

---

## Importance Sampling

Here we introduce the core classes for importance sampling, which is the key technique for forward modeling and reweighting posterior samples under new priors.

The `ImportanceSamplingDataBase` class encapsulates shared data and utilities for importance sampling,
while `ISDataAssumeIndependence` and `ISDataGeneric` implement specific algorithms for computing Bayes factors and reweighted samples under different assumptions.
The `ImportanceSamplingData` class provides a unified interface that can switch between these algorithms based on user configuration.

```mermaid
classDiagram
  class ImportanceSamplingDataBase {
    prior_samples : pd.DataFrame
    posterior_samples : pd.DataFrame
    new_prior_samples : pd.DataFrame
    bounds : dict[str, Domain]
    common_columns : list[str]
    binsize_mass : float
    binsize_spin : float
    get_binsize(col_name: str) float
    get_edges(col_name: str) np.ndarray
    get_nbins(col_name: str) int
    ztol: float
  }

  class ISDataAssumeIndependence {
    get_bayes_factor_1d(bootstrapping: bool) float
    get_likelihood_samples_1d(random_state: int) np.ndarray
    get_reweighted_samples_1d(random_state: int) pd.DataFrame
  }

  class ISDataGeneric {
    get_bayes_factor_dd(bootstrapping: bool) float
    get_likelihood_samples_dd(random_state: int) np.ndarray
    get_reweighted_samples_dd(random_state: int) pd.DataFrame
  }

  class ImportanceSamplingData {
    assume_parameter_independence : bool
    get_bayes_factor(bootstrapping: bool) float
    get_likelihood_samples(random_state: int) np.ndarray
    get_reweighted_samples(random_state: int) pd.DataFrame
    sample_bayes_factor(n: int, is_parallel: bool, n_threads: int) BayesFactor
  }

  class BayesFactor {
    samples: list[float]
    median: float
    confidence_interval(percent: float) tuple[float, float]
  }

  ISDataAssumeIndependence --|> ImportanceSamplingDataBase : inherits
  ISDataGeneric --|> ImportanceSamplingDataBase : inherits
  ImportanceSamplingData --|> ISDataAssumeIndependence : inherits
  ImportanceSamplingData --|> ISDataGeneric : inherits
  ImportanceSamplingData --> BayesFactor : generates
```

---

## Importance Sampling: Compute Bayes Factor Curve

Here we introduce the important classes for computing the Bayes factor curve,
which is a key analysis tool for comparing different priors/models.

We first define a `CandidatePrior` class to represent candidate prior distributions
and provide utilities for sampling escape velocities and conditional priors.
Then the `BayesFactorCurve` class computes the Bayes factor as a function of escape velocity
by comparing the candidate prior with the inferred posterior distribution.

```mermaid
classDiagram
  class CandidatePrior {
    df_bh1 : pd.DataFrame
    df_bh2 : pd.DataFrame
    get_host_escape_velocities(n_pts: int, log_scale: bool) list[float]
    get_conditional_prior(v_esc: float, n_min: int, random_state: int) pd.DataFrame
  }

  class BayesFactorCurve {
    n_pts : int
    log_scale : bool
    n_bootstrapping : int
    get_bayes_factor_over_escape_velocity(prior: pd.DataFrame, posterior: pd.DataFrame, candidate_prior: CandidatePrior, n_workers: int) dict[float, BayesFactor]
  }

  class ImportanceSamplingData {
    assume_parameter_independence : bool
    get_bayes_factor(bootstrapping: bool) float
    get_likelihood_samples(random_state: int) np.ndarray
    get_reweighted_samples(random_state: int) pd.DataFrame
    sample_bayes_factor(n: int, is_parallel: bool, n_threads: int) BayesFactor
  }

  class BayesFactor {
    samples: list[float]
    median: float
    confidence_interval(percent: float) tuple[float, float]
  }

  BayesFactorCurve --> CandidatePrior : uses
  BayesFactorCurve --> ImportanceSamplingData : uses
  ImportanceSamplingData --> BayesFactor : generates
```
