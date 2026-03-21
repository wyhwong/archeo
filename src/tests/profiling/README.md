# Profiling tests for archeo

Here we profile the simulation and estimation of archeo to identify bottlenecks and optimize performance.
We use the `pyinstrument` profiler to measure the execution time of the code and generate HTML reports for analysis.
In the following, we profile both the second generation aligned spin binary simulation and the second generation precession spin binary simulation, as well as their parallel versions.

## Profiling simulations

```bash
pyinstrument \
    -t \
    -r html \
    -o tests/profiling/simulation_profile.html \
    tests/profiling/simulation.py
```

## Profiling esimation

```bash
pyinstrument \
    -t \
    -r html \
    -o tests/profiling/estimation_profile.html \
    tests/profiling/estimation.py
```
