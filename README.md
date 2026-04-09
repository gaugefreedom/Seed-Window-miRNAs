# Seed-Window-miRNAs

Code release for the seed-window algebraic instability study on dementia-associated miRNAs.

## Included

- `code/seed_window_profiler.py`: main profiler implementation
- `Sliding_Window_Algebraic_Profiler.py`: root CLI entrypoint
- `generate_manuscript_figures.py`: figure/table generator used for the paper
- `test_2_regressions.py`, `test_3_foundations.py`: regression and foundation tests
- `data/pilot_cohort_16.csv`: minimal cohort input used for the pilot workflow

## Install

```bash
python3 -m pip install -r requirements.txt
```

If `Singular` is installed and available on `PATH`, the profiler will use it as the primary Groebner backend and fall back to `SymPy` otherwise.

## Run tests

```bash
python3 -m unittest test_2_regressions.py test_3_foundations.py
```

## Entry point

```bash
python3 Sliding_Window_Algebraic_Profiler.py --help
```
## Archived software release used for the manuscript:
- Zenodo DOI: https://doi.org/10.5281/zenodo.19489106
