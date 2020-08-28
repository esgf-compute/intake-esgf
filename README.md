# Intake-esgf
Intake plugin for ESGF search service.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jasonb5/intake-esgf/master?filepath=default_catalog.ipynb)

# Installation
```bash
conda create -n intake-esgf -c jasonb5 intake-esgf
```
# Quickstart
```python
import intake

cat = intake.open_esgf_default_catalog()

list(cat)

results = cat.cmip6.search(variable="clt", frequency="mon")
```
