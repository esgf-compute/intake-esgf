# Intake-esgf
Intake plugin for ESGF search service.

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
