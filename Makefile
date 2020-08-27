SHELL := /bin/bash

ifeq ($(workdir),)
ifeq ($(wildcard $(PWD)/.tempdir),)
workdir := $(shell mktemp -d "/tmp/build-intake-esgf_XXXXXXXX")
$(shell echo $(workdir) >> $(PWD)/.tempdir)
endif

workdir := $(shell cat $(PWD)/.tempdir)
endif

conda_bin := $(shell find /opt/*conda*/bin $(HOME)/*conda*/bin -type f -iname "conda")
conda_rc := $(workdir)/condarc
conda_cmd := CONDARC=$(conda_rc) conda
conda_act_cmd := $(subst bin/conda,bin/activate,$(conda_bin))
conda_act := source $(conda_act_cmd)

feedstock := $(workdir)/intake-esgf-feedstock
artifacts := $(workdir)/artifacts

ifneq ($(and $(CONDA_TOKEN),$(CONDA_USER)),)
conda_upload := --token $(CONDA_TOKEN) --user $(CONDA_USER)
endif

.PHONY: purge
purge:
	$(conda_act) build; \
		$(conda_cmd) build purge

.PHONY: clean
clean:
	rm $(PWD)/.tempdir
	rm -rf $(workdir)

.PHONY: prep-conda
prep-conda:
	rm -f $(conda_rc)
	rm -rf $(feedstock)

	conda config --file $(conda_rc) --append channels defaults
	conda config --file $(conda_rc) --append channels conda-forge
	conda config --file $(conda_rc) --set always_yes true
	conda config --file $(conda_rc) --set anaconda_upload true

	echo -e "conda-build:\n  root-dir: $(artifacts)\n" >> $(conda_rc)

.PHONY: build-environment
build-environment: prep-conda
	$(conda_act) base; \
		$(conda_cmd) create -n build conda-build anaconda-client conda-smithy

.PHONY: prep-feedstock
prep-feedstock: build-environment
	$(conda_act) build; \
		$(conda_cmd) smithy init --feedstock-directory $(feedstock) recipe/; \
		$(conda_cmd) smithy rerender --feedstock_directory $(feedstock)

	sed -i"" "s/\(.*path: \).*/\1$(subst /,\/,$(PWD))/g" $(feedstock)/recipe/meta.yaml

.PHONY: build
build: prep-feedstock
	$(conda_act) build; \
		$(conda_cmd) build -m $(feedstock)/.ci_support/linux_64_.yaml -c conda-forge $(feedstock)/recipe \
		$(conda_upload)

.PHONY: dev-environment
dev-environment: pkgs := intake intake-xarray fsspec pytest pytest-cov pytest-mock bump2version \
	jupyterlab ipywidgets
dev-environment:
	$(conda_act) base; \
		$(conda_cmd) create -n dev-intake-esgf -c conda-forge $(pkgs); \
		$(conda_act) dev-intake-esgf; \
		pip install -e .

.PHONY: build-docs
build-docs: build-environment
	$(conda_act) build; \
		$(conda_cmd) install -c conda-forge sphinx sphinx_rtd_theme recommonmark; \
		make -C docsrc/ html

	@cp -a docsrc/build/html/. docs/
