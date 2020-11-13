.DEFAULT_GOAL := build

CONDA := $(shell find /opt/conda/bin $(HOME)/conda/bin $(PWD)/conda/bin -type f -name conda 2>/dev/null)
CONDA_PATH := $(patsubst %/conda,%,$(CONDA))

FEEDSTOCK_PATH := $(PWD)/feedstock

.PHONY: build-env
build-env: PKGS := conda-build conda-smithy anaconda-client bump2version
build-env:
	$(CONDA) install -y -c conda-forge $(PKGS)

.PHONY: rerender
rerender:
	$(CONDA) smithy rerender \
		--feedstock_directory $(FEEDSTOCK_PATH)

.PHONY: build
build:
	$(CONDA) build \
		-c conda-forge \
		-m $(FEEDSTOCK_PATH)/.ci_support/linux_64_.yaml \
		--output-folder $(PWD)/output \
		$(CONDA_EXTRA) \
		$(FEEDSTOCK_PATH) 

.PHONY: build-docs
build-docs: build-environment
	$(CONDA) install -c conda-forge sphinx sphinx_rtd_theme recommonmark m2r

	@make -C dodsrc/ html

	@cp -a dodsrc/build/html/. docs/
