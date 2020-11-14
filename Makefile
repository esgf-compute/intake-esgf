.DEFAULT_GOAL := build

CONDA := $(shell find /opt/conda/bin $(HOME)/conda/bin -type f -name conda 2>/dev/null)
CONDA_PATH := $(patsubst %/conda,%,$(CONDA))

SRC_PATH := $(PWD)
FEEDSTOCK_PATH := $(SRC_PATH)/feedstock

ifneq ($(CONDA_TOKEN),)
CONDA_BUILD_EXTRA := --token=$(CONDA_TOKEN)
endif

.PHONY: setup-env
setup-env:
	export PATH=$(CONDA_PATH):$(PATH)

.PHONY: build-env
build-env: PKGS := conda-build conda-smithy anaconda-client bump2version
build-env: setup-env
	$(CONDA) install -y \
		-c conda-forge -c cdat \
		$(PKGS)

.PHONY: rerender
rerender: setup-env
	cd $(SRC_PATH)

	$(CONDA) smithy rerender \
		--feedstock_directory=$(FEEDSTOCK_PATH)

.PHONY: build
build: setup-env
	cd $(SRC_PATH)

	$(CONDA) build \
		-c conda-forge -c cdat \
		-m $(FEEDSTOCK_PATH)/.ci_support/linux_64_.yaml \
		--output-folder $(PWD)/output \
		$(CONDA_BUILD_EXTRA) \
		$(FEEDSTOCK_PATH) 

.PHONY: build-docs
build-docs: setup-env
	cd $(SRC_PATH)

	$(CONDA) install -c conda-forge sphinx sphinx_rtd_theme recommonmark m2r

	@make -C dodsrc/ html

	@cp -a dodsrc/build/html/. docs/
