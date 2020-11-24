import datetime
import math
import os
import logging
from collections import abc
from collections import OrderedDict
from contextlib import contextmanager

import requests
import pandas as pd
import yaml
import pkg_resources
from intake import catalog
from intake.catalog import exceptions
from intake.catalog import local
from intake.source import base

logger = logging.getLogger("intake_esgf.core")


class ESGFCatalogError(exceptions.CatalogException):
    pass


class PresetDoesNotExistError(ESGFCatalogError):
    pass

class ESGFEntryMissingDriver(ESGFCatalogError):
    def __init__(self, dataset_id):
        msg = f"Dataset {dataset_id!r} could not be handled by any intake drivers"
        super(ESGFEntryMissingDriver, self).__init__(msg)


drivers = OrderedDict([
    ("OPENDAP", "esgf-opendap"),
    ("HTTPServer", "netcdf"),
])


def fix_col(x):
    if isinstance(x[0], (list, tuple)) and len(x[0]) == 1:
        return x[0][0]
    return x


class ESGFOpenDapSource(base.DataSource):
    """ESGF OPENDap source.
    """
    version = "1.0.0"
    name = "esgf-opendap"
    container = "xarray"
    partition_access = False

    def __init__(self, url, chunks=None, xarray_kwargs=None, metadata=None, **kwargs):
        """ESGFOpenDapSource __init__.

        Args:
            url: URL of ESGF search node.
            chunks: Mapping of coords and chunk sizes.
            xarray_kwargs: Arguments passed to xarray.
            metadata: Metadata describing the source.
        """
        if not isinstance(url, list):
            url = list(url)

        self._url = url
        self._chunks = chunks
        self._kwargs = xarray_kwargs or kwargs
        self._ds = None

        super(ESGFOpenDapSource, self).__init__(metadata=metadata)

    def _open_dataset(self):
        """Opens an xarray dataset.

        Opens dataset depending on number of of source files.
        """
        import xarray as xr

        logger.info(f"Opening data with {len(self._url)} files")

        if len(self._url) > 1:
            self._ds = xr.open_mfdataset(self._url, chunks=self._chunks, **self._kwargs)
        else:
            self._ds = xr.open_dataset(self._url[0], chunks=self._chunks, **self._kwargs)

    def to_esgf_compute(self, variable):
        """Converts urls to cwt variables.
        """
        import cwt

        return [cwt.Variable(x, variable) for x in self._url]

    def _get_schema(self):
        """Builds a schema for the datasource.
        """
        if self._ds is None:
            self._open_dataset()

            metadata = {
                "dims": dict(self._ds.dims),
                "data_vars": {k: list(self._ds[k].coords)
                              for k in self._ds.data_vars.keys()},
                "coords": tuple(self._ds.coords.keys()),
            }

            metadata.update(self._ds.attrs)

            self._schema = base.Schema(
                datashape=None,
                dtype=None,
                shape=None,
                npartitions=None,
                extra_metadata=metadata
            )

        return self._schema

    def read(self):
        "Reads dataset."
        return self.read_chunked()

    def read_chunked(self):
        "Reads dataset as chunked."
        self._load_metadata()

        return self._ds

    def to_dask(self):
        "Loads dataset for dask."
        return self.read_chunked()

    def close(self):
        "Closes a dataset."
        self._ds = None

        self._schema = None

class ESGFCatalogEntry(local.LocalCatalogEntry):
    """ESGFs catalog entry.
    """
    def __init__(self, dataset, files, name, driver, args=None, parameters=None, catalog=None):
        """ESGFCatalogEntry __init__.

        Args:
            dataset: DataFrame containing dataset details.
            files: DataFrame containing file details.
            name: Name of the catalog entry.
            driver: Driver to load data.
            args: Arguments passed to driver.
            parameters: List of parameters.
            catalog: Instance of parent catalog.
        """
        self._dataset = dataset
        self._files = files

        super().__init__(
            name,
            "",
            driver,
            args=args or {},
            parameters=parameters or [],
            catalog=catalog,
        )

    @classmethod
    def from_dataframes(cls, dataset, files, catalog=None):
        """Creates catalog entry from dataframes.

        Args:
            dataset: DataFrame detailing dataset.
            files: DataFrame detailing files associated with dataset.
            catalog: Parent catalog.
        """
        name = dataset.id.values[0]

        logger.info(f"Building datasource for {name}")

        access = [x.split("|")[-1] for x in files.url.values[0]]

        logger.info(f"Access methods {access}")

        candidates = [(method, drivers[method]) for method in drivers if method in access]

        logger.info(f"Candidate methods {candidates}")

        def get_file_access(access_methods, method):
            for x in access_methods:
                parts = x.split("|")

                if parts[-1] == method:
                    return parts[0].replace(".html", "")

            raise ESGFEntryMissingDriver(name)

        args = {
            "url": files.url.apply(get_file_access, method=candidates[0][0]).to_list(),
        }

        return cls(dataset, files, name, candidates[0][1], args=args, catalog=catalog)


class ESGFCatalogEntries(abc.Mapping):
    """Collection of catalog entries.
    """
    def __init__(self, catalog):
        """ESGFCatalogEntries __init__.

        Args:
            catalog: Reference to the catalog whose entries being managed.
        """
        self._catalog = catalog
        self._num_found = None
        self._df = None

    def __repr__(self):
        return f"<{self.__class__.__name__}: {len(self)}>"

    def __getitem__(self, key):
        """Gets item from collection.

        Args:
            key: Key for item being retrieved.

        Returns:
            A ESGFCatalogEntry identified by key.
        """
        entry = self._df[self._df.id == key].iloc[0:1]

        if len(entry) == 0:
            raise KeyError(key)

        files = self._catalog.fetch_files(entry.id.values[0])

        return ESGFCatalogEntry.from_dataframes(entry, files, self._catalog)

    def _next_page(self, offset):
        """Gets next page of search results.

        Args:
         offset: Offset get retrieve results from.

        Returns:
            A Pandas Dataframe containing a page of search results.
        """
        limit = self._catalog._limit

        logger.info(f"Page offset {offset}")

        try:
            page = self._df.iloc[offset : offset + limit]
        except AttributeError:
            logger.info(f"Found not cached page for offset {offset}")
        else:
            logger.info(f"Found cached page for offset {offset}")

            if len(page) > 0:
                return page

        self._num_found, page = self._catalog.fetch_page(offset)

        page = page.set_index(pd.RangeIndex(offset, offset + len(page)))

        logger.info(f"Adjusted index to {offset} - {offset+len(page)}")

        if self._df is None:
            self._df = page
        else:
            self._df = self._df.append(page, ignore_index=True)

        return page

    def pages(self):
        """Generates search result pages.

        Yields:
            A Pandas Dataframe containing search results.
        """
        offset = 0
        limit = self._catalog._limit

        while offset < self._num_found:
            page = self._next_page(offset)

            page_size = len(page)

            logger.info(f"Got page offset {offset} len {page_size}")

            yield page

            offset += page_size

    def num_pages(self):
        """Returns the number of pages."""
        return math.ceil(len(self) / self._catalog._limit)

    def __iter__(self):
        """Generates identifiers for each entry."""
        for page in self.pages():
            for _, x in page.iterrows():
                yield x.id

    def __contains__(self, key):
        """Tests if key is in collection."""
        if self._df is None:
            raise KeyError(key)

        return key in self._df.id.values

    def __len__(self):
        """Returns the number of entries in collection."""
        if self._df is None:
            self._next_page(0)

        return self._num_found or len(self._df)

    def _widget_header(self):
        from ipywidgets import Layout, Button, HBox

        facets = []
        button_layout = Layout(height="px", width="150px")
        for x, y in self._catalog._facets.items():
            if isinstance(y, str):
                y = y.split(",")

            for z in y:
                desc = f"{x}: {z}"
                facets.append(Button(description=desc, button_style="success"))

        header = HBox(facets)
        header.layout.flex_flow = "row wrap"
        header.layout.align_items = "flex-start"
        header.layout.align_content = "flex-start"

        return header

    def _widget_footer(self, output):
        from ipywidgets import Button, HBox, Label

        next = Button(description="Next")
        prev = Button(description="Previous")

        page_label = Label()
        page_label.value = f"{self._widget_current_page} - {self._widget_max_pages}"

        footer = HBox([prev, page_label, next])
        footer.layout.justify_content = "center"
        footer.layout.align_items = "center"

        def update_buttons():
            prev.disabled = True if self._widget_current_page == 1 else False
            next.disabled = (
                True if self._widget_current_page == self._widget_max_pages else False
            )

        @contextmanager
        def disable_buttons(*args, **kwargs):
            old_next = next.disabled
            old_prev = prev.disabled

            prev.disabled = True
            next.disabled = True

            yield

            prev.disabled = False
            next.disabled = False

        def next_page(b):
            self._widget_current_page += 1
            offset = (self._widget_current_page - 1) * self._catalog._limit
            data = self._next_page(offset)
            with output, disable_buttons():
                output.clear_output(wait=True)
                display(data)
            update_buttons()
            page_label.value = f"{self._widget_current_page} - {self._widget_max_pages}"

        def prev_page(b):
            self._widget_current_page -= 1
            offset = (self._widget_current_page - 1) * self._catalog._limit
            data = self._next_page(offset)
            with output, disable_buttons():
                output.clear_output(wait=True)
                display(data)
            update_buttons()
            page_label.value = f"{self._widget_current_page} - {self._widget_max_pages}"

        next.on_click(next_page)
        prev.on_click(prev_page)

        update_buttons()

        return footer

    def _widget(self):
        try:
            from IPython.display import display
            from ipywidgets import (
                AppLayout,
                HTML,
                Output,
                Button,
                HBox,
                GridspecLayout,
                Layout,
                Label,
            )
        except ImportError:
            return None

        self._widget_current_page = 0
        self._widget_max_pages = self.num_pages()

        output = Output()
        center = HBox([output,])

        header = self._widget_header()

        data = self._next_page(0)
        self._widget_current_page = 1
        self._widget_max_pages = self.num_pages()

        footer = self._widget_footer(output)

        with output:
            display(data)

        container = AppLayout(
            left_sidebar=None,
            center=center,
            right_sidebar=None,
            header=header,
            footer=footer,
            pane_heights=["120px", "400px", "50px"],
        )

        return container

    def _ipython_display_(self, **kwargs):
        widget = self._widget()

        if widget is not None:
            widget._ipython_display_(**kwargs)

class ESGFFacetWrapper(abc.Mapping):
    """Wraps ESGF facets.
    """
    def __init__(self, items=None):
        """ESGFFacetWrapper __init__.

        Args:
            items: A dict mapping facets and values.
        """
        super().__init__()

        self._items = items or {}

    def __repr__(self):
        return f"<{self.__class__.__name__}: {len(self._items)}>"

    def __getitem__(self, key):
        """Gets item identified by key.

        Args:
            key: Key identifying item.

        Returns:
            A list of str values.
        """
        return self._items[key]

    def __iter__(self):
        """Generates a list of facets names."""
        for x in self._items:
            yield x

    def __len__(self):
        """Returns the number of facets."""
        return len(self._items)

    def _ipython_display_(self, **kwargs):
        widget = self._widget()

        if widget is not None:
            widget._ipython_display_(**kwargs)

    def _widget(self):
        try:
            from ipywidgets import AppLayout, Layout, Dropdown, Select, Output
        except ImportError:
            return None

        expanded = Layout(width="auto", height="200px")

        facets = Dropdown(options=sorted(list(self._items)))

        values = Select(options=self._items[facets.value], layout=expanded)

        output = Output()

        def load_values(kwargs):
            new_value = kwargs["new"]

            values.options = sorted(self._items[new_value])

        facets.observe(load_values, names="value")

        container = AppLayout(
            left_sidebar=facets, center=values, right_sidebar=None, footer=output
        )

        return container


class ESGFDefaultCatalog(local.YAMLFileCatalog):
    """ESGFDefaultCatalog.

    Loads a preset YAML catalog.
    """
    name = "esgf-default-catalog"

    def __init__(self, *args, **kwargs):
        """ESGFDefaultCatalog __init__.

        Args:
            *args: Position arguments.
            **kwargs: Key/value arguments.
        """
        default_catalog_path = pkg_resources.resource_filename(
            __name__, "default_catalog.yaml"
        )

        super().__init__(default_catalog_path, *args, **kwargs)


class ESGFCatalog(catalog.Catalog):
    """ESGFCatalog.
    """
    name = "esgf-catalog"

    def __init__(self, url=None, params=None, fields=None, constraints=None, facets=None,
                 limit=10000, preset=None, presets=None, preset_file=None, metadata=None):
        """ESGFCatalog __init__.

        Args:
            url: URL of an ESGF index node.
            params: A dict of search parameters.
            fields: A list of fields to retrieve.
            constraints: A dict mapping facets to values that constrain the results.
            facets: A dict mapping facet names and values to search for.
            limit: Number of maximum results per page.
            preset: Facet preset.
            presets: A dict containing preset definitions.
            preset_file: A str path to a file containing facet presets.
            metadata: A dict containing catalog metadata.
        """
        self._required_fields = [
            "dataset_id",
            "master_id",
            "instance_id",
            "id",
            "url",
            "data_node",
            "size"
        ]

        self._default_params = {
            "format": "application/solr+json",
            "type": "Dataset",
            "distrib": "true",
            "replica": "false",
            "latest": "true",
        }

        self._params = params or {}

        # Search config
        self._url = url or "https://esgf-node.llnl.gov/esg-search/search"
        self._fields = fields or []
        self._constraints = constraints or {}
        self._facets = facets or {}
        self._limit = limit

        # Preset config
        default_preset_file = pkg_resources.resource_filename(__name__, "presets.yaml")
        self._preset_file = preset_file or default_preset_file
        self._presets = presets
        self._preset = preset

        super().__init__(name=self._preset, metadata=metadata)

    @property
    def df(self):
        """Returns DataFrame containing search results."""
        return self._entries._df

    def filter(self, func):
        """Filters current search results.

        Args:
            func: Function applied to DataFrame.

        Returns:
            An ESGFCatalog containing a subset of the search results.
        """
        entries = ESGFCatalogEntries(None)

        entries._df = self._entries._df[func]

        cat = ESGFCatalog.from_dict(
            entries,
            url=self._url,
            params=self._params,
            fields=self._fields,
            constraints=self._constraints,
            facets=self._facets,
            limit=self._limit,
            preset=self._preset,
            presets=self._presets,
            preset_file=self._preset_file,
        )

        entries._catalog = cat

        return cat

    def search(self, **kwargs):
        """Searchs for datasets matching facets.

        Args:
            **kwargs: Facets to search for datasets.

        Returns:
            An ESGFCatalog representing the search results.
        """
        new_facets = self._facets.copy()
        new_facets.update(kwargs)

        logger.info(f"Creating new catalog with search facets {new_facets!r}")

        cat = ESGFCatalog(
            url=self._url,
            params=self._params,
            fields=self._fields,
            constraints=self._constraints,
            facets=new_facets,
            limit=self._limit,
            preset=self._preset,
            presets=self._presets,
            preset_file=self._preset_file,
        )

        return cat

    def _load_preset(self):
        """Loads facet presets from file."""
        if self._presets is not None:
            return

        logger.info(f"Loading preset from {self._preset_file!r}")

        with open(self._preset_file) as f:
            data = yaml.safe_load(f.read())

        if self._preset is None:
            self._preset = data["default"]

            logger.info(f"Setting default preset to {self._preset!r}")

        self._presets = data["presets"]

        try:
            preset = data["presets"][self._preset]
        except KeyError as e:
            raise PresetDoesNotExistError(e)
        else:
            self._fields += preset["fields"]
            self._constraints = preset.get("constraints", {})

            logger.info(f"Preset fields: {self._fields} constraints: {self._constraints}")

    def facet_values(self, include_count=False):
        """Retrieves facets and values.

        Args:
            include_count: Include number of entries for each value.

        Returns:
            An ESGFFacetWrapper containing search facets and possible values.
        """
        self._load_preset()

        params = {
            "format": "application/solr+json",
            "facets": list(set(self._fields)),
            "limit": 0,
        }
        params.update(self._constraints)
        params.update(self._facets)

        for x in params:
            if isinstance(params[x], (list, tuple)):
                params[x] = ",".join(params[x])

        logger.info(f"Retrieving facet values with parameters {params!r}")

        response = requests.get(self._url, params)

        response.raise_for_status()

        data = response.json()

        output = {}

        for x, y in data["facet_counts"]["facet_fields"].items():
            if include_count:
                output[x] = [f"{a} ({b})" for a, b in zip(y[::2], y[1::2])]
            else:
                output[x] = y[::2]

        logger.info(f"Retrieved {len(output)} facets with values")

        return ESGFFacetWrapper(output)

    def _search_params(self, facets=None):
        """Builds parameters for search request.

        Args:
            facets: A dict containing override facets.

        Returns:
            A dict passed to requests calls.
        """
        params = self._default_params.copy()
        # user fields can override the defaults
        params.update(self._params)
        params["fields"] = list(set(self._required_fields + self._fields))
        params.update(self._facets)
        # user facets can override the defaults
        params.update(facets or {})
        params.update(self._constraints)
        # ensure some defaults are not over written by user
        params["type"] = "Dataset"
        params["format"] = "application/solr+json"
        params["limit"] = self._limit

        for x in params:
            if isinstance(params[x], (tuple, list)):
                params[x] = ",".join(params[x])

        return params

    def fetch_files(self, dataset_id):
        """Fetchs files associated with a dataset.

        Args:
            dataset_id: Dataset identified used for searching.

        Returns:
            A DataFrame containing the file results for a dataset.
        """
        params = self._search_params()
        params["offset"] = 0
        params["query"] = f"dataset_id:{dataset_id}"
        params["type"] = "File"

        logger.info(f"Fetching dataset {dataset_id!r} files parameters {params!r}")

        response = requests.get(self._url, params=params)

        response.raise_for_status()

        data = response.json()

        df = pd.DataFrame.from_dict(data["response"]["docs"])

        df = df.apply(fix_col)

        logger.info(f"Retrieved {len(df)} files associated with dataset {dataset_id!r}")

        return df

    def fetch_page(self, offset=0, **kwargs):
        """Fetchs a page of search results.

        Args:
            offset: Search offset.
            **kwargs: Facets and values to search with.

        Returns:
            A DataFrame containing a page of search results.
        """
        params = self._search_params(kwargs)
        params["offset"] = offset

        logger.info(f"Fetching search result page with parameters {params!r}")

        response = requests.get(self._url, params=params)

        response.raise_for_status()

        data = response.json()

        num_found = data["response"]["numFound"]

        df = pd.DataFrame.from_dict(data["response"]["docs"])

        df = df.apply(fix_col)

        logger.info(f"Retrieved {len(df)} results")

        return num_found, df

    def load(self):
        """Loads all pages from search results.

        Returns:
            A DataFrame containing all results from search.
        """
        pages = self._entries.num_pages()

        logger.info(f"Loading {pages} pages of search results")

        try:
            from ipywidgets import IntProgress
            from IPython.display import display
        except ImportError:
            progress = None
        else:
            progress = IntProgress(min=0, max=pages)

            display(progress)

        for x in self._entries.pages():
            if progress is not None:
                progress.value += 1

        return self._entries._df

    def _make_entries_container(self):
        return ESGFCatalogEntries(self)

    def _ipython_display_(self):
        return self._entries._ipython_display_()
