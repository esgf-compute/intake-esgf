import math
import os
import logging
from collections import abc
from contextlib import contextmanager

import intake
import requests
import pandas as pd
import yaml
import pkg_resources

logger = logging.getLogger("intake_esgf.core")


class ESGFCatalogError(intake.catalog.exceptions.CatalogException):
    pass


class PresetDoesNotExistError(ESGFCatalogError):
    pass


class ESGFCatalogEntry(intake.catalog.entry.CatalogEntry):
    def __init__(self, df, **kwargs):
        """ ESGFCatalogEntry.

        Args:
            df (pandas.DataFrame): Dataframe holding search results.
            chunks (dict): Mapping dimensions to chunk sizes.
            container (str, list): List of containers to attempt to open data with.
            xarray_kwargs (dict): Keyword options to be passed to xarray open functions.
            storage_options (dict): Keyword options to be password to fsspec open_files.
            description (str): Description of the catalog entry.
        """
        self._df = df
        self.chunks = kwargs.get("chunks", {})
        self.containers = kwargs.get("container", "netcdf")
        self.xarray_kwargs = kwargs.get("xarray_kwargs", {})
        self.storage_options = kwargs.get("storage_options", {})
        self.description = kwargs.get("description", "")

        if not isinstance(self.containers, list):
            self.containers = [self.containers]

        self.container_map = {
            "netcdf": ["OPENDAP", "HTTPServer"],
        }

        super().__init__()

    def get(self):
        """ Opens source using appropriate driver. """
        access = dict(
            (x.split("|")[-1], x.split("|")[0].replace(".html", ""))
            for x in self._df.url.to_list()[0]
        )

        data = None

        for container in self.containers:
            for access_method in self.container_map[container]:
                if access_method in access:
                    data = intake.registry[container](
                        access[access_method],
                        self.chunks,
                        xarray_kwargs=self.xarray_kwargs,
                        storage_options=self.storage_options,
                    )

                    break

        if data is None:
            raise ESGFCatalogError(
                f"Did not find container for any access methods {', '.join(list(access))}."
            )

        data._entry = self

        return data

    def describe(self):
        """ Describes the entry. """
        return {
            "name": self._df.id.to_list()[0],
            "container": "xarray",
            "description": self.description,
            "direct_access": "allow",
            "user_parameters": [],
        }


class ESGFCatalogEntries(abc.Mapping, abc.Sequence):
    def __init__(self, catalog):
        """ ESGFCatalogEntries.

        Args:
            catalog (ESGFCatalog): Reference to the catalog whose entries being managed.
        """
        self._catalog = catalog
        self._offset = 0
        self._num_found = 0
        self._df = None

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def __getitem__(self, key):
        """ Get entry.

        Args:
            key (str): Key for item being retrieved.

        Returns:
            ESGFCatalogEntry: Entry related to key.
        """
        if isinstance(key, slice):
            try:
                value = self._df.iloc[key]

                if len(value) == 0:
                    raise IndexError()
            except (AttributeError, IndexError):
                logger.info(f"Missing entries for {key}")

                for x in range(key.start, key.stop, self._catalog._limit):
                    self._next_page(x)

                value = self._df.iloc[key]
        else:
            entry = self._df[self._df.id == key].iloc[0:1]

            if len(entry) == 0:
                raise KeyError(key)

            value = ESGFCatalogEntry(
                entry,
                container=self._catalog._container,
                chunks=self._catalog._chunks,
                storage_options=self._catalog._storage_options,
            )

        return value

    def _next_page(self, offset):
        """ Retrieves next search result page.

        Args:
         offset (int): Offset get retrieve results from.

        Returns:
            pandas.DataFrame: DataFrame holding search results.
        """
        limit = self._catalog._limit

        logger.info(f"Requesting page @ offset {offset} length {limit}")

        try:
            page = self._df.iloc[offset : offset + limit]
        except AttributeError:
            logger.info(f"Did not find page for offset {offset}")
        else:
            if len(page) > 0:
                return page

        self._num_found, page = self._catalog.fetch_page(offset)

        self._offset += len(page)

        page = page.set_index(pd.RangeIndex(offset, offset + len(page)))

        if self._df is None:
            self._df = page
        else:
            self._df = self._df.append(page, ignore_index=True)

        return page

    def __iter__(self):
        """ Iterates over search results. """
        offset = 0

        while True:
            page = self._next_page(offset)

            for _, x in page.iterrows():
                yield x.id

            offset += len(page)

            if offset >= self._num_found:
                break

    def __contains__(self, key):
        """ Checks if map contains key. """
        if self._df is None:
            raise KeyError(key)

        return key in self._df.id.values

    def __len__(self):
        """ Number of entries in map. """
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
        self._widget_max_pages = 0

        output = Output()
        center = HBox([output,])

        header = self._widget_header()

        data = self._next_page(0)
        self._widget_current_page = 1
        self._widget_max_pages = math.ceil(self._num_found / self._catalog._limit)

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
        else:
            from IPython.display import display

            display(self._df)


class ESGFFacetWrapper(abc.Mapping):
    def __init__(self, items=None):
        super().__init__()

        self._items = items or {}

    def __getitem__(self, key):
        return self._items[key]

    def __iter__(self):
        for x in self._items:
            yield x

    def __len__(self):
        return len(self._items)

    def _repr_javascript_(self):
        return f"""
let values = {{ {",".join([f"{x}: {y}" for x, y in self._items.items()]) } }};

let style = document.createElement("style");
style.innerHTML = `
.esgf-facet-container {{
    display: flex;
    flex-direction: row;
    align-items: flex-start;
    width: 100%;
}}

.esgf-facet-names {{
    margin: 8px;
}}

.esgf-facet-values {{
    margin: 8px;
    flex: 1;
    overflow: auto;
    height: 200px;
}}
`;

let container = document.createElement("div");
container.setAttribute("class", "esgf-facet-container");

let nameSelect = document.createElement("SELECT");
nameSelect.setAttribute("id", "facet-keys");
nameSelect.setAttribute("class", "esgf-facet-names");

Object.keys(values).sort().forEach(function (value, i) {{
    let c = document.createElement("option");
    c.text = value;
    c.value = value;
    nameSelect.options.add(c, i);
}});

nameSelect.onchange = function() {{
    valueSelect.options.length = 0;
    let selection = document.getElementById("facet-keys").value;
    values[selection].sort().forEach(function (value, i) {{
        let c = document.createElement("option");
        c.text = value;
        valueSelect.options.add(c, i);
    }});
}};

let valueSelect = document.createElement("SELECT");
valueSelect.setAttribute("class", "esgf-facet-values");
valueSelect.setAttribute("multiple", true);

values[nameSelect.value].sort().forEach(function (value, i) {{
    let c = document.createElement("option");
    c.text = value;
    valueSelect.options.add(c, i);
}});

container.appendChild(nameSelect);
container.appendChild(valueSelect);
element.appendChild(container);
element.appendChild(style);
        """

    def _ipython_display_(self, **kwargs):
        widget = self._widget()

        if widget is not None:
            widget._ipython_display_(**kwargs)
        else:
            from IPython.display import display

            mimetype = {
                "text/plain": repr(self),
                "text/javascript": self._repr_javascript_(),
            }

            display(mimetype, raw=True)

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


class ESGFDefaultCatalog(intake.catalog.local.YAMLFileCatalog):
    name = "esgf-default-catalog"

    def __init__(self, *args, **kwargs):
        default_catalog_path = pkg_resources.resource_filename(
            __name__, "default_catalog.yaml"
        )

        super().__init__(default_catalog_path, *args, **kwargs)


class ESGFCatalog(intake.catalog.Catalog):
    name = "esgf-catalog"

    def __init__(self, *args, **kwargs):
        """ ESGFCatalog.

        Args:
            url (str): URL of ESGF search service.
            fields (list): List of fields returned from search.
            constraints (dict): Search constraints, the same as facets.
            limit (int): The number of results returned per page.
            facets (dict): Default facets used for search.
            params (dict): Options for ESGF search service, see `keywords` https://earthsystemcog.org/projects/cog/esgf_search_restful_api.
            requests_kwargs (dict): Options passed to `requests` calls.
            storage_options (dict): Options for FSSpec when used by containers.
            chunks (dict): Chunking used when loading data.
            preset_file (str): Path to a preset file.
            preset (str): Name of the preset to use.
            skip_load (bool): If True then presets are not loaded.
            container (list): List of containers to use. These are used to
                find the appropriate container for the access methods available.
                Order does matter.
        """
        self._minimum_fields = ["id", "url"]
        self._default_params = {
            "format": "application/solr+json",
            "type": "File",
        }

        # Search config
        self._url = kwargs.get("url", "https://esgf-node.llnl.gov/esg-search/search")
        self._fields = kwargs.get("fields", [])
        self._constraints = kwargs.get("constraints", {})
        self._limit = kwargs.get("limit", 100)
        self._facets = kwargs.get("facets", {})
        self._params = kwargs.get("params", {})

        # Requests config
        self._requests_kwargs = kwargs.get("requests_kwargs", {})

        # FSSpec config
        self._storage_options = kwargs.get("storage_options", {})

        # Container config
        self._chunks = kwargs.get("chunks", None)

        # Preset config
        default_preset_file = pkg_resources.resource_filename(__name__, "presets.yaml")
        preset_file = kwargs.get("preset_file", default_preset_file)
        preset_file = os.environ.get("INTAKE_ESGF_PRESET_FILE", preset_file)
        self._preset_file = preset_file
        self._preset = kwargs.get("preset", None)
        self._skip_load = kwargs.get("skip_load", False)

        self._container = kwargs.get("container", "netcdf")

        self._load_preset()

        super().__init__()

    @property
    def entries(self):
        """ Cached search entries.

        Returns:
            pandas.DataFrame: DataFrame of cached entries.
        """
        return self._entries

    @property
    def facets(self):
        return self._facets

    @property
    def params(self):
        return self._params

    @property
    def fields(self):
        return self._fields

    @property
    def constraints(self):
        """ Preset constraints. """
        return self._constraints

    def filter(self, func):
        """ Filters the current set of reuslts.

        Args:
            func (func): Function used to filter the results.

        Returns:
            ESGFCatalog: Catalog with filter results.

        Examples:
            >>> import intake_esgf
            >>> cat = intake_esgf.ESGFCatalog()
            >>> page = cat.fetch_page(variable="clt,tas", frequency="mon")
            >>> results = cat.filter(lambda x: x.variable.to_list()[0] == "tas")
        """
        entries = ESGFCatalogEntries(None)

        entries._df = pd.concat([y for _, y in self._entries._df.iterrows() if func(y)])

        if not isinstance(entries._df, pd.DataFrame):
            entries._df = entries._df.to_frame().transpose()

        cat = ESGFCatalog.from_dict(
            entries,
            url=self._url,
            fields=self._fields,
            constraints=self._constraints,
            limit=self._limit,
            params=self._params,
            requests_kwargs=self._requests_kwargs,
            storage_options=self._storage_options,
            facets=self._facets,
            skip_load=True,
            container=self._container,
            chunks=self._chunks,
        )

        entries._catalog = cat

        return cat

    def search(self, **kwargs):
        """ Refines the search criteria.

        Args:
            **kwargs: Key/value pairs of facets.

        Returns:
            ESGFCatalog: Catalog with new search criteria.

        Examples:
            >>> import intake_esgf
            >>> cat = intake_esgf.ESGFCatalog()
            >>> results = cat.search(variable="clt,tas", frequency="mon", model=["amip", "amip4k"])
        """
        new_facets = self._facets.copy()
        new_facets.update(kwargs)

        cat = ESGFCatalog(
            url=self._url,
            fields=self._fields,
            constraints=self._constraints,
            limit=self._limit,
            params=self._params,
            requests_kwargs=self._requests_kwargs,
            storage_options=self._storage_options,
            facets=new_facets,
            skip_load=True,
            container=self._container,
            chunks=self._chunks,
        )

        cat.cat = self

        return cat

    def _load_preset(self):
        if not self._skip_load:
            with open(self._preset_file) as f:
                data = yaml.safe_load(f.read())

            if self._preset is None:
                self._preset = data["default"]

            try:
                preset = data["presets"][self._preset]
            except KeyError as e:
                raise PresetDoesNotExistError(e)
            else:
                self._fields += preset["fields"]
                self._constraints = preset.get("constraints", {})
            finally:
                self._skip_load = True

    def facet_values(self, include_count=False):
        """ Gets facets based on current criteria.

        Args:
            include_count (bool): Include counts for each value.

        Returns:
            dict: Map of facet names and possible values.
        """
        params = {
            "format": "application/solr+json",
            "facets": ",".join(list(set(self._fields))),
            "limit": 0,
        }
        params.update(self._constraints)
        params.update(self._facets)

        response = requests.get(self._url, params, **self._requests_kwargs)

        if response.status_code == 400:
            raise Exception(
                "Client error, please check facet names/values for correctness."
            )

        response.raise_for_status()

        data = response.json()

        output = {}

        for x, y in data["facet_counts"]["facet_fields"].items():
            if include_count:
                output[x] = [f"{a} ({b})" for a, b in zip(y[::2], y[1::2])]
            else:
                output[x] = y[::2]

        return ESGFFacetWrapper(output)

    def __getitem__(self, key):
        e = self._entries[key]
        e._catalog = self
        e._pmode = self.pmode

        return e()

    def fetch_page(self, offset=0, raw=False, **facets):
        """ Fetchs a page of search results.

        `facets` are key/value pairs of facets. The value can be str or list of str.

        Args:
            offset (int): Offset of search results.
            raw (bool): Returns raw JSON rather than pandas DataFrame.
            **facets: Additional facets used for search parameters.

        Returns:
            dict: Dictionary when `raw` is True.
            pandas.DataFrame: DataFrame container page of search results.

        Examples:
            >>> import intake_esgf
            >>> cat = intake_esgf.ESGFCatalog()
            >>> page = cat.fetch_page(variable="clt")
        """
        params = self._params.copy()
        params.update(self._facets)
        params.update(self._default_params)
        params["offset"] = offset
        params["limit"] = self._limit
        params["fields"] = list(set(self._minimum_fields + self._fields))
        params.update(self._constraints)
        params.update(facets)

        for x in params:
            if isinstance(params[x], (tuple, list)):
                params[x] = ",".join(params[x])

        response = requests.get(self._url, params=params, **self._requests_kwargs)

        response.raise_for_status()

        data = response.json()

        if raw:
            return data

        num_found = data["response"]["numFound"]

        df = pd.DataFrame.from_dict(data["response"]["docs"])

        def fix_col(x):
            if isinstance(x[0], (list, tuple)) and len(x[0]) == 1:
                return x[0][0]
            return x

        df = df.apply(fix_col)

        return num_found, df

    def _make_entries_container(self):
        return ESGFCatalogEntries(self)

    def _ipython_display_(self):
        return self._entries._ipython_display_()
