import tempfile

import pytest

from intake_esgf import core
from intake_esgf.core import ESGFCatalog


def test_fetch_raw(preset, mocker):
    search_data = {
        "response": {
            "numFound": 2,
            "docs": [{"id": "id1", "name": "test1",}, {"id": "id2", "name": "test2",},],
        }
    }

    get = mocker.patch("requests.get")

    get.return_value.json.side_effect = [
        search_data,
    ]

    cat = ESGFCatalog(preset_file=preset, preset="test")

    page = cat.fetch_page(raw=True)

    assert page == search_data


def test_search(preset, mocker):
    search_data = {
        "response": {
            "numFound": 2,
            "docs": [{"id": "id1", "name": "test1",}, {"id": "id2", "name": "test2",},],
        }
    }

    facet_data = {
        "facet_counts": {"facet_fields": {"variable": ["tas", "2", "clt", "4"],},},
    }

    get = mocker.patch("requests.get")

    get.return_value.json.side_effect = [
        search_data,
        facet_data,
    ]

    facets = {"variable": ["tas", "clt"]}
    params = {"limit": 10}
    fields = ["frequency", "model"]

    cat = ESGFCatalog(
        preset_file=preset, preset="test", facets=facets, params=params, fields=fields
    )

    matches = cat.search(frequency="mon")

    assert len(matches) == 2

    assert cat._url == matches._url
    assert cat.fields == matches.fields
    assert cat.constraints == matches.constraints
    assert cat._limit == matches._limit
    assert cat.params == matches.params
    assert cat._requests_kwargs == matches._requests_kwargs
    assert cat._storage_options == matches._storage_options
    assert matches.facets == {"variable": ["tas", "clt"], "frequency": "mon"}
    assert matches._skip_load


def test_filter(preset, mocker):
    search_data = {
        "response": {
            "numFound": 2,
            "docs": [{"id": "id1", "name": "test1",}, {"id": "id2", "name": "test2",},],
        }
    }

    facet_data = {
        "facet_counts": {"facet_fields": {"variable": ["tas", "2", "clt", "4"],},},
    }

    get = mocker.patch("requests.get")

    get.return_value.json.side_effect = [
        search_data,
        facet_data,
    ]

    facets = {"variable": ["tas", "clt"]}
    params = {"limit": 10}
    fields = ["frequency", "model"]

    cat = ESGFCatalog(
        preset_file=preset, preset="test", facets=facets, params=params, fields=fields
    )

    list(cat)

    filtered = cat.filter(lambda x: x.id == "id1")

    assert len(filtered) == 1


def test_catalog_ipython_display(preset, mocker):
    cat = ESGFCatalog(preset_file=preset)

    import IPython.display

    display = mocker.spy(IPython.display, "display")

    cat._ipython_display_()

    assert display.call_count == 2


def test_catalog_default(preset):
    cat = ESGFCatalog(preset_file=preset)

    assert cat._preset == "test"


def test_catalog_bad_preset(preset):
    facets = {"variable": ["tas", "clt"]}
    params = {"limit": 10}
    fields = ["frequency", "model"]

    with pytest.raises(core.PresetDoesNotExistError):
        ESGFCatalog(preset_file=preset, preset="test2")


def test_catalog(test_data, preset, mocker):
    search_data = {
        "response": {
            "numFound": 2,
            "docs": [
                {
                    "id": "id1",
                    "name": "test1",
                    "url": [f"{test_data}||OPENDAP", f"{test_data}||HTTPServer",],
                    "single_value": ["hello",],
                },
                {"id": "id2", "name": "test2",},
            ],
        }
    }

    facet_data = {
        "facet_counts": {"facet_fields": {"variable": ["tas", "2", "clt", "4"],},},
    }

    get = mocker.patch("requests.get")

    get.return_value.json.side_effect = [
        search_data,
        facet_data,
        facet_data,
    ]

    facets = {"variable": ["tas", "clt"]}
    params = {"limit": 10}
    fields = ["frequency", "model"]

    cat = ESGFCatalog(
        preset_file=preset, preset="test", facets=facets, params=params, fields=fields
    )

    assert len(cat) == 2
    assert get.call_count == 1

    assert list(cat) == ["id1", "id2"]
    assert get.call_count == 1

    assert cat.entries
    assert cat.facets == facets
    assert cat.params == params
    assert cat.fields == fields
    assert cat.constraints == {"project": "test, test2"}

    entry = cat["id1"]
    assert entry

    facets = cat.facet_values()

    assert list(facets) == ["variable"]
    assert facets["variable"] == ["tas", "clt"]

    facets = cat.facet_values(True)

    assert list(facets) == ["variable"]
    assert facets["variable"] == ["tas (2)", "clt (4)"]

    type(get.return_value).status_code = mocker.PropertyMock(return_value=400)

    with pytest.raises(Exception):
        cat.facet_values()
