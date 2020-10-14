import tempfile

import pytest

from intake_esgf import core

def test_load(preset, mocker):
    search_data = {
        "response": {
            "numFound": 2,
            "docs": [
                {"id": "id1", "name": "test1",},
                {"id": "id2", "name": "test2",},
            ],
        }}

    get = mocker.patch("requests.get")
    get.return_value.json.return_value = search_data

    cat = core.ESGFCatalog(preset_file=preset)

    df = cat.load()

    assert df is not None
    assert len(df) == 2

def test_facet_values(preset, mocker):
    cat = core.ESGFCatalog(preset_file=preset)

    get = mocker.patch("requests.get")
    get.return_value.json.return_value = {
        "facet_counts": {
            "facet_fields": {
                "id": [
                    "test", 2,
                ],
                "name": [
                    "name1", 4,
                ]
            }
        }
    }

    values = cat.facet_values()

    assert len(values) == 2
    assert values["id"] == ["test"]

    values = cat.facet_values(True)

    assert len(values) == 2
    assert values["id"] == ["test (2)"]

def test_search(preset, mocker):
    search_data = {
        "response": {
            "numFound": 2,
            "docs": [
                {"id": "id1", "name": "test1",},
                {"id": "id2", "name": "test2",},
            ],
        }}

    facet_data = {
        "facet_counts": {"facet_fields": {"variable": ["tas", "2", "clt", "4"],},},}

    get = mocker.patch("requests.get")
    get.return_value.json.side_effect = [
        search_data,
        facet_data]

    facets = {"variable": ["tas", "clt"]}
    params = {"limit": 10}
    fields = ["frequency", "model"]

    cat = core.ESGFCatalog(
        preset_file=preset, preset="test", facets=facets, params=params, fields=fields)

    matches = cat.search(frequency="mon")

    list(matches)
    assert matches.df is not None
    assert len(matches) == 2

    assert cat._url == matches._url
    assert cat._fields == matches._fields
    assert cat._constraints == matches._constraints
    assert cat._limit == matches._limit
    assert cat._params == matches._params
    assert matches._facets == {"variable": ["tas", "clt"], "frequency": "mon"}


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
        facet_data]

    facets = {"variable": ["tas", "clt"]}
    params = {"limit": 10}
    fields = ["frequency", "model"]

    cat = core.ESGFCatalog(
        preset_file=preset, preset="test", facets=facets, params=params, fields=fields)

    list(cat)

    filtered = cat.filter(lambda x: x.id == "id1")

    assert len(filtered) == 1

def test_catalog_default_preset(preset):
    cat = core.ESGFCatalog(preset_file=preset)

    cat._load_preset()

    assert cat._presets != {}

    cat._load_preset()

    assert cat._preset == "test"

def test_default_catalog():
    cat = core.ESGFDefaultCatalog()

    assert cat is not None

def test_catalog_bad_preset(preset):
    facets = {"variable": ["tas", "clt"]}
    params = {"limit": 10}
    fields = ["frequency", "model"]

    cat = core.ESGFCatalog(preset_file=preset, preset="test2")

    with pytest.raises(core.PresetDoesNotExistError):
        cat._load_preset()
