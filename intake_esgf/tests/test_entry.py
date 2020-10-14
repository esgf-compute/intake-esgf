import json

import pytest
import pandas as pd
from intake_xarray.netcdf import NetCDFSource

from intake_esgf import core

def test_from_dataframes_no_accessmethod(test_data):
    ds = pd.DataFrame.from_dict([
        {"id": "test1",},
    ])

    df = pd.DataFrame.from_dict([
        {"id": "file1", "url": [F"{test_data}||OPENDAP"],},
        {"id": "file1", "url": [F"{test_data}||LAS"],},
    ])

    with pytest.raises(core.ESGFEntryMissingDriver):
        core.ESGFCatalogEntry.from_dataframes(ds, df)

def test_from_dataframes(test_data):
    ds = pd.DataFrame.from_dict([
        {"id": "test1",},
    ])

    df = pd.DataFrame.from_dict([
        {"id": "file1", "url": [F"{test_data}||OPENDAP"],},
    ])

    entry = core.ESGFCatalogEntry.from_dataframes(ds, df)

    data = entry.get()

    assert isinstance(data, core.ESGFOpenDapSource)

def test_unsupported(test_data):
    ds = pd.DataFrame.from_dict([
        {"id": "test1", "url": [f"{test_data}||LAS", f"{test_data}||TEST"],},
    ])

    df = pd.DataFrame.from_dict([
        {"id": "file1",},
    ])

    entry = core.ESGFCatalogEntry(ds, df, "test", "opendap")

    with pytest.raises(Exception):
        entry.get()

def test_http(test_data):
    ds = pd.DataFrame.from_dict([
        {"id": "test1", "url": [f"{test_data}||HTTPServer"],},
    ])

    df = pd.DataFrame.from_dict([
        {"id": "file1",},
    ])

    entry = core.ESGFCatalogEntry(ds, df, "test", "esgf-opendap", args={"url": [test_data,]})

    data = entry.get()

    assert isinstance(data, core.ESGFOpenDapSource)

def test_opendap(test_data):
    ds = pd.DataFrame.from_dict([
        {"id": "test1", "url": [f"{test_data}||OPENDAP"],},
    ])

    df = pd.DataFrame.from_dict([
        {"id": "file1",},
    ])

    entry = core.ESGFCatalogEntry(ds, df, "test", "esgf-opendap", args={"url": [test_data,]})

    desc = entry.describe()

    print(desc)

    expected_desc = {
        "name": "test",
        "container": "xarray",
        "plugin": ["esgf-opendap"],
        "description": "",
        "direct_access": True,
        "user_parameters": [],
        "metadata": {},
        "args": {
            "url": [test_data,]
        },
    }

    assert desc == expected_desc

    data = entry.get()

    assert isinstance(data, core.ESGFOpenDapSource)
