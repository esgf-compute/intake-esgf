import json

import pytest
import pandas as pd
from intake_xarray.netcdf import NetCDFSource

from intake_esgf.core import ESGFCatalogEntry


def test_unsupported(test_data):
    data = [
        {"id": "test1", "url": [f"{test_data}||LAS", f"{test_data}||TEST"],},
    ]

    df = pd.DataFrame.from_dict(data)

    entry = ESGFCatalogEntry(df.iloc[0:1])

    with pytest.raises(Exception):
        entry.get()


def test_http(test_data):
    data = [
        {"id": "test1", "url": [f"{test_data}||HTTPServer"],},
    ]

    df = pd.DataFrame.from_dict(data)

    entry = ESGFCatalogEntry(df.iloc[0:1])

    data = entry.get()

    assert isinstance(data, NetCDFSource)


def test_opendap(test_data):
    data = [
        {"id": "test1", "url": [f"{test_data}||OPENDAP"],},
    ]

    df = pd.DataFrame.from_dict(data)

    entry = ESGFCatalogEntry(df.iloc[0:1])

    desc = entry.describe()

    expected_desc = {
        "name": "test1",
        "container": "xarray",
        "description": "",
        "direct_access": "allow",
        "user_parameters": [],
    }

    assert desc == expected_desc

    data = entry.get()

    assert isinstance(data, NetCDFSource)
