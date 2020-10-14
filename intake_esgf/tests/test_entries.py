import pytest
import pandas as pd

from intake_esgf import core

def test_multiple_pages(mocker):
    cat = core.ESGFCatalog(limit=2)

    entries = core.ESGFCatalogEntries(cat)

    get = mocker.patch("requests.get")
    get.return_value.json.side_effect = [
        {
            "response": {
                "numFound": 4,
                "docs": [
                    {"id": "test1"},
                    {"id": "test2"},
                ]
            }
        },
        {
            "response": {
                "numFound": 4,
                "docs": [
                    {"id": "test3"},
                    {"id": "test4"},
                ]
            }
        }
    ]

    page = entries._next_page(offset=0)

    assert len(page) == 2

    page = entries._next_page(offset=2)

    assert len(page) == 2

    assert entries.num_pages() == 2


def test_esgfcatalogentries(mocker):
    cat = core.ESGFCatalog()

    entries = core.ESGFCatalogEntries(cat)

    with pytest.raises(KeyError):
        assert "test1" not in entries

    entries._df = pd.DataFrame.from_dict(
        [
            {"id": "test1"},
            {"id": "test2"},
        ])

    entries._num_found = 2

    assert "test1" in entries

    get = mocker.patch("requests.get")
    get.return_value.json.return_value = {
        "response": {
            "docs": [
                {"master_id": "test1", "url": ["/data.nc|test|OPENDAP", "/data.nc|test|HTTPServer"]},
                {"master_id": "test2", "url": ["/data.nc|test|OPENDAP", "/data.nc|test|HTTPServer"]},
            ]
        }
    }

    entry = entries["test1"]

    assert entry is not None

    with pytest.raises(KeyError):
        entries["test10"]

    all_entries = list(entries)

    assert len(all_entries) == 2
