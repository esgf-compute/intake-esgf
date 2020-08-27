import pandas
import pytest

from intake_esgf import core


def test_entries(preset, mocker):
    cat = core.ESGFCatalog(
        preset_file=preset, preset="test", storage_options={"path": "/tmp"}
    )

    data = [
        {"id": "test1",},
        {"id": "test2",},
    ]

    entries = core.ESGFCatalogEntries(cat)
    entries._df = pandas.DataFrame.from_dict(data)

    contains = "test1" in entries
    assert contains

    entry = entries["test1"]
    assert entry is not None
    assert entry.storage_options == {"path": "/tmp"}

    subset = entries[0:2]
    assert len(subset) == 2

    response = {"response": {"numFound": 10, "docs": data}}

    entries._df = None
    get = mocker.patch("requests.get")
    get.return_value.json.return_value = response

    with pytest.raises(KeyError):
        "test1" in entries

    subset = entries[0:2]

    assert len(subset) == 2

    with pytest.raises(KeyError):
        entries["missing"]


def test_multiple_pages(preset, mocker):
    cat = core.ESGFCatalog(preset_file=preset, preset="test", limit=2)

    entries = core.ESGFCatalogEntries(cat)

    data = [
        {"response": {"numFound": "2", "docs": [{"id": "test1"}, {"id": "test2"},]}},
        {"response": {"numFound": "2", "docs": [{"id": "test3"}, {"id": "test4"},]}},
    ]

    get = mocker.patch("requests.get")
    get.return_value.json.side_effect = data

    page = entries[0:2]
    assert len(page) == 2

    page = entries[2:4]
    print(entries._df)
    assert len(page) == 2


def test_ipython_display(preset, mocker):
    facets = {"variable": "tas,clt", "frequency": "mon"}
    cat = core.ESGFCatalog(preset_file=preset, preset="test", limit=2, facets=facets)

    entries = core.ESGFCatalogEntries(cat)

    data = [
        {"response": {"numFound": 2, "docs": [{"id": "test1"}, {"id": "test2"},]}},
        {"response": {"numFound": 2, "docs": [{"id": "test3"}, {"id": "test4"},]}},
    ]

    get = mocker.patch("requests.get")
    get.return_value.json.side_effect = data

    widget = entries._ipython_display_()

    assert widget is None

    import IPython.display

    display = mocker.spy(IPython.display, "display")

    mocker.patch.object(entries, "_widget", return_value=None)

    entries._ipython_display_()

    assert display.call_count == 1
