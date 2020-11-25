import json
import hashlib

import numpy as np
import pandas as pd
from intake_esgf import core


def test_fix_col():
    df = pd.DataFrame.from_dict([
        {"id": "test1", "single": ["item1",], "double": ["item1", "item2"]},
    ])

    df = df.apply(core.fix_col)

    assert df.single.to_list() == ["item1"]
    assert df.double.to_list() == [["item1", "item2"]]

def test_esgfopendapsource(test_data, mocker):
    dataset = pd.DataFrame()

    files = pd.DataFrame([
        {'variable': 'tas'},
        {'variable': 'clt'},
    ])

    data = core.ESGFOpenDapSource(
        [test_data],
        files=files,
        dataset=dataset,
        chunks={"time": 10},
        metadata={"source": "testdata"},
        xarray_kwargs={"decode_times": False})

    assert data._url == [test_data,]
    assert data._chunks == {"time": 10}
    assert data.metadata == {"source": "testdata"}

    data._open_dataset()

    assert data._ds is not None
    data.close()

    assert hashlib.sha256(json.dumps(data._get_schema()).encode()).hexdigest() == "152883fce815c51c497b7087122493cad31d2057899ba034b30a39f748b16fae"

    assert data.read() == data._ds
    assert data.read_chunked() == data._ds
    assert data.to_dask() == data._ds
    assert data.variables == ['tas', 'clt']

    vars = data.to_esgf_compute('tas')

    assert len(vars) == 1
    assert vars[0].uri == test_data
    assert vars[0].var_name == 'tas'
