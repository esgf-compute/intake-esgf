import os
import tempfile

import pytest
import requests


@pytest.fixture(scope='session')
def test_data():
    data_path = os.path.join(os.getcwd(), 'clt.nc')

    if not os.path.exists(data_path):
        response = requests.get('https://cdat.llnl.gov/cdat/sample_data/clt.nc')

        response.raise_for_status()

        with open('clt.nc', 'wb') as f:
            for chunk in response.iter_content(2048):
                f.write(chunk)

    return data_path

@pytest.fixture
def preset():
    data = """
default: test
presets:
    test:
        constraints:
            project: test, test2
        fields:
            - id
            - name
    test_data:
        fields:
            - id
            - name
    """

    with tempfile.NamedTemporaryFile() as temp:
        temp.write(data.encode())

        temp.seek(0)

        yield temp.name
