import yaml

from intake_esgf import tools


def test_build_defaults(mocker):
    get = mocker.patch("requests.get")

    data = {"response": {"docs": [{"id": 1, "variable": "tas", "institute": "LLNL",}]}}

    get.return_value.json.return_value = data

    defaults = tools.build_defaults("https://localhost:9999", ["CMIP5"])

    expected = yaml.dump(
        {
            "presets": {
                "CMIP5": {
                    "constraints": {"project": "CMIP5",},
                    "fields": ["id", "variable", "institute",],
                }
            }
        }
    )

    assert isinstance(defaults, str)
    assert defaults == expected
