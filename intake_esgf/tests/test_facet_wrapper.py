from intake_esgf import core


def test_facet_wrapper(mocker):
    wrapper = core.ESGFFacetWrapper(
        items={"variable": ["tas", "clt"], "frequency": "mon"}
    )

    assert wrapper["frequency"] == "mon"
    assert list(wrapper) == ["variable", "frequency"]
    assert len(wrapper) == 2
