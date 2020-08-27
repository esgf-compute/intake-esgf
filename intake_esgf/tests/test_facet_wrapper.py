from intake_esgf import core


def test_facet_wrapper(mocker):
    wrapper = core.ESGFFacetWrapper(
        items={"variable": ["tas", "clt"], "frequency": "mon"}
    )

    assert wrapper["variable"] == ["tas", "clt"]

    assert list(wrapper) == ["variable", "frequency"]

    assert len(wrapper) == 2

    js = wrapper._repr_javascript_()

    assert js is not None

    import IPython.display

    display = mocker.spy(IPython.display, "display")

    wrapper._widget()

    wrapper._ipython_display_()

    mocker.patch.object(wrapper, "_widget", return_value=None)

    wrapper._ipython_display_()

    assert display.call_count == 1
