import intake
from .core import ESGFCatalog
from .core import ESGFDefaultCatalog
from .core import ESGFCatalogError
from .core import PresetDoesNotExistError

def presets():
    import yaml
    import pkg_resources

    preset_file = pkg_resources.resource_string(__name__, "presets.yaml")

    return yaml.safe_load(preset_file)

def default_catalog():
    import yaml
    import pkg_resources

    preset_file = pkg_resources.resource_string(__name__, "default_catalog.yaml")

    return yaml.safe_load(preset_file)
