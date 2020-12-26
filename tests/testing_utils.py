# Helper Functions
try:
    from src.utils import load_json
except ImportError as error:
    print(f"Error: {error}; Local modules not found")
except Exception as exception:
    print(exception)


def load_params_1():
    """Returns source path of images and number of exposures
    """
    PKG_1_PARAMS = load_json("config/pkg_1_config.json")

    return PKG_1_PARAMS


def load_params_2():
    """Returns source path of images and number of exposures
    """
    PKG_2_PARAMS = load_json("config/pkg_2_config.json")

    return PKG_2_PARAMS
