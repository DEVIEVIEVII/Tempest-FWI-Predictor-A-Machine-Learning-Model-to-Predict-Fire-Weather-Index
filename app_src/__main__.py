from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


APP_FILE = Path(__file__).with_name("app.py")
SPEC = spec_from_file_location("fwi_predictor_app", APP_FILE)
MODULE = module_from_spec(SPEC)

if SPEC.loader is None:
    raise ImportError(f"Unable to load Flask app from '{APP_FILE}'.")

SPEC.loader.exec_module(MODULE)


if __name__ == "__main__":
    MODULE.app.run(debug=True)
