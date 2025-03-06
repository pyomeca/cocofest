import os
import importlib
import pytest

# List of examples to test
EXAMPLE_MODULES = [
    "examples.getting_started.force_tracking_parameter_optimization",
    "examples.getting_started.frequency_optimization",
    "examples.getting_started.model_integration",
    "examples.getting_started.muscle_model_id",
    "examples.getting_started.pulse_intensity_optimization",
    "examples.getting_started.pulse_mode_example",
    "examples.getting_started.pulse_width_optimization",
    "examples.getting_started.pulse_width_optimization_nmpc",
    "examples.identification.ding2003_model_id",
    "examples.identification.ding2007_model_id",
    "examples.identification.hmed2018_model_id",
]


@pytest.mark.parametrize("module_name", EXAMPLE_MODULES)
def test_examples(module_name):
    ocp_module = importlib.import_module(module_name)
    ocp_module.main(plot=False)


MULTIBODY_EXAMPLE_MODULES = [
    "examples.getting_started.frequency_optimization_multibody",
    "examples.getting_started.pulse_intensity_optimization_multibody",
    "examples.getting_started.pulse_width_optimization_multibody",
]
from examples.model_msk import init as model_path

biomodel_folder = os.path.dirname(model_path.__file__)
biorbd_model_path = biomodel_folder + "/arm26_biceps_1dof.bioMod"


@pytest.mark.parametrize("module_name", MULTIBODY_EXAMPLE_MODULES)
def test_multibody_examples(module_name):
    ocp_module = importlib.import_module(module_name)
    ocp_module.main(plot=False, biorbd_path=biorbd_model_path)
