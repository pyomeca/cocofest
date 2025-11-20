import os
import importlib
import pytest

# List of examples to test
EXAMPLE_MODULES = [
    "examples.getting_started.identification.muscle_model_id",
    "examples.getting_started.initial_value_problem.pulse_mode_example",
    "examples.getting_started.initial_value_problem.model_integration",
    "examples.getting_started.optimization.force_tracking",
    "examples.getting_started.optimization.frequency_optimization",
    "examples.getting_started.optimization.pulse_intensity_optimization",
    "examples.getting_started.optimization.pulse_width_optimization",
    "examples.getting_started.optimization.pulse_width_optimization_nmpc",
    "examples.identification.force_model.ding2003_model_id",
    "examples.identification.force_model.ding2007_model_id",
    "examples.identification.force_model.hmed2018_model_id",
]


@pytest.mark.parametrize("module_name", EXAMPLE_MODULES)
def test_examples(module_name):
    ocp_module = importlib.import_module(module_name)
    ocp_module.main(plot=False)


# --- Get the path to the biomodels --- #
from examples.msk_models import init as model_path

biomodel_folder = os.path.dirname(model_path.__file__)


# List of examples to test
EXAMPLE_OTHER_MODEL = [
    "examples.other_fes_models.marion2009_example",
    "examples.other_fes_models.marion2013_example",
    "examples.other_fes_models.veltink1992_example",
]


@pytest.mark.parametrize("module_name", EXAMPLE_OTHER_MODEL)
@pytest.mark.parametrize("with_pulse_width", [False, True])
@pytest.mark.parametrize("with_fatigue", [False, True])
def test_other_model_examples(module_name, with_pulse_width, with_fatigue):
    ocp_module = importlib.import_module(module_name)
    if module_name == "examples.other_fes_models.veltink1992_example":
        if with_pulse_width:
            return
        ocp_module.main(with_fatigue=with_fatigue, plot=False)
    else:
        ocp_module.main(with_pulse_width=with_pulse_width, with_fatigue=with_fatigue, plot=False)


# --- Multibody examples --- #

MULTIBODY_EXAMPLE_MODULES = [
    "examples.getting_started.multibody.pulse_intensity_optimization_multibody",
    "examples.getting_started.multibody.pulse_width_optimization_multibody",
]

multibody_biorbd_model_path = biomodel_folder + "/Arm26/arm26_biceps_1dof.bioMod"


@pytest.mark.parametrize("module_name", MULTIBODY_EXAMPLE_MODULES)
def test_multibody_examples(module_name):
    ocp_module = importlib.import_module(module_name)
    ocp_module.main(plot=False, biorbd_path=multibody_biorbd_model_path)


MULTIBODY_CYCLING_EXAMPLE_MODULES = [
    "examples.fes_multibody.cycling.cycling_with_different_driven_methods",
]

cycling_biorbd_model_path = biomodel_folder + "/Wu/Modified_Wu_Shoulder_Model_Cycling.bioMod"
cycling_initial_guess_biorbd_model_path = biomodel_folder + "/Wu/Modified_Wu_Shoulder_Model_Cycling_for_IK.bioMod"

# TODO: Update it
# @pytest.mark.parametrize("module_name", MULTIBODY_CYCLING_EXAMPLE_MODULES)
# def test_cycling_multibody_examples(module_name):
#     ocp_module = importlib.import_module(module_name)
#     ocp_module.main(
#         plot=False,
#         model_path=cycling_biorbd_model_path,
#         initial_guess_model_path=cycling_initial_guess_biorbd_model_path,
#     )


MULTIBODY_FLEXION_EXAMPLE_MODULES = [
    "examples.fes_multibody.elbow_flexion.elbow_flexion_task",
]

flexion_biorbd_model_path = biomodel_folder + "/Arm26/arm26_biceps_triceps.bioMod"


@pytest.mark.parametrize("module_name", MULTIBODY_FLEXION_EXAMPLE_MODULES)
def test_flexion_multibody_examples(module_name):
    ocp_module = importlib.import_module(module_name)
    ocp_module.main(plot=False, model_path=flexion_biorbd_model_path)


MULTIBODY_REACHING_EXAMPLE_MODULES = [
    "examples.fes_multibody.reaching.reaching_task",
]

reaching_biorbd_model_path = biomodel_folder + "/Arm26/arm26_with_reaching_target.bioMod"


@pytest.mark.parametrize("module_name", MULTIBODY_REACHING_EXAMPLE_MODULES)
def test_reaching_multibody_examples(module_name):
    ocp_module = importlib.import_module(module_name)
    ocp_module.main(plot=False, model_path=reaching_biorbd_model_path)
