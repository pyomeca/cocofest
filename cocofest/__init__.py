from . import _matplotlib_compat  # Temporary fix
from .misc.__version__ import __version__
from .custom_objectives import CustomObjective
from .custom_constraints import CustomConstraint
from .models.fes_model import FesModel
from cocofest.models.ding2003.ding2003 import DingModelFrequency
from cocofest.models.ding2003.ding2003_with_fatigue import DingModelFrequencyWithFatigue
from cocofest.models.ding2007.ding2007 import DingModelPulseWidthFrequency
from cocofest.models.ding2007.ding2007_with_fatigue import DingModelPulseWidthFrequencyWithFatigue
from cocofest.models.marion2009.marion2009 import Marion2009ModelFrequency
from cocofest.models.marion2009.marion2009_with_fatigue import Marion2009ModelFrequencyWithFatigue
from cocofest.models.marion2009.marion2009_modified import Marion2009ModelPulseWidthFrequency
from cocofest.models.marion2009.marion2009_modified_with_fatigue import Marion2009ModelPulseWidthFrequencyWithFatigue
from cocofest.models.marion2013.marion2013 import Marion2013ModelFrequency
from cocofest.models.marion2013.marion2013_with_fatigue import Marion2013ModelFrequencyWithFatigue
from cocofest.models.marion2013.marion2013_modified import Marion2013ModelPulseWidthFrequency
from cocofest.models.marion2013.marion2013_modified_with_fatigue import Marion2013ModelPulseWidthFrequencyWithFatigue
from cocofest.models.hmed2018.hmed2018 import DingModelPulseIntensityFrequency
from cocofest.models.hmed2018.hmed2018_with_fatigue import DingModelPulseIntensityFrequencyWithFatigue
from cocofest.models.veltink1992.veltink1992 import VeltinkModelPulseIntensity
from cocofest.models.veltink1992.veltink1992_and_riener1998 import VeltinkRienerModelPulseIntensityWithFatigue
from .models.dynamical_model import FesMskModel
from .models.model_maker import ModelMaker
from .optimization.fes_ocp import OcpFes
from .optimization.fes_id_ocp import OcpFesId
from .optimization.fes_ocp_multibody import OcpFesMsk
from .optimization.fes_mhe import FesMhe
from .optimization.fes_mhe_multibody import FesMheMsk
from .integration.ivp_fes import IvpFes
from .identification.identification_method import DataExtraction
from .fourier_approx import FourierSeries
from .dynamics.inverse_kinematics_and_dynamics import (
    get_circle_coord,
    inverse_kinematics_cycling,
    inverse_dynamics_cycling,
)
from .result.plot import PlotCyclingResult
from .result.pickle import SolutionToPickle

# from .result.animate import PickleAnimate
from .result.graphics import FES_plot
