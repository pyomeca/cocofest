from .ding2003 import DingModelFrequency
from .ding2003_with_fatigue import DingModelFrequencyWithFatigue
from .ding2007 import DingModelPulseWidthFrequency
from .ding2007_with_fatigue import DingModelPulseWidthFrequencyWithFatigue
from .hmed2018 import DingModelPulseIntensityFrequency
from .hmed2018_with_fatigue import DingModelPulseIntensityFrequencyWithFatigue
from .marion2009 import Marion2009ModelFrequency
from .marion2009_with_fatigue import Marion2009ModelFrequencyWithFatigue
from .marion2009_modified import Marion2009ModelPulseWidthFrequency
from .marion2009_modified_with_fatigue import Marion2009ModelPulseWidthFrequencyWithFatigue
from .marion2013 import Marion2013ModelFrequency
from .marion2013_with_fatigue import Marion2013ModelFrequencyWithFatigue
from .marion2013_modified import Marion2013ModelPulseWidthFrequency
from .marion2013_modified_with_fatigue import Marion2013ModelPulseWidthFrequencyWithFatigue


class ModelMaker:
    @staticmethod
    def create_model(model_type, **kwargs):
        model_dict = {
            "ding2003": DingModelFrequency,
            "ding2003_with_fatigue": DingModelFrequencyWithFatigue,
            "ding2007": DingModelPulseWidthFrequency,
            "ding2007_with_fatigue": DingModelPulseWidthFrequencyWithFatigue,
            "hmed2018": DingModelPulseIntensityFrequency,
            "hmed2018_with_fatigue": DingModelPulseIntensityFrequencyWithFatigue,
            "marion2009": Marion2009ModelFrequency,
            "marion2009_with_fatigue": Marion2009ModelFrequencyWithFatigue,
            "marion2009_modified": Marion2009ModelPulseWidthFrequency,
            "marion2009_modified_with_fatigue": Marion2009ModelPulseWidthFrequencyWithFatigue,
            "marion2013": Marion2013ModelFrequency,
            "marion2013_with_fatigue": Marion2013ModelFrequencyWithFatigue,
            "marion2013_modified": Marion2013ModelPulseWidthFrequency,
            "marion2013_modified_with_fatigue": Marion2013ModelPulseWidthFrequencyWithFatigue,
        }
        if model_type not in model_dict:
            raise ValueError(f"Unknown model type: {model_type}")
        return model_dict[model_type](**kwargs)
