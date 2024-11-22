from .ding2003 import DingModelFrequency
from .ding2003_with_fatigue import DingModelFrequencyWithFatigue
from .ding2007 import DingModelPulseWidthFrequency
from .ding2007_with_fatigue import DingModelPulseWidthFrequencyWithFatigue
from .hmed2018 import DingModelPulseIntensityFrequency
from .hmed2018_with_fatigue import DingModelPulseIntensityFrequencyWithFatigue


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
        }
        if model_type not in model_dict:
            raise ValueError(f"Unknown model type: {model_type}")
        return model_dict[model_type](**kwargs)
