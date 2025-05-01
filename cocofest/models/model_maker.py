from cocofest.models.ding2003.ding2003 import DingModelFrequency
from cocofest.models.ding2003.ding2003_with_fatigue import DingModelFrequencyWithFatigue
from cocofest.models.ding2007.ding2007 import DingModelPulseWidthFrequency
from cocofest.models.ding2007.ding2007_with_fatigue import DingModelPulseWidthFrequencyWithFatigue
from cocofest.models.hmed2018.hmed2018 import DingModelPulseIntensityFrequency
from cocofest.models.hmed2018.hmed2018_with_fatigue import DingModelPulseIntensityFrequencyWithFatigue
from cocofest.models.marion2009.marion2009 import Marion2009ModelFrequency
from cocofest.models.marion2009.marion2009_with_fatigue import Marion2009ModelFrequencyWithFatigue
from cocofest.models.marion2009.marion2009_modified import Marion2009ModelPulseWidthFrequency
from cocofest.models.marion2009.marion2009_modified_with_fatigue import Marion2009ModelPulseWidthFrequencyWithFatigue
from cocofest.models.marion2013.marion2013 import Marion2013ModelFrequency
from cocofest.models.marion2013.marion2013_with_fatigue import Marion2013ModelFrequencyWithFatigue
from cocofest.models.marion2013.marion2013_modified import Marion2013ModelPulseWidthFrequency
from cocofest.models.marion2013.marion2013_modified_with_fatigue import Marion2013ModelPulseWidthFrequencyWithFatigue


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
