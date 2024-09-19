import pickle
import numpy as np
from bioptim import SolutionMerge


class SolutionToPickle:
    def __init__(self, solution, file_name, path):
        self.sol = solution
        self.file_name = file_name
        self.path = path

    def pickle(self):
        bounds_key = self.sol.ocp.parameter_bounds.keys()
        bounds = {}
        for key in bounds_key:
            bounds[key] = (
                self.sol.ocp.parameter_bounds[key].min[0][0],
                self.sol.ocp.parameter_bounds[key].max[0][0],
            )

        time = self.sol.decision_time(
            to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES]
        )
        time = time.reshape(time.shape[0])

        time, states = self.remove_duplicates(time)

        dictionary = {
            "time": time,
            "states": states,
            "control": self.sol.decision_controls(
                to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES]
            ),
            "parameters": self.sol.decision_parameters(),
            "parameters_bounds": bounds,
            "time_to_optimize": self.sol.real_time_to_optimize,
            "bio_model_path": (
                self.sol.ocp.nlp[0].model.bio_model.path
                if hasattr(self.sol.ocp.nlp[0].model, "bio_model")
                else None
            ),
        }

        with open(self.path + self.file_name, "wb") as file:
            pickle.dump(dictionary, file)

        return print(
            f"Solution values has been exported in pickle format in {self.path + self.file_name}"
        )

    def remove_duplicates(self, time):
        states = self.sol.decision_states(
            to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES]
        )
        vals, idx_start, count = np.unique(time, return_counts=True, return_index=True)
        time = time[idx_start]
        state_keys = states.keys()
        for key in state_keys:
            if states[key].shape[0] == 1:
                states[key] = states[key][0][idx_start]
            else:
                temps_states = []
                for dim in range(states[key].shape[0]):
                    temps_states.append(states[key][dim][idx_start])
                states[key] = np.array(temps_states)

        return time, states
