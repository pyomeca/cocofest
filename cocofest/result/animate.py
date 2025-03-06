import pyorerun as prr
import biorbd
import pickle
import numpy as np


class PickleAnimate:
    def __init__(self, path: str):
        self.path = path
        self.data = None
        self.model = None
        self.time = None
        self.state_q = None
        self.frames = None

    def load(self):
        with open(self.path, "rb") as f:
            self.data = pickle.load(f)

        model_path = self.data["bio_model_path"] if "bio_model_path" in self.data.keys() else None
        if model_path is None:
            raise ValueError("The bio model path is not available, please provide it to animate the solution.")

        # Load a predefined model
        if self.model is None:
            self.model = biorbd.Model(model_path)
        self.time = self.data["time"]
        self.state_q = self.data["states"]["q"]
        self.state_q = self.state_q if self.state_q.ndim == 2 else np.expand_dims(self.state_q, axis=0)
        self.frames = self.state_q.shape[1]

    def animate(self, model: biorbd.Model = None):
        self.model = model
        self.load()

        # pyorerun animation
        prr_model = prr.BiorbdModel.from_biorbd_object(self.model)

        nb_seconds = self.time[-1]
        t_span = np.linspace(0, nb_seconds, self.frames)

        viz = prr.PhaseRerun(t_span)
        viz.add_animated_model(prr_model, self.state_q)
        viz.rerun("msk_model")

    def multiple_animations(self, additional_path: list[str], model: biorbd.Model = None):
        self.model = model
        self.load()
        nb_seconds = self.time[-1]
        t_span = np.linspace(0, nb_seconds, self.frames)
        prr_model = prr.BiorbdModel.from_biorbd_object(self.model)

        # pyorerun animation
        rerun_biorbd = prr.MultiPhaseRerun()
        rerun_biorbd.add_phase(t_span=t_span, phase=0, window="animation")
        rerun_biorbd.add_animated_model(prr_model, self.state_q, phase=0, window="animation")

        for path in additional_path:
            with open(path, "rb") as f:
                data = pickle.load(f)

            state_q = data["states"]["q"]
            frames = state_q.shape[1]

            t_span = np.linspace(0, nb_seconds, frames)
            prr_model = prr.BiorbdModel.from_biorbd_object(self.model)

            rerun_biorbd.add_phase(t_span=t_span, phase=0, window="split_animation")
            rerun_biorbd.add_animated_model(prr_model, state_q, phase=0, window="split_animation")

        rerun_biorbd.rerun("multi_model_test")
