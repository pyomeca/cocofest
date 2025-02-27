import pickle
import numpy as np


class DataExtraction:
    @staticmethod
    def flatten(nested_list):
        """Flatten a list of lists."""
        return [item for sublist in nested_list for item in sublist]

    @staticmethod
    def apply_offset(self, time_list, offset):
        """Apply a cumulative offset to a list of time values."""
        return [t + offset for t in time_list]

    @staticmethod
    def load_and_adjust(file_path):
        """
        Load data from file and adjust stimulation and time data so that they start at zero.

        Returns:
            adjusted_time_data (list of lists): The time data with each row shifted if needed.
            adjusted_stim_times (list): The stimulation times shifted to start at 0.
            force_data (list): The muscle force data.
            raw_data (dict): The full data dictionary (for additional processing if needed).
        """
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        stim_times = data["stim_time"]
        # Adjust times if the first stimulation time is not zero
        if stim_times[0] != 0:
            adjusted_stim_times = [t - stim_times[0] for t in stim_times]
            adjusted_time_data = [[t - stim_times[0] for t in row] for row in data["time"]]
        else:
            adjusted_stim_times = stim_times
            adjusted_time_data = data["time"]
        return adjusted_time_data, adjusted_stim_times, data["force"], data

    @staticmethod
    def compute_stim_frequency(stim_times, multiplier=1.5):
        """
        Compute stimulation frequency based on valid intervals.

        A valid interval is one that is less than multiplier times the first interval.
        """
        if len(stim_times) < 2:
            return None
        threshold = stim_times[1] - stim_times[0]
        valid_intervals = [
            stim_times[j] - stim_times[j - 1]
            for j in range(1, len(stim_times))
            if (stim_times[j] - stim_times[j - 1]) < threshold * multiplier
        ]
        return round(1 / np.mean(valid_intervals), 0)

    @staticmethod
    def average_force_curves(force_data):
        """
        Average a list of force curves by truncating to the smallest length.

        Returns:
            avg_force (list): The averaged force curve.
            smallest_length (int): The length to which curves were truncated.
        """
        smallest_length = min(len(curve) for curve in force_data)
        avg_force = np.mean([curve[:smallest_length] for curve in force_data], axis=0).tolist()
        return avg_force, smallest_length

    def full_data_extraction(self, data_path):
        """
        Extracts full data from a list of file paths.

        For each file, the function:
          - Loads and adjusts the stimulation times and time data.
          - Offsets the times to ensure continuity across files.
          - Collects raw force data.

        Returns
        -------
        tuple
            (all_time_data, all_stim_times, all_force_data, discontinuity_phase_list)
        """
        all_force_data = []
        all_stim_times = []
        all_time_data = []
        discontinuity_phase_list = []

        cumulative_time_offset = 0
        cumulative_stim_count = 0

        for idx, file_path in enumerate(data_path):
            adj_time_data, adj_stim_times, force_data, _ = self.load_and_adjust(file_path)
            flat_time = self.flatten(adj_time_data)

            # If not the first file, apply the cumulative offset and record discontinuity
            if idx > 0:
                discontinuity_phase_list.append(cumulative_stim_count)
                flat_time = self.apply_offset(flat_time, cumulative_time_offset)
                adj_stim_times = self.apply_offset(adj_stim_times, cumulative_time_offset)

            if flat_time:
                cumulative_time_offset = flat_time[-1]
            cumulative_stim_count += len(adj_stim_times)

            all_force_data.extend(force_data)
            all_stim_times.extend(adj_stim_times)
            all_time_data.extend(flat_time)

        return all_time_data, all_stim_times, all_force_data, discontinuity_phase_list

    def average_data_extraction(self, model_data_paths, train_duration):
        """
        Extracts averaged data from a list of file paths.

        For each file, the function:
          - Loads and adjusts time data.
          - Computes stimulation frequency based on stimulation intervals.
          - Averages the force curves.
          - Generates an evenly spaced stimulation timeline.
          - Offsets times to ensure continuity across files.

        Parameters
        ----------
        model_data_paths : list of str
            Paths to the data files.
        train_duration : float
            Duration used to compute the number of points for the stimulation timeline.

        Returns
        -------
        tuple
            (all_time_data, all_stim_times, all_force_data, discontinuity_phase_list)
        """
        all_force_data = []
        all_stim_times = []
        all_time_data = []
        discontinuity_phase_list = []

        cumulative_time_offset = 0
        cumulative_stim_count = 0

        for idx, file_path in enumerate(model_data_paths):
            adj_time_data, adj_stim_times, force_data, raw_data = self.load_and_adjust(file_path)
            flat_time = self.flatten(adj_time_data)

            # Compute stimulation frequency and average the force curves
            frequency = self.compute_stim_frequency(adj_stim_times)
            avg_force, smallest_length = self.average_force_curves(force_data)

            # Truncate time data to match the averaged force curve length
            flat_time = flat_time[:smallest_length]

            # Generate stimulation instants based on computed frequency
            train_width = 1  # Assumed constant train width
            num_points = int(frequency * train_duration)
            stim_instants = np.linspace(0, train_width, num_points + 1)[:-1].tolist()
            if idx == len(model_data_paths) - 1 and flat_time:
                stim_instants.append(flat_time[-1])

            # Apply cumulative offset for continuity
            if idx > 0:
                discontinuity_phase_list.append(cumulative_stim_count)
                flat_time = self.apply_offset(flat_time, cumulative_time_offset)
                stim_instants = self.apply_offset(stim_instants, cumulative_time_offset)

            if flat_time:
                cumulative_time_offset = flat_time[-1]
            cumulative_stim_count += len(stim_instants)

            all_force_data.extend(avg_force)
            all_stim_times.extend(stim_instants)
            all_time_data.extend(flat_time)

        return all_time_data, all_stim_times, all_force_data, discontinuity_phase_list

    @staticmethod
    def force_at_node_in_ocp(time, force, n_shooting, final_time):
        """
        Interpolates the force at each node in the optimal control problem (OCP).

        Parameters
        ----------
        time : list
            List of time data.
        force : list
            List of force data.
        n_shooting : int
            List of number of shooting points for each phase.

        Returns
        -------
        list
            List of force at each node in the OCP.
        """

        temp_time = np.linspace(0, final_time, n_shooting + 1)
        force_at_node = list(np.interp(temp_time, time, force))
        return force_at_node
