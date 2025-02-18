from datetime import datetime
import json
import numpy as np
from collections.abc import Iterable

from qm import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from configuration_NV2QEG import *


import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


class NVExperiment:
    def __init__(self, custom_config=None):
        self.qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name, octave=octave_config)

        # containers for commands
        self.var_vec = None
        self.commands = []
        self.use_fixed = False
        self.measure_len = None
        self.measure_mode = None
        self.measure_channel = None
        self.initialize = False
        self.measure_delay = 0
        self.laser_channel = None

        # containers for results
        self.counts0 = None
        self.counts_ref0 = None
        self.counts1 = None
        self.counts_ref1 = None
        self.iteration = None

        # store current config
        if custom_config is None:
            self.config = config
        else:
            self.config = custom_config

    def add_pulse(self, name, element, amplitude, length=x180_len_NV, cycle=False):
        """
        Adds a type "microwave" command to the experiment on the
        desired `element`.

        Args:
            name (string): Name of the pulse. 8 predefined pulses are avaialble,
                "+/-" * "x/y" * "90/180", eg "y180" or "-x90"
            element (string): Channel to play the pulse on, like "NV" or "C13" in the config
            amplitude (float?): amplitude of pulse
            variable (bool): If True, the pulse amplitude is a variable defined by the loop.
        """
        command = {
            "type": "pulse",
            "element": element,
            "name": name,
            "cycle": cycle,
        }
        if isinstance(amplitude, Iterable):
            command["scale"] = self.update_loop(amplitude)
            self.use_fixed = True
            command["length"] = length
        elif isinstance(length, Iterable):
            command["scale"] = self.update_loop(length)
            command["amplitude"] = amplitude
        else:
            command["length"] = length
            command["amplitude"] = amplitude
        self.commands.append(command)

    def add_cw_drive(self, element, length, amplitude):
        """
        Adds a type "microwave" command to the experiment on the
        desired `element`.

        Args:
            element (string): Channel to play the pulse on, like "NV" or "C13" in the config
            length (int): time of pulse in ns
            amplitude (float): amplitude of pulse
        """
        command = {"type": "cw", "element": element}
        if isinstance(amplitude, Iterable):
            command["scale"] = self.update_loop(amplitude)
            self.use_fixed = True
            command["length"] = length
        elif isinstance(length, Iterable):
            command["scale"] = self.update_loop(length)
            command["amplitude"] = amplitude
        else:
            command["length"] = length
            command["amplitude"] = amplitude
        self.commands.append(command)

    def add_measure_delay(self, length=meas_len_1):
        """
        Adds a type "measure_delay" command to the experiment.

        Args:
            length (int): time of measurement acquisition in ns
        """
        self.measure_delay = length

    def add_laser(self, mode="laser_ON", channel="AOM1", length=initialization_len_1):
        """
        Adds a type "laser" command to the experiment

        Args:
            name (string): command name
            length (int): time of laser illumination in ns
        """
        self.commands.append({"type": "laser", "mode": mode, "channel": channel, "length": length})
        self.laser_channel = channel

    def add_align(self):
        """
        Adds a type "align" command to the experiment.
        """
        self.commands.append({"type": "align"})

    def add_wait(self, length):
        """
        Adds a type "wait" command to the experiment.

        Args:
            length (int): time to wait in ns
            variable (bool): If True, the wait time is a variable defined by the loop.
        """
        if isinstance(length, Iterable):
            scale = self.update_loop(length)
            self.commands.append({"type": "wait", "scale": scale})
        else:
            self.commands.append({"type": "wait", "length": length})

    def add_measure(self, mode="readout", channel="SPCM1", meas_len=meas_len_1):
        """
        Adds a type "measure" command to the experiment.

        Args:
            mode (string): Measurement mode, like "readout" or "long_readout"
            channel (string): Channel to measure on, like "SPCM1" in the config
            meas_len (int): time of measurement acquisition in ns
        """
        self.commands.append({"type": "measure", "channel": channel, "mode": mode, "meas_len": meas_len})
        if self.measure_len is None:
            self.measure_len = meas_len
            self.measure_mode = mode
            self.measure_channel = channel
        elif self.measure_len != meas_len:
            raise ValueError("Inconsistent measurement lengths.")

    def add_frequency_update(self, element, freq_list):
        """
        Adds a type "update_frequency" command to the experiment.

        Args:
            element (string): Name of the element to update the frequency of
            freq_list (array): Array of frequencies to update the element to
        """
        self.commands.append({"type": "update_frequency", "element": element})
        self.update_loop(freq_list)

    def update_loop(self, var_vec):
        """
        Updates the variable vector for the experiment. This is used to define the loop
        that the experiment will run over. If the variable vector is already defined, this
        function will check that the new vector is consistent with the previous one by determining
        if the new vector is a constant multiple of the old one.

        Args:
            var_vec (array): Array of values for the variable in the experiment

        Returns:
            float: The constant multiple of the new vector to the old vector, 1 if this is the first update.

        Raises:
            ValueError: Throws an error if the new vector is not a constant multiple of the old one, or if
                the new vector is all zeros.
        """
        if np.all(var_vec == 0):
            raise ValueError("Variable vector cannot be all zeros.")

        if self.var_vec is None:
            self.var_vec = var_vec
            return 1

        two = self.var_vec
        if np.dot(var_vec, two) * np.dot(two, var_vec) == np.dot(var_vec, var_vec) * np.dot(two, two):
            div = -1
            idx = 0
            while div < 0:
                div = two[idx] / var_vec[idx] if var_vec[idx] != 0 else -1
                idx += 1
            if div > 0:
                return div

        raise ValueError("Inconsistent loop variables.")

    def add_initialization(self):
        """
        Adds a laser pulse to polarize the system before the first sequence. This is controlled with the config file.
        """
        self.initialize = True

    def setup_cw_odmr(self, f_vec, readout_len=long_meas_len_1, wait_time=1_000, amplitude=1):  # vector of frequencies
        """
        A pre-fab collection of commands to run a continuous wave ODMR experiment.

        Args:
            f_vec (array): Array of frequencies to sweep over
            readout_len (int): time of measurement acquisition in ns
            wait_time (int, optional): Wait time after CW before readout. Should exceed metastable state lifetime.
                Defaults to 1_000.
            amplitude (int, optional): Amplitude of the microwave drive. Defaults to 1.
        """
        self.add_align()
        self.add_frequency_update("NV", f_vec)

        self.add_laser(mode="laser_ON", channel="AOM1", length=readout_len)
        self.add_cw_drive("NV", readout_len, amplitude)

        self.add_wait(wait_time)
        self.add_measure(channel="SPCM1", mode="long_readout", meas_len=readout_len)
        self.add_measure_delay(1_000)

    def setup_time_rabi(self, t_vec=np.arange(4, 400, 1)):
        """
        A pre-fab collection of commands to run a Rabi experiment sweeping time of MW.

        Args:

        """
        self.add_initialization()  # pass element here to assign hardware channel
        self.add_pulse("x180", "NV", amplitude=1, length=t_vec)
        self.add_align()
        self.add_laser(mode="laser_ON", channel="AOM1")  # pass element here to assign hardware channel
        self.add_measure(channel="SPCM1")

    def _translate_command(self, command, var, times, counts, counts_st, invert):
        """
        Helper function whcih translates a command dictionary into a QUA command. Plays qua commands, can only
        be called from within a qua program.

        Args:
            command (dict): Command dictionary

        Returns:
            qua command: The QUA command
        """
        scale = command.get("scale", 1)
        match command["type"]:
            case "update_frequency":
                update_frequency(command["element"], var)

            case "pulse":
                amplitude = command.get("amplitude", var * scale)
                length = command.get("length", var * scale)
                name = command["name"]
                if invert and command["cycle"]:
                    if name[0] == "-":
                        name = name[1:]
                    else:
                        name = "-" + name
                play(name * amp(amplitude), command["element"], duration=length)

            case "cw":
                amplitude = command.get("amplitude", var * scale)
                length = command.get("length", var * scale)
                play("cw" * amp(amplitude), command["element"], duration=length)

            case "wait":
                duration = command.get("length", var * scale)
                wait(duration)

            case "laser":
                play(command["mode"], command["channel"], duration=command["length"])

            case "measure":
                measure(
                    command["mode"],
                    command["channel"],
                    None,
                    time_tagging.analog(times, command["meas_len"], counts),
                )
                save(counts, counts_st)

            case "align":
                return align()

    def _reference_counts(self, times, counts, counts_st, pi_amp):
        """
        Wrapper for measuring reference counts. Plays qua commands, can only be called from
        within a qua program.

        """
        wait(wait_between_runs)
        align()

        play("x180" * amp(pi_amp), "NV")  # Pi-pulse toggle
        align()

        if self.measure_delay > 0:
            wait(self.measure_delay, self.measure_channel)
            play("laser_ON", self.laser_channel, duration=self.measure_len)
        else:
            play("laser_ON", self.laser_channel)
        measure(self.measure_mode, self.measure_channel, None, time_tagging.analog(times, self.measure_len, counts))

        save(counts, counts_st)  # save counts
        wait(wait_between_runs, self.laser_channel)

    def create_experiment(self, n_avg, measure_contrast):
        """
        Creates the Quantum Machine program for the experiment, and returns the
        experiment object as a qua `program`. This is used by the `execute_experiment` and
        `simulate_experiment` methods.

        Args:
            n_avg (int, optional): Number of averages for each data acquisition point.
            measure_contrast (bool): If True, only the |0> state is measured, if False, both |0> and |1> are measured.

        Returns:
            program: The QUA program for the experiment defined by this class's commands.
        """

        with program() as experiment:

            # define the variables and datastreams
            counts0 = declare(int)
            counts0_st = declare_stream()
            counts_ref0 = declare(int)
            counts_ref0_st = declare_stream()

            if not measure_contrast:
                counts1 = declare(int)
                counts1_st = declare_stream()
                counts_ref1 = declare(int)
                counts_ref1_st = declare_stream()

            times = declare(int, size=100)  # QUA vector for storing time-tags

            if self.use_fixed:
                var = declare(fixed)
            else:
                var = declare(int)

            n = declare(int)  # averaging var
            n_st = declare_stream()  # stream for number of iterations

            # start the experiment
            if self.initialize:
                play("laser_ON", self.laser_channel)
                wait(wait_for_initialization, self.laser_channel)

            # turn on microwave
            sg384_NV.set_amplitude(NV_LO_amp)
            sg384_NV.set_frequency(NV_LO_freq)
            sg384_NV.ntype_on(1)
            sg384_NV.do_set_Modulation_State("ON")
            sg384_NV.do_set_modulation_type("IQ")

            with for_(n, 0, n < n_avg, n + 1):  # averaging loop
                with for_(*from_array(var, self.var_vec)):  # scanning loop

                    # do the sequence as defined by the commands, measure |0>
                    for command in self.commands:
                        self._translate_command(command, var, times, counts0, counts0_st, invert=False)

                    # measure reference counts for |0>
                    self._reference_counts(times, counts_ref0, counts_ref0_st, pi_amp=0)

                    # redo above sequennce with a pi-pulse, measuring |1>, if desired
                    if not measure_contrast:
                        for command in self.commands:
                            self._translate_command(command, var, times, counts1, counts1_st, invert=True)

                        self._reference_counts(times, counts_ref1, counts_ref1_st, pi_amp=1)

                    # always end with a wait and saving the number of iterations
                    wait(wait_between_runs)
                save(n, n_st)

            # turn off microwave after experiment concludes
            sg384_NV.rf_off()

            with stream_processing():
                # save the data from the datastream as 1D arrays on the OPx, with a
                # built in running average
                counts0_st.buffer(len(self.var_vec)).average().save("counts0")
                counts_ref0_st.buffer(len(self.var_vec)).average().save("counts_ref0")
                if not measure_contrast:
                    counts1_st.buffer(len(self.var_vec)).average().save("counts1")
                    counts_ref1_st.buffer(len(self.var_vec)).average().save("counts_ref1")
                n_st.save("iteration")

        return experiment

    def simulate_experiment(self, sim_length=10_000, n_avg=100_000, measure_contrast=True, **kwargs):
        """
        Simulates the experiment using the configuration defined by this class.

        Parameters:
            n_avg (int, optional): The number of averages per point. Defaults to 100_000.
            measure_contrast (bool): If True, only the |0> state is measured, if False, both |0> and |1> are measured.
            kwargs (dict): Additional parameters to pass to the simulation

        Raises:
            ValueError: Throws an error if insufficient details about the experiment are defined.
        """
        if len(self.commands) == 0:
            raise ValueError("No commands have been added to the experiment.")
        if self.var_vec is None:
            raise ValueError("No variable vector has been defined.")

        expt = self.create_experiment(n_avg=n_avg, measure_contrast=measure_contrast)
        simulation_config = SimulationConfig(duration=sim_length)  # In clock cycles = 4ns
        job = self.qmm.simulate(config, expt, simulation_config)
        job.get_simulated_samples().con1.plot()
        plt.show()
        return job

    def execute_experiment(self, n_avg=100_000, measure_contrast=True, **kwargs):
        """
        Executes the experiment using the configuration defined by this class. The results are
        stored in the class instance. The results will be visualized live, but this can be
        disabled by setting `live=False` as a keyword arguments. For each value in the variable
        `var_vec`, the experiment will be run `n_avg` times.

        Parameters:
            n_avg (int, optional): The number of averages per point. Defaults to 100_000.
            measure_contrast (bool): If True, only the |0> state is measured, if False, both |0> and |1> are measured.
            kwargs (dict): Additional parameters to pass to the simulation

        Raises:
            ValueError: Throws an error if insufficient details about the experiment are defined.
        """
        if len(self.commands) == 0:
            raise ValueError("No commands have been added to the experiment.")
        if self.var_vec is None:
            raise ValueError("No variable vector has been defined.")

        expt = self.create_experiment(n_avg=n_avg, measure_contrast=measure_contrast)

        # Open the quantum machine
        qm = self.qmm.open_qm(config)

        # Send the QUA program to the OPX, which compiles and executes it
        job = qm.execute(expt)

        # get some optional keyword arguments for advanced features
        live = kwargs.get("live", True)

        # set the mode for fetching the data
        mode = "live" if live else "wait_for_all"

        # set the data lists being generated to later fetch
        data_list = ["counts0", "counts_ref0"]
        if not measure_contrast:
            data_list.extend(["counts1", "counts_ref1"])
        data_list.append("iteration")

        # create the fetch tool
        results = fetching_tool(job, data_list=data_list, mode=mode)

        if live:
            # Live plotting kwargs
            offset_freq = kwargs.get("offset_freq", 0)
            title = kwargs.get("title", "Data Acquisition")
            xlabel = kwargs.get("xlabel", "Dependent Variable")

            fig = plt.figure()
            interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

            while results.is_processing():
                # Fetch results
                if measure_contrast:
                    counts0, counts_ref0, iteration = results.fetch_all()
                else:
                    counts0, counts_ref0, counts1, counts_ref1, iteration = results.fetch_all()
                # Progress bar
                progress_counter(iteration, n_avg, start_time=results.get_start_time())
                # Plot data
                plt.cla()
                plt.plot(
                    (offset_freq + self.var_vec) / u.MHz,
                    counts0 / 1000 / (self.measure_len * 1e-9),
                    label="photon counts |0>",
                )
                plt.plot(
                    (offset_freq + self.var_vec) / u.MHz,
                    counts_ref0 / 1000 / (self.measure_len * 1e-9),
                    label="reference counts |0>",
                )

                if not measure_contrast:
                    plt.plot(
                        (offset_freq + self.var_vec) / u.MHz,
                        counts1 / 1000 / (self.measure_len * 1e-9),
                        label="photon counts |1>",
                    )
                    plt.plot(
                        (offset_freq + self.var_vec) / u.MHz,
                        counts_ref1 / 1000 / (self.measure_len * 1e-9),
                        label="reference counts |1>",
                    )

                plt.xlabel(xlabel)
                plt.ylabel("Intensity [kcps]")
                plt.title(title)
                plt.legend()
                plt.pause(0.1)
        else:
            # Get results from QUA program
            results.wait_for_all_values()
            # Fetch results
            if measure_contrast:
                counts0, counts_ref0, iteration = results.fetch_all()
            else:
                counts0, counts_ref0, counts1, counts_ref1, iteration = results.fetch_all()

        self.counts0 = counts0
        self.counts_ref0 = counts_ref0
        if not measure_contrast:
            self.counts1 = counts1
            self.counts_ref1 = counts_ref1
        self.iteration = iteration

    def save(self, filename=None):
        """
        Saves the experiment configuration to a JSON file.

        Args:
            filename (string): Path to the JSON file to save, defaults to a timestamped filename if
                none is provided
        """
        attributes = {k: v for k, v in self.__dict__.items() if k != "qmm"}
        if filename is None:
            filename = f"expt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        try:
            with open(filename, "w") as f:
                json.dump(attributes, f, cls=NumpyEncoder)
        except (OSError, IOError) as e:
            print(f"Error saving file: {e}")

    def load(self, filename):
        """
        Loads the experiment configuration from a JSON file.

        Args:
            filename (string): Path to the JSON file to load
        """
        try:
            with open(filename, "r") as f:
                attributes = json.load(f)
            for k, v in attributes.items():
                self.__dict__[k] = v
        except (OSError, IOError, FileNotFoundError) as e:
            print(f"Error loading file: {e}")


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
