from datetime import datetime
import json
import numpy as np

from qm import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from configuration_NV2QEG import *


class NVExperiment:
    def __init__(self):
        self.qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name, octave=octave_config)

        # containers for commands
        self.var_vec = None
        self.commands = []
        self.use_fixed = False
        self.measure_len = None
        self.initialize = False

        # containers for results
        self.counts0 = None
        self.counts_ref0 = None
        self.counts1 = None
        self.counts_ref1 = None
        self.iteration = None

    def add_pulse(self, name, element, amplitude, variable=False, cycle=False):
        """
        Adds a type "microwave" command to the experiment, with length in `u.ns` on the
        desired `element`.

        Args:
            name (string): Name of the pulse. 8 predefined pulses are avaialble,
                "+/-" * "x/y" * "90/180", eg "y180" or "-x90"
            element (string): Channel to play the pulse on, like "NV" or "C13" in the config
            amplitude (float?): amplitude of pulse
            variable (bool): If True, the pulse amplitude is a variable defined by the loop.
        """
        self.commands.append(
            {
                "type": "pulse",
                "element": element,
                "name": name,
                "amplitude": amplitude,
                "variable": variable,
                "cycle": cycle,
            }
        )
        if variable:
            self.use_fixed = True

    def add_cw_drive(self, element, length, amplitude):
        """
        Adds a type "microwave" command to the experiment, with length in `u.ns` on the
        desired `element`.

        Args:
            element (string): Channel to play the pulse on, like "NV" or "C13" in the config
            length (int): time of pulse in ns
            amplitude (float): amplitude of pulse
        """
        self.commands.append({"type": "cw", "element": element, "length": length, "amplitude": amplitude})

    def add_laser(self, name, length):
        """
        Adds a type "laser" command to the experiment, with length in `u.ns`.
        test1 help

        Args:
            name (string): command name
            length (int): time of laser illumination in ns
        """
        self.commands.append({"type": "laser", "name": name, "length": length})

    def add_align(self):
        """
        Adds a type "align" command to the experiment.
        """
        self.commands.append({"type": "align"})

    def add_wait(self, duration, variable=False):
        """
        Adds a type "wait" command to the experiment.

        Args:
            duration (int): time to wait in ns
            variable (bool): If True, the wait time is a variable defined by the loop.
        """
        self.commands.append({"type": "wait", "duration": duration, "variable": variable})

    def add_measure(self, name="SPCM", meas_len=1000):
        """
        Adds a type "measure" command to the experiment.

        Args:
            name (string): Name of the photon counter
            duration (int): time of measurement acquisition in ?ns?
        """
        self.commands.append({"type": "measure", "name": name, "meas_len": meas_len})
        if self.measure_len is None:
            self.measure_len = meas_len
        elif self.measure_len != meas_len:
            raise ValueError("Inconsistent measurement lengths.")

    def add_frequency_update(self, element):
        """
        Adds a type "update_frequency" command to the experiment.

        Args:
            element (string): Name of the element to update the frequency of
        """
        self.commands.append({"type": "update_frequency", "element": element})

    def add_initialization(self):
        """
        Adds a laser pulse to polarize the system before the first sequence. This is controlled with the config file.
        """
        self.initialize = True

    def define_loop(self, var_vec):
        """
        Defines the loop over the variable vector.

        Args:
            var_vec (list): List of values to loop over
        """
        if len(var_vec) == 0:
            raise ValueError("Variable vector cannot be empty.")
        self.var_vec = var_vec

    def setup_cw_odmr(self, readout_len=long_meas_len_1, wait_time=1_000, amplitude=1):
        """
        A pre-fab collection of commands to run a continuous wave ODMR experiment.

        Args:
            readout_len (int): _description_
            wait_time (int, optional): _description_. Defaults to 1_000.
            amplitude (int, optional): _description_. Defaults to 1.
        """
        self.add_align()
        self.add_frequency_update("NV")

        self.add_laser("laser_ON", readout_len)
        self.add_cw_drive("NV", readout_len, amplitude)

        self.add_wait(wait_time)
        self.add_measure("long_readout", readout_len)

    def _translate_command(self, command, var, times, counts, counts_st, invert):
        """
        Helper function whcih translates a command dictionary into a QUA command. Plays qua commands, can only
        be called from within a qua program.

        Args:
            command (dict): Command dictionary

        Returns:
            qua command: The QUA command
        """
        match command["type"]:
            case "update_frequency":
                update_frequency(command["element"], var)

            case "pulse":
                amplitude = var if command["variable"] else command["amplitude"]
                name = command["name"]
                if invert and command["cycle"]:
                    if name[0] == "-":
                        name = name[1:]
                    else:
                        name = "-" + name
                play(name * amp(amplitude), command["element"])

            case "cw":
                duration = var if command["variable"] else command["length"]
                play("cw" * amp(command["amplitude"]), command["element"], duration=duration * u.ns)

            case "wait":
                duration = var if command["variable"] else command["length"]
                wait(duration * u.ns)

            case "laser":
                play("laser_ON", "AOM1", duration=command["length"] * u.ns)

            case "measure":
                measure(
                    command["name"],
                    "SPCM1",
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
        wait(wait_between_runs * u.ns)
        align()

        play("x180" * amp(pi_amp), "NV")  # Pi-pulse toggle
        align()

        play("laser_ON", "AOM1")
        measure("readout", "SPCM1", None, time_tagging.analog(times, self.measure_len, counts))

        save(counts, counts_st)  # save counts
        wait(wait_between_runs * u.ns, "AOM1")

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
                play("laser_ON", "AOM1")
                wait(wait_for_initialization * u.ns, "AOM1")

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
                    wait(wait_between_runs * u.ns)
                save(n, n_st)

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

    def simulate_experiment(self, n_avg=100_000, measure_contrast=True, **kwargs):
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
        simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
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
