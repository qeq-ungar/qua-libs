from qm import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from configuration import *


class NVExperiment:
    def __init__(self):
        self.qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name, octave=octave_config)
        self.var_vec = None
        self.measure_len = None

    def add_pulse(self, name, element, length, amplitude):
        """
        Adds a type "microwave" command to the experiment, with length in `u.ns` on the
        desired `element`.

        Args:
            name (string): _description_
            element (string): _description_
            length (int): time of pulse in ns
            amplitude (float?): amplitude of pulse
        """
        self.commands.append(
            {"type": "microwave", "element": element, "name": name, "length": length, "amplitude": amplitude}
        )

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

    def add_wait(self, duration):
        """
        Adds a type "wait" command to the experiment.

        Args:
            duration (int): time to wait in ns
        """
        self.commands.append({"type": "wait", "duration": duration})

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

    def add_save(self, dark=False):
        """
        Adds a type "save" command to the experiment.

        Args:
            dark (bool): If True, saves dark counts, otherwise saves bright counts
        """
        self.commands.append({"type": "save", "dark": dark})

    def define_loop(self, var_vec):
        """
        Defines the loop over the variable vector.

        Args:
            var_vec (list): List of values to loop over
        """
        if len(var_vec) == 0:
            raise ValueError("Variable vector cannot be empty.")
        self.use_fixed = any(not isinstance(x, int) for x in var_vec)
        self.var_vec = var_vec

    def setup_cw_odmr(self, readout_len, wait_time=1_000, amplitude=1):
        """
        A pre-fab collection of commands to run a continuous wave ODMR experiment.

        Args:
            readout_len (int): _description_
            wait_time (int, optional): _description_. Defaults to 1_000.
            amplitude (int, optional): _description_. Defaults to 1.
        """
        self.add_align()
        self.add_frequency_update("NV")

        # bright count cw odmr
        self.add_laser("laser_ON", readout_len)
        self.add_pulse("cw", "NV", readout_len, amplitude)
        self.add_wait(wait_time)
        self.add_measure("long_readout", readout_len)
        # save bright counts
        self.add_save()
        self.add_wait(wait_between_runs)

        # dark count cw odmr
        self.add_align()
        self.add_pulse("cw", "NV", readout_len, 0)
        self.add_laser("laser_ON", readout_len)
        self.add_wait(wait_time)
        self.add_measure("long_readout", readout_len)
        self.add_save(dark=True)

    def create_experiment(self, n_avg=100_000):
        """
        Runs the NV experiment with the specified configuration.

        Parameters:
        use_fixed (bool): If True, a fixed variable type is used;
          otherwise, an integer variable type is used. Use fixed-type if
          looping over floats, like in power Rabi.
        """

        # Code to run the NV experiment
        with program() as experiment:
            # generic logic
            counts = declare(int)  # variable for number of counts
            counts_st = declare_stream()  # stream for counts
            counts_dark_st = declare_stream()  # stream for counts
            times = declare(int, size=100)  # QUA vector for storing the time-tags

            if self.use_fixed:
                var = declare(fixed)
            else:
                var = declare(int)

            n = declare(int)  # averaging var
            n_st = declare_stream()  # stream for number of iterations\

            # looping abstraction for freq, time, phase, and amplitude
            with for_(n, 0, n < n_avg, n + 1):
                with for_(*from_array(var, self.var_vec)):
                    # do some logic
                    for command in self.commands:
                        if command["type"] == "update_frequency":
                            update_frequency(command["element"], var)
                        elif command["type"] == "microwave":
                            play(command, command["element"], duration=self.pulses[command]["length"] * u.ns)
                        elif command["type"] == "wait":
                            wait(command["duration"] * u.ns)
                        elif command["type"] == "laser":
                            play(command, "AOM1", duration=self.pulses[command]["length"] * u.ns)
                        elif command["type"] == "measure":
                            measure(
                                command["name"], "SPCM1", None, time_tagging.analog(times, command["meas_len"], counts)
                            )
                        elif command["type"] == "align":
                            align()
                        elif command["type"] == "save":
                            if command["dark"]:
                                save(counts, counts_dark_st)
                            else:
                                save(counts, counts_st)

                    # always end with a wait and saving the number of iterations
                    wait(wait_between_runs * u.ns)
                    save(n, n_st)

            with stream_processing():
                # save the data from the datastream as 1D arrays on the OPx, with a
                # built in running average
                counts_st.buffer(len(self.var_vec)).average().save("counts")
                counts_dark_st.buffer(len(self.var_vec)).average().save("counts_dark")
                n_st.save("iteration")

        return experiment

    def simulate_experiment(self, n_avg=100_000, **kwargs):
        """
        Simulates the experiment using the provided configuration.

        Parameters:
        kwargs (dict): Additional parameters to pass to the simulation
        """
        if len(self.commands) == 0:
            raise ValueError("No commands have been added to the experiment.")
        if self.var_vec is None:
            raise ValueError("No variable vector has been defined.")

        expt = self.create_experiment(n_avg=n_avg)
        simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
        job = self.qmm.simulate(config, expt, simulation_config)
        job.get_simulated_samples().con1.plot()
        plt.show()
        return job

    def execute_experiment(self, n_avg=100_000, **kwargs):
        if len(self.commands) == 0:
            raise ValueError("No commands have been added to the experiment.")
        expt = self.create_experiment(n_avg=n_avg)

        # Open the quantum machine
        qm = self.qmm.open_qm(config)
        # Send the QUA program to the OPX, which compiles and executes it
        job = qm.execute(expt)
        # Get results from QUA program
        mode = kwargs.get("mode", "live")
        results = fetching_tool(job, data_list=["counts", "counts_dark", "iteration"], mode=mode)
        if mode == "live":
            # Live plotting
            fig = plt.figure()
            interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

            while results.is_processing():
                # Fetch results
                counts, counts_dark, iteration = results.fetch_all()
                # Progress bar
                progress_counter(iteration, n_avg, start_time=results.get_start_time())
                # Plot data
                plt.cla()
                plt.plot(
                    (NV_LO_freq * 0 + self.var_vec) / u.MHz, counts / 1000 / (readout_len * 1e-9), label="photon counts"
                )
                plt.plot(
                    (NV_LO_freq * 0 + self.var_vec) / u.MHz,
                    counts_dark / 1000 / (readout_len * 1e-9),
                    label="dark counts",
                )
                plt.xlabel("MW frequency [MHz]")
                plt.ylabel("Intensity [kcps]")
                plt.title("ODMR")
                plt.legend()
                plt.pause(0.1)

    def save_results(self, filename):
        # Code to save the results to a file
        pass

    def load_results(self, filename):
        # Code to load results from a file
        pass


# Example usage
if __name__ == "__main__":
    config = {}  # Define your configuration here
    experiment = NVExperiment(config)
    experiment.run_experiment()
    experiment.process_results()
    experiment.save_results("results.txt")
