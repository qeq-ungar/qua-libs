from qm import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from configuration import *


class NVExperiment:
    def __init__(self, config=None):
        if config is not None:
            self.config = config
        else:
            #from configuration import *
            print("complaints")

        self.setup_experiment()

    def setup_experiment(self):
        # Initialize the QUA library with the provided configuration
        pass

    def add_pulse(self, name, element, length, amplitude):
        self.commands.append({"type": "microwave", "element": element, "name": name, "length": length, "amplitude": amplitude})

    def setup_cw_odmr(self, readout_len, wait_time=1_000, amplitude=1):
        # Code to setup the ODMR experiment
        self.commands = [
            {"type": "align"},
            {"type": "update_frequency", "element": "NV"},
            {"type": "laser", "name": "laser_ON", "duration": readout_len*u.ns},
            {"type": "microwave", "element":"NV","name": "cw", "duration": readout_len*u.ns, "amplitude": amplitude},
            {"type": "wait", "duration": wait_time},
            {"type": "measure", "name": "long_readout", "duration": readout_len},

            {"type": "align"},
        ]
        pass


    def run_experiment(self,n_avg=100_000, use_fixed=False):
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

            if use_fixed:
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
                            measure(command["name"], "SPCM1", None, time_tagging.analog(times, readout_len, counts))
                            save(counts, counts_st)  # save counts on stream
                        elif command["type"] == "align":
                            align()

            with stream_processing():        
                # counts_st.buffer(len(f_vec)).average().save("counts")
                # counts_dark_st.buffer(len(f_vec)).average().save("counts_dark")
                # n_st.save("iteration")



    def process_results(self):
        # Code to process the results of the experiment
        pass

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
