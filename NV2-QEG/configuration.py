from base_config import ConfigNV, u
from SG384 import SG384Control


from qm import QuantumMachinesManager


class ConfigNV2(ConfigNV):
    def __init__(self, filename=None):
        # load parameters and addresses, use the default configuration if no filename is provided
        if filename is None:
            self.load_global_default()
            self.load_settup_detault()
        else:
            self.load(filename)

        # connect to hardware
        self.qmm = QuantumMachinesManager(host=self.qop_ip, cluster_name=self.cluster_name, octave=self.octave_config)
        self.SG384_NV = SG384Control(self.mw_port1)
        self.SG384_X = SG384Control(self.mw_port2)

        # do not save control classes, and prepare to update the configuration dictionary
        # whenever we make any changes to this object
        self._dns = ["qmm", "SG384_NV", "SG384_X"]
        self.update_config()
        self.__initialized = True

    def load_settup_detault(self):
        """
        Loads additional default parameters for the NV2-QEG experiment.
        """
        self.mw_port1 = "TCPIP0::18.25.11.6::5025::SOCKET"
        self.mw_port2 = "TCPIP0::18.25.11.5::5025::SOCKET"

        self.X_LO_amp = -19
        self.X_LO_freq = 2.83 * u.GHz

    def enable_mw1(self):
        """
        Enables the microwave source for the NV center.
        """
        self.SG384_NV.set_amplitude(self.NV_LO_amp)
        self.SG384_NV.set_frequency(self.NV_LO_freq)
        self.SG384_NV.rf_on()
        self.SG384_NV.do_set_Modulation_State("ON")
        self.SG384_NV.do_set_modulation_type("IQ")

    def disable_mw1(self):
        """
        Disables the microwave source for the NV center.
        """
        self.SG384_NV.rf_off()

    def enable_mw2(self):
        self.SG384_X.set_amplitude(self.X_LO_amp)
        self.SG384_X.set_frequency(self.X_LO_freq)
        self.SG384_X.rf_on()
        self.SG384_X.do_set_Modulation_State("ON")
        self.SG384_X.do_set_modulation_type("IQ")
