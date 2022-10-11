from config import Config
import functions as f
import numpy as np
import math

class Channel:

    def __init__(self, ap, ue):

        self.ap = ap
        self.ue = ue
        self.index = np.array([ap.index, ue.index])

        self.norminal_aod = f.get_azimuth(ap.location, ue.location)
        self.angular_spread = Config().angular_spread
        self.multi_paths = Config().multi_paths
        self.rho = Config().rho

        self.d = np.linalg.norm(self.ap.location - self.ue.location)
        self.path_loss = 1 / f.dB2num(120.9 + 37.6 * np.log10(self.d / 1000) + np.random.normal(0, 8))
        self._check_is_link_()
        self._generate_steering_vector_()
        self.g = (np.random.randn(1, self.multi_paths) + np.random.randn(1, self.multi_paths) * 1j) / np.sqrt(2 * self.multi_paths)
        self._cal_csi_(ir_change=True)
        # for saving outdated csi
        self.h1, self.h2 = None, None
        self.r_power10, self.r_power11, self.r_power20, self.r_power21 = None, None, None, None
        self.gain10, self.gain11, self.gain20, self.gain21 = None, None, None, None

        self.utility, self.utility10, self.utility11, self.utility20, self.utility21 = 0, 0, 0, 0, 0
        self.mos, self.mos10, self.mos11, self.mos20, self.mos21 = 0, 0, 0, 0, 0
        self.SINR, self.SINR10, self.SINR11, self.SINR20, self.SINR21 = None, None, None, None, None
        self.IN, self.IN10, self.IN11, self.IN20, self.IN21 = 0, 0, 0, 0, 0
        self.interferer_neighbors, self.interferer_neighbors10, self.interferer_neighbors11, \
        self.interferer_neighbors20, self.interferer_neighbors21 = None, None, None, None, None
        self.interfered_neighbors, self.interfered_neighbors10, self.interfered_neighbors11, \
        self.interfered_neighbors20, self.interfered_neighbors21 = None, None, None, None, None

    def _generate_steering_vector_(self):

        self.aod = self.norminal_aod + (np.random.rand(self.multi_paths) - 0.5) * self.angular_spread
        self.sv = np.zeros((self.multi_paths, self.ap.n_antennas), dtype=complex)
        for i in range(self.multi_paths):
            self.sv[i, :] = np.exp(1j * np.pi * np.cos(self.aod[i]) * np.arange(self.ap.n_antennas)) \
                              / np.sqrt(self.ap.n_antennas)

    def _cal_csi_(self, ir_change):
        if ir_change:
            self.h = np.matmul(self.g, self.sv)
            self.H = self.h.reshape((Config().n_antennas, )) * np.sqrt(self.path_loss)

        self.gain = self.path_loss * np.square(np.linalg.norm(np.matmul(self.h, self.ap.code)))
        self.r_power = self.ap.power * self.gain

    def _check_is_link_(self):
        if self.ap.flag == 'AP':
            if self.ap.association == self.ue.index:
                self.is_link = True
            else:
                self.is_link = False

    def evaluate_mos(self, utility):
        if utility:
            self.mos = Config().lamda * math.log10(Config().tao * utility)
        else:
            self.mos = 0

    def _save_csi_(self, ir_change):

        if ir_change:

            self.h2 = self.h1
            self.h1 = self.h

            self.r_power21 = self.r_power11
            self.r_power11 = self.r_power

            self.gain21 = self.gain11
            self.gain11 = self.gain

            if self.is_link:
                self.IN21 = self.IN11
                self.IN11 = self.IN

                self.SINR21 = self.SINR11
                self.SINR11 = self.SINR

                #self.utility21 = self.utility11
                self.utility11 = self.utility

                self.mos21 = self.mos11
                self.mos11 = self.mos

                self.interferer_neighbors21 = self.interferer_neighbors11
                self.interferer_neighbors11 = self.interferer_neighbors

                self.interfered_neighbors21 = self.interfered_neighbors11
                self.interfered_neighbors11 = self.interfered_neighbors
        else:

            self.r_power20 = self.r_power10
            self.r_power10 = self.r_power

            self.gain20 = self.gain10
            self.gain10 = self.gain

            if self.is_link:
                self.IN20 = self.IN10
                self.IN10 = self.IN

                self.SINR20 = self.SINR10
                self.SINR10 = self.SINR

                #self.utility20 = self.utility10
                self.utility10 = self.utility

                self.mos20 = self.mos10
                self.mos10 = self.mos

                self.interferer_neighbors20 = self.interferer_neighbors10
                self.interferer_neighbors10 = self.interferer_neighbors

                self.interfered_neighbors20 = self.interfered_neighbors10
                self.interfered_neighbors10 = self.interfered_neighbors

    def update(self, ir_change):

        self._save_csi_(ir_change)
        if ir_change:
            # Fading
            e = (np.random.randn(1, self.multi_paths) + np.random.randn(1, self.multi_paths) * 1j) \
                * np.sqrt(1 - np.square(self.rho)) / np.sqrt(2)
            self.g = self.rho * self.g + e

            # self.norminal_aod = f.get_azimuth(self.ap.location, self.ue.location)
            # self.d = np.linalg.norm(self.ap.location - self.ue.location)
            # self.path_loss = 1 / f.dB2num(120.9 + 37.6 * np.log10(self.d / 1000) + np.random.normal(0, 8))
            # self._generate_steering_vector_()

        self._cal_csi_(ir_change)
