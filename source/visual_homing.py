# @File: visual_homing.py
# @Info: to create an agent of VISUAL HOMING based on the insect brain model in insect_brain_model.py
# @Author: Xuelong Sun, UoL, UK
# @Time: 2020-02-17

import numpy as np
from scipy.special import expit

from image_processing import visual_sense
from insect_brain_model import MushroomBodyModel, CentralComplexModel


class VisualHomingAgent(object):
    """Class for the implementation of visual homing model
    """
    def __init__(self, world, route_mem, home_mem, zm_n_max, learning_rate, kc_tau, num_pn, num_kc):

        self.mb = MushroomBodyModel(learning_rate, kc_tau, num_pn=num_pn, num_kc=num_kc)
        self.cx = CentralComplexModel()

        # simulated 3D world, an array with size Nx3
        self.world = world
        # a dictionary with keys: ['imgs', 'h', 'ZM_Ps', 'pos', 'ZM_As']
        self.route_mem = route_mem
        # a dictionary with keys: ['imgs', 'h', 'ZM_Ps', 'pos', 'ZM_As']
        self.home_mem = home_mem
        # frequency encoding parameters
        self.zm_n_max = zm_n_max
        if self.zm_n_max % 2:
            self.zm_coeff_num = int(((1 + zm_n_max) / 2) * ((3 + zm_n_max) / 2))
        else:
            self.zm_coeff_num = int((zm_n_max / 2.0 + 1) ** 2)

    def train_mb_network(self):
        self.mb.reward = True
        en_t = []
        for stimuli in self.route_mem['ZM_As']:
            en_t.append(self.mb.run(stimuli[:self.zm_coeff_num]))
        self.mb.reward = False
        return en_t

    def desired_heading_output(self, mb_delta, vh_k):
        # shifted TB1 to obtain VH desired heading
        shift = np.min([np.max([int(mb_delta * vh_k), 0]), 3])
        vh = np.roll(self.cx.I_tb1, shift)
        vh = expit(vh * 5.0 - 2.5)
        return vh

    def homing(self, start_pos, start_h, time_out, vh_k, motor_k, step_size=4):
        mb_out = np.zeros(time_out)
        mb_delta = np.zeros(time_out)
        pos = np.zeros([time_out, 2])
        velocity = np.zeros([time_out, 2])
        h = np.zeros(time_out)
        pos[0] = start_pos
        h[0] = start_h
        for t in range(time_out-1):
            # frequency coding info
            zm_a, p_temp = visual_sense(self.world, pos[t, 0]/100.0, pos[t, 1]/100.0, h[t], nmax=self.zm_n_max)

            # MB output
            mb_out[t] = self.mb.run(zm_a)

            # change of MBON via SMP
            if t == 0:
                mb_delta[t] = 0
            else:
                mb_delta[t] = mb_out[t] - mb_out[t - 1]

            # update global heading - TB1 neurons
            self.cx.global_current_heading(h[t])

            # generate VH desired heading
            vh = self.desired_heading_output(mb_delta[t], vh_k)

            # steering circuit for motor command
            self.cx.desired_heading_memory = np.hstack([vh, vh])
            self.cx.current_heading_memory = self.cx.I_tb1
            self.cx.steering_circuit_out()

            # moving forward
            h[t+1] = (h[t] + self.cx.motor_value * motor_k + np.pi) % (2.0 * np.pi) - np.pi
            velocity[t + 1, :] = np.array([np.cos(h[t + 1]), np.sin(h[t + 1])]) * step_size
            pos[t + 1, :] = pos[t, :] + velocity[t + 1, :]

        return pos, h, velocity, mb_out, mb_delta






