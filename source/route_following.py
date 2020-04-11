# @File: route_following.py
# @Info: to create an agent of ROUTE FOLLOWING based on the insect brain model in insect_brain_model.py
# @Author: Xuelong Sun, UoL, UK
# @Time: 2020-02-17

import numpy as np
from insect_brain_model import CentralComplexModel, AOTuVisualPathwayModel
from image_processing import visual_sense


class RouteFollowingAgent(object):
    """Class for the implementation of route following model
    """
    def __init__(self, world, route_mem, home_mem, zm_n_max, num_neurons=30):
        # central complex
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

        # re arrange the memory
        mem_scene = self.route_mem['ZM_As'][:, :self.zm_coeff_num].copy()
        mem_phase = self.route_mem['ZM_Ps'][:, 16].copy()

        mem_phase_ring = np.zeros([len(mem_phase), 8])
        for i in range(len(mem_phase)):
            mem_scene[i, :] = (mem_scene[i, :] - np.min(mem_scene[i, :])) / np.max(mem_scene[i, :])
            mem_phase_ring[i, :] = np.cos(np.deg2rad(mem_phase[i]) - self.cx.phase_prefs)
        mem_phase_ring_sig = 1 / (1 + np.exp(-mem_phase_ring * 3 - 1.0))

        x = mem_scene
        y = mem_phase_ring_sig

        self.ann = AOTuVisualPathwayModel(x, y, num_neurons)

    def train_nn_network(self, step=500, learning_rate=1.0, dyna_lr=True):
        for t in range(step):
            self.ann.forward_propagation()
            temp = np.mean(np.abs(self.ann.output - self.ann.y))
            self.ann.error.append(temp)
            if dyna_lr:
                self.ann.learning_rate.append(learning_rate*temp/self.ann.error[0])
            else:
                self.ann.learning_rate.append(learning_rate)

            self.ann.back_propagation(learning_rate=self.ann.learning_rate[t])

    def homing(self, start_pos, start_h, time_out, motor_k, step_size=4):
        nn_out = np.zeros([time_out, 8])
        pos = np.zeros([time_out, 2])
        velocity = np.zeros([time_out, 2])
        h = np.zeros(time_out)
        pos[0] = start_pos
        h[0] = start_h
        for t in range(time_out - 1):
            # frequency coding info
            zm_a, p_temp = visual_sense(self.world, pos[t, 0] / 100.0, pos[t, 1] / 100.0, h[t], nmax=self.zm_n_max)

            # update local current heading
            self.cx.local_current_heading(p_temp[16])

            # neural network output and desired heading of RF
            input_temp = zm_a.copy()
            input_temp = (input_temp - np.min(input_temp)) / np.max(input_temp)
            nn_out[t, :] = self.ann.nn_output(input_temp)

            # steering circuit for motor command
            self.cx.desired_heading_memory = np.hstack([nn_out[t, :], nn_out[t, :]])
            self.cx.current_heading_memory = self.cx.II_tb1
            self.cx.steering_circuit_out()

            # moving forward
            h[t + 1] = (h[t] + self.cx.motor_value * motor_k + np.pi) % (2.0 * np.pi) - np.pi
            velocity[t + 1, :] = np.array([np.cos(h[t + 1]), np.sin(h[t + 1])]) * step_size
            pos[t + 1, :] = pos[t, :] + velocity[t + 1, :]

        return pos, h, velocity, nn_out

