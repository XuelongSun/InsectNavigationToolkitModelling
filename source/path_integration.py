# @File: path_integration.py
# @Info: to create an agent of PATH INTEGRATION based on the insect brain model in insect_brain_model.py
# @Author: Xuelong Sun, UoL, UK
# @Time: 2020-02-17

import numpy as np
from insect_brain_model import CentralComplexModel, noisy_sigmoid


class PathIntegrationAgent(object):
    """Class for the implementation of path integration.
       This implementation is adapted from Stone et.al 2017, https://doi.org/10.1016/j.cub.2017.08.052
    """
    def __init__(self, initial_memory):
        self.cx = CentralComplexModel()

        # integrating the CelestialCurrentHeading (TB1) and speed (TN1,TN2)
        self.W_TB1_CPU4 = np.tile(np.eye(8), (2, 1))
        self.W_TN_CPU4 = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        ]).T

        self.tn1 = 0
        self.tn2 = 0

        self.cpu4_mem_gain = 0.005

        self.cpu4_memory = np.ones(16) * initial_memory

    def _update_neuron_activation(self, heading, velocity):
        # update the celestial current heading
        self.cx.global_current_heading(heading)

        # optic flow and the activation of TN1 and TN2 neurons
        flow = get_flow(heading, velocity)
        output = (1.0 - flow) / 2.0
        if self.cx.noise > 0.0:
            output += np.random.normal(scale=self.cx.noise, size=flow.shape)
        self.tn1 = np.clip(output, 0.0, 1.0)
        output = flow
        if self.cx.noise > 0.0:
            output += np.random.normal(scale=self.cx.noise, size=flow.shape)
        self.tn2 = np.clip(output, 0.0, 1.0)

        # CPU4
        cpu4_mem_reshaped = self.cpu4_memory.reshape(2, -1)
        mem_update = (0.5 - self.tn1.reshape(2, 1)) * (1.0 - self.cx.I_tb1)
        mem_update -= 0.5 * (0.5 - self.tn1.reshape(2, 1))
        cpu4_mem_reshaped += self.cpu4_mem_gain * mem_update
        self.cpu4_memory = np.clip(cpu4_mem_reshaped.reshape(-1), 0.0, 1.0)

        # self.cpu4_memory += (np.clip(np.dot(self.W_TN_CPU4, 0.5 - self.tn1), 0, 1) *
        #                      self.cpu4_mem_gain * np.dot(self.W_TB1_CPU4, 1.0 - self.cx.tb1))
        #
        # self.cpu4_memory -= self.cpu4_mem_gain * 0.25 * np.dot(self.W_TN_CPU4, self.tn2)
        # self.cpu4_memory = np.clip(self.cpu4_memory, 0.0, 1.0)
        # self.cpu4_memory = noisy_sigmoid(self.cpu4_memory, 5.0, 2.5, self.cx.noise)

    def generate_pi_memory(self, pi_len, pi_dir, initial_memory):
        """
        generate PI memory (population coding of CPU4)
        :param pi_len: the length of the home vector in meters
        :param pi_dir: the direction of the home vector in degree
        :param initial_memory: initial memory
        :return: CPU4 activation with size 16x1
        """

        # outbound route parameters
        route_length = pi_len * 100.0  # m->cm
        pi_dir = np.deg2rad(pi_dir)
        velocity = 1.0  # cm/s
        dtt = 1  # s

        T_out = int(route_length / velocity / dtt)
        v_out = np.zeros([T_out, 2])
        v_out[:, 0] = np.ones(T_out) * velocity * np.cos(pi_dir)
        v_out[:, 1] = np.ones(T_out) * velocity * np.sin(pi_dir)

        movement_angle = np.arctan2(v_out[:, 1], v_out[:, 0])
        h_out = movement_angle * np.ones(T_out)
        pos_out = np.cumsum(v_out * dtt, axis=0)

        # reset neuron activation
        self.cpu4_memory = np.ones(16) * initial_memory
        self.cx.tb1 = np.zeros(8)

        T = len(pos_out)
        for t in range(T):
            self._update_neuron_activation(h_out[t], v_out[t])

        return self.cpu4_memory

    def homing(self, start_pos, start_h, time_out, motor_k, step_size=4):
        pos = np.zeros([time_out, 2])
        velocity = np.zeros([time_out, 2])
        h = np.zeros(time_out)
        pos[0] = start_pos
        h[0] = start_h
        pi_memory = np.zeros([time_out, 16])

        for t in range(time_out - 1):
            self._update_neuron_activation(h[t], velocity[t])
            pi_memory[t, :] = self.cpu4_memory
            # steering circuit
            self.cx.current_heading_memory = self.cx.I_tb1
            self.cx.desired_heading_memory = self.cpu4_memory
            self.cx.steering_circuit_out()
            # moving forward
            h[t + 1] = (h[t] + self.cx.motor_value * motor_k + np.pi) % (2.0 * np.pi) - np.pi
            velocity[t + 1, :] = np.array([np.cos(h[t + 1]), np.sin(h[t + 1])]) * step_size
            pos[t + 1, :] = pos[t, :] + velocity[t + 1, :]
        return pos, h, velocity, pi_memory


def get_flow(heading, velocity, tn_prefs=np.pi / 4.0, filter_steps=0):
    """Calculate optic flow depending on preference angles. [L, R]"""

    A = np.array([[np.cos(heading + tn_prefs),
                   np.sin(heading + tn_prefs)],
                  [np.cos(heading - tn_prefs),
                   np.sin(heading - tn_prefs)]])
    flow = np.dot(A, velocity)

    return flow

