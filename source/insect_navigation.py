# @File: insect_navigation.py
# @Info: to create an agent of insect navigation based on the insect brain model in insect_brain_model.py
# @Author: Xuelong Sun, UoL, UK
# @Time: 2020-02-17

import numpy as np
from scipy.special import expit
from image_processing import visual_sense
from insect_brain_model import CentralComplexModel, MushroomBodyModel
from insect_brain_model import SuperiorMedialProtocerebrumModel, RingAttractorModel, AOTuVisualPathwayModel


class InsectNavigationAgent(object):
    def __init__(self, world, route_mem, home_mem, zm_n_max,
                 learning_rate, kc_tau, num_pn, num_kc,
                 tun_k, sn_thr,
                 ann_num_neurons,
                 pi_initial_memory):
        self.mb = MushroomBodyModel(learning_rate, kc_tau, num_pn=num_pn, num_kc=num_kc)
        self.cx = CentralComplexModel()
        self.smp = SuperiorMedialProtocerebrumModel(tun_k, sn_thr)
        self.ra = RingAttractorModel()

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

        # ann for route following network
        self.ann = AOTuVisualPathwayModel(x, y, ann_num_neurons)

        # path integration extra connection
        # integrating the CelestialCurrentHeading (TB1) and speed (TN1,TN2)
        self.W_TB1_CPU4 = np.tile(np.eye(8), (2, 1))
        self.W_TN_CPU4 = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        ]).T

        self.tn1 = 0
        self.tn2 = 0

        self.cpu4_mem_gain = 0.0025

        self.cpu4_memory = np.ones(16) * pi_initial_memory

    def train_mb_network(self):
        self.mb.reward = True
        print("$-Start training MB network with lr=%4.4s kc_tau=%4.4s" % (self.mb.learning_rate, self.mb.tau))
        en_t = []
        for stimuli in self.route_mem['ZM_As']:
            en_t.append(self.mb.run(stimuli[:self.zm_coeff_num]))
        self.mb.reward = False
        print("$-Finish training MB network")
        return en_t

    def train_ann_network(self, step=500, learning_rate=1.0, dyna_lr=True):
        print("$-Start training ANN network with lr=%4.4s" % learning_rate)
        for t in range(step):
            self.ann.forward_propagation()
            temp = np.mean(np.abs(self.ann.output - self.ann.y))
            self.ann.error.append(temp)
            if dyna_lr:
                self.ann.learning_rate.append(learning_rate*temp/self.ann.error[0])
            else:
                self.ann.learning_rate.append(learning_rate)

            self.ann.back_propagation(learning_rate=self.ann.learning_rate[t])
        print("$-Finish training ANN network for %s steps" % step)
        return self.ann.error

    def _update_pi_neuron_activation(self, heading, velocity):
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

        return self.cpu4_memory

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
            self._update_pi_neuron_activation(h_out[t], v_out[t])

        return self.cpu4_memory

    def coordination_output(self, mb_delta, vh_k, ann_output):
        # shifted TB1 to obtain VH desired heading
        shift = np.min([np.max([int(mb_delta * vh_k), 0]), 3])
        vh = np.roll(self.cx.I_tb1, shift)
        vh = np.hstack([vh, vh])
        # ann output for RF
        rf = np.hstack([ann_output, ann_output])
        # MB->SMP TN neuron tuned cpu4 memory for PI
        pi = self.cpu4_memory * self.smp.tun
        # RA optimally integrate VH and  PI
        self.ra.cue_integration_output(pi, vh)
        # non-linear integration of RF and optimally integrated PI_VH (output of RA)
        current_heading = self.cx.I_tb1 * self.smp.sn2 + self.cx.II_tb1 * self.smp.sn1
        desired_heading = self.ra.integration_neuron * self.smp.sn2 + rf * self.smp.sn1
        return vh, current_heading, desired_heading

    def homing(self, start_pos, start_h, time_out, vh_k, sn_thr, tn_k, motor_k, step_size=4):
        pos = np.zeros([time_out, 2])
        velocity = np.zeros([time_out, 2])
        h = np.zeros(time_out)
        pos[0] = start_pos
        h[0] = start_h

        dis = 0

        # output of neuron networks
        mb_out = np.zeros(time_out)
        mb_delta = np.zeros(time_out)
        ann_out = np.zeros([time_out, 8])
        pi_memory = np.zeros([time_out, 16])
        vh_memory = np.zeros([time_out, 16])
        ra_memory = np.zeros([time_out, 16])

        # output of SMP neurons
        sn1 = np.zeros(time_out)
        sn2 = np.zeros(time_out)
        tn = np.zeros(time_out)

        # parameter for coordination
        self.smp.sn_thr = sn_thr
        self.smp.tn_k = tn_k

        print("$-Start homing...")
        for t in range(time_out - 1):
            # frequency coding info
            zm_a, p_temp = visual_sense(self.world, pos[t, 0] / 100.0, pos[t, 1] / 100.0, h[t], nmax=self.zm_n_max)

            # update celestial current heading - TB1 neurons and path integration - PI
            self._update_pi_neuron_activation(h[t], velocity[t])
            pi_memory[t, :] = self.cpu4_memory

            # update terrestrial current heading - Frequency phase sensitive neurons in PB
            self.cx.local_current_heading(p_temp[16])

            # MB output - VH
            mb_out[t] = self.mb.run(zm_a)
            # change of MBON via SMP
            if t == 0:
                mb_delta[t] = 0
            else:
                mb_delta[t] = mb_out[t] - mb_out[t - 1]

            # ANN output - RF
            input_temp = zm_a.copy()
            input_temp = (input_temp - np.min(input_temp)) / np.max(input_temp)
            ann_out[t, :] = self.ann.nn_output(input_temp)

            # SMP for guidance coordination
            self.smp.neurons_output(mb_out[t])

            tn[t] = self.smp.tun
            sn1[t] = self.smp.sn1
            sn2[t] = self.smp.sn2

            # get the integrated output
            vh_memory[t, :], current_heading, desired_heading = self.coordination_output(mb_delta[t], vh_k, ann_out[t])

            # store the output of ring attractor
            ra_memory[t, :] = self.ra.integration_neuron

            # steering circuit for motor command
            self.cx.desired_heading_memory = desired_heading
            self.cx.current_heading_memory = current_heading
            self.cx.steering_circuit_out()

            # moving forward
            h[t + 1] = (h[t] + self.cx.motor_value * motor_k + np.pi) % (2.0 * np.pi) - np.pi
            velocity[t + 1, :] = np.array([np.cos(h[t + 1]), np.sin(h[t + 1])]) * step_size
            pos[t + 1, :] = pos[t, :] + velocity[t + 1, :]

            dis = np.sqrt(np.sum(pos[t+1, 0]**2 + pos[t+1, 1]**2))
            if dis < 20:
                break

        print("$-End homing with nest distance %4.4s m" % (dis/100.0))
        return t, pos, h, velocity, mb_out, mb_delta, ann_out, vh_memory, pi_memory, ra_memory, tn, sn1, sn2


def noisy_sigmoid(v, slope=1.0, bias=0.5, noise=0.01):
    """Takes a vector v as input, puts through sigmoid and
    adds Gaussian noise. Results are clipped to return rate
    between 0 and 1"""
    sig = expit(v * slope - bias)
    if noise > 0:
        sig += np.random.normal(scale=noise, size=len(v))
    return np.clip(sig, 0, 1)


def get_flow(heading, velocity, tn_prefs=np.pi / 4.0, filter_steps=0):
    """Calculate optic flow depending on preference angles. [L, R]"""

    A = np.array([[np.cos(heading + tn_prefs),
                   np.sin(heading + tn_prefs)],
                  [np.cos(heading - tn_prefs),
                   np.sin(heading - tn_prefs)]])
    flow = np.dot(A, velocity)

    return flow
