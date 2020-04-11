# @File: insect_brain_model.py
# @Info: the modelling of insect brain, eg, the MushroomBody, the Central Complex etc.
# @Author: Xuelong Sun, UoL, UK
# @Time: 2020-02-17

import numpy as np
import random
from scipy.special import expit


class MushroomBodyModel(object):
    """Class for the MB of visual novelty learning ."""
    def __init__(self, learning_rate, tau, num_pn=81, num_kc=4000, num_en=1):
        # learning para
        self.learning_rate = learning_rate
        self.tau = tau

        # network structure para
        self.num_pn_per_kc = 10
        self.num_stimuli = 80
        self.num_pn = num_pn
        self.num_kc = num_kc
        self.num_en = num_en

        # activation functions
        self.af_pn = lambda x: np.maximum(x, 0)
        self.af_kc = lambda x: np.float32(x > self.tau)
        self.af_en = lambda x: np.maximum(x, 0)

        # initializtion
        self.stimuli = np.zeros(self.num_stimuli)
        self.r_pn = np.zeros(self.num_pn)
        self.r_kc = np.zeros(self.num_kc)
        self.r_en = np.zeros(self.num_en)

        # weights
        self.w_pn2kc = generate_pn2kc_weights(self.num_pn,self.num_kc)
        self.w_kc2en = np.ones([self.num_kc, self.num_en])

        # learning control
        self.reward = False

    def generate_w_pn2kc(self):
        w = np.zeros([self.num_pn, self.num_kc])
        for i in range(self.num_kc):
            random_index = random.sample(range(self.num_pn), self.num_pn_per_kc)
            w[random_index, i] = 1.0
        return w

    def run(self, stimuli):
        self.stimuli = stimuli
        # PN directly receive the stimuli's input
        self.r_pn = self.af_pn(self.stimuli)
        # KC receive randomsly selected 'self.num_pn_per_kc' PNs input
        self.r_kc = self.af_kc(self.r_pn.dot(self.w_pn2kc))
        # EN summed all KCs's activation via the weights
        self.r_en = self.af_en(self.r_kc.dot(self.w_kc2en))

        if self.reward:
            self.learning(self.r_kc)

        return self.r_en

    def learning(self, kc):
        """
            THE LEARNING RULE:
        ----------------------------

          KC  | KC2EN(t)| KC2EN(t+1)
        ______|_________|___________
           1  |    1    |=>   0
           1  |    0    |=>   0
           0  |    1    |=>   1
           0  |    0    |=>   0

        """
        temp = (kc >= self.w_kc2en[:, 0]).astype(bool)
        self.w_kc2en[:, 0][temp] = np.maximum(self.w_kc2en[:, 0][temp] - self.learning_rate, 0)

    def reset(self):
        # initialization
        self.stimuli = np.zeros(self.num_stimuli)
        self.r_pn = np.zeros(self.num_pn)
        self.r_kc = np.zeros(self.num_kc)
        self.r_en = np.zeros(self.num_en)

        # weights
        self.w_pn2kc = self.generate_w_pn2kc()
        self.w_kc2en = np.ones(self.num_kc, self.num_en)

        # learning control
        self.reward = False


class SuperiorMedialProtocerebrumModel(object):
    """Class for the modelling of the SMP neurons (TUN, SN1, SN2)

    """
    def __init__(self, tun_k, sn_thr):
        self.tun = 0.0
        self.tun_k = tun_k
        self.sn1 = 0.0
        self.sn2 = 0.0
        self.sn_thr = sn_thr

    def neurons_output(self, mb_output):
        # linear activation function of TUN
        self.tun = np.min([mb_output * self.tun_k, 1])
        # binary neurons (SN1, SN2)
        self.sn2 = 1.0 if mb_output > self.sn_thr else 0.0
        self.sn1 = 1.0 if self.sn2 == 0.0 else 0.0

        return self.tun, self.sn1, self.sn2


class AOTuVisualPathwayModel(object):
    """
    class for implementation the visual pathway: OL -> AOTU -> BU -> EB
    Route following network generating RF desired heading 
    - artificial neural network with one hidden layer
    """
    def __init__(self, x, y, num_neuron):
        self.num_neurons = num_neuron
        # initialize the network
        self.input = x
        self.y = y

        self.sample_num = x.shape[0]
        self.weight1 = np.random.rand(self.input.shape[1], self.num_neurons)
        self.weight2 = np.random.rand(self.num_neurons, y.shape[1])

        self.layer1_z = np.zeros([self.sample_num, self.num_neurons])
        self.layer1_a = np.zeros([self.sample_num, self.num_neurons])

        self.layer2_z = np.zeros(y.shape)
        self.layer2_a = np.zeros(y.shape)

        self.output = np.zeros(y.shape)

        self.error = []
        self.learning_rate = []

    def forward_propagation(self):
        self.layer1_z = np.dot(self.input, self.weight1)
        self.layer1_a = sigmoid(self.layer1_z)
        self.layer2_z = np.dot(self.layer1_a, self.weight2)
        self.layer2_a = sigmoid(self.layer2_z)
        self.output = self.layer2_a

    def back_propagation(self, learning_rate=1.0):
        delta = 2.0 * (self.y - self.output) * self.layer2_a * (1 - self.layer2_a)
        delta_weight2 = np.dot(self.layer1_a.T, delta)
        delta_weight1 = np.dot(self.input.T, np.dot(delta, self.weight2.T) * self.layer1_a * (1 - self.layer1_a))
        self.weight1 += delta_weight1 * learning_rate
        self.weight2 += delta_weight2 * learning_rate

    def nn_output(self, x):
        layer1 = sigmoid(np.dot(x, self.weight1))
        out = sigmoid(np.dot(layer1, self.weight2))
        # reverse the direction (+pi)
        return np.roll(out, 4)


class CentralComplexModel(object):
    """Class for the CX of current heading (PB), cue integration (FB) and steering circuit (FB).
       This implementation is adapted from Stone et.al 2017, https://doi.org/10.1016/j.cub.2017.08.052
    """
    def __init__(self):
        # global current heading
        # tl2 neurons
        self.tl2_prefs = np.tile(np.linspace(0, 2 * np.pi, 8, endpoint=False), 2)
        self.tl2 = np.zeros(8)
        # cl1 neurons
        self.cl1 = np.zeros(8)
        # I_tb1 neurons
        self.I_tb1 = np.zeros(8)
        # connection weights
        # cl1 -> I_tb1
        self.W_CL1_TB1 = np.tile(np.eye(8), 2)
        # I_tb1 -> I_tb1
        self.W_TB1_TB1 = gen_tb_tb_weights()

        # local current heading
        self.phase_prefs = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        # local compass neuron - II_tb1
        self.II_tb1 = np.zeros(8)

        # steering circuit
        self.desired_heading_memory = np.zeros(16)
        self.current_heading_memory = np.zeros(8)
        # pre-motor neuron CPU1
        self.cpu1a = np.zeros(14)
        self.cpu1b = np.zeros(2)
        self.cpu1 = np.zeros(16)
        # motor[0] for left and motor[1] for right
        self.motor = np.zeros(2)
        # positive -> turn left, negative -> turn right
        self.motor_value = 0
        # connection weights from CPU1 to motor
        self.W_CPU1a_motor = np.array([
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]])
        self.W_CPU1b_motor = np.array([[0, 1],
                                       [1, 0]])

        # connection weights to steering circuit
        # current heading -> steering circuit
        self.W_CH_CPU1a = np.tile(np.eye(8), (2, 1))[1:14 + 1, :]
        self.W_CH_CPU1b = np.array([[0, 0, 0, 0, 0, 0, 0, 1],
                                    [1, 0, 0, 0, 0, 0, 0, 0]])

        # desired heading -> steering circuit
        self.W_DH_CPU1a = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.W_DH_CPU1b = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 8
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 9
        ])

        # other para
        self.noise = 0.0

    def global_current_heading(self, theta):
        # tl2
        output = np.cos(theta - self.tl2_prefs)
        self.tl2 = noisy_sigmoid(output, 6.8, 3.0, self.noise)
        # cl1
        self.cl1 = noisy_sigmoid(-self.tl2, 3.0, -0.5, self.noise)
        # I_tb1
        prop_cl1 = 0.667  # Proportion of input from CL1 vs TB1
        prop_I_tb1 = 1.0 - prop_cl1
        output = (prop_cl1 * np.dot(self.W_CL1_TB1, self.cl1) -
                  prop_I_tb1 * np.dot(self.W_TB1_TB1, self.I_tb1))
        self.I_tb1 = noisy_sigmoid(output, 5.0, 0.0, self.noise)

        return self.I_tb1

    def local_current_heading(self, phase):
        vc_mem_ring = np.cos(np.deg2rad(phase) - self.phase_prefs)
        self.II_tb1 = 1 / (1 + np.exp(-vc_mem_ring * 3 - 1.0))
        return self.II_tb1

    def steering_circuit_out(self):
        inputs = np.dot(self.W_DH_CPU1a, self.desired_heading_memory) * np.dot(self.W_CH_CPU1a,
                                                                               1.0 - self.current_heading_memory)
        self.cpu1a = noisy_sigmoid(inputs, 5.0, 2.5, self.noise)

        inputs = np.dot(self.W_DH_CPU1b, self.desired_heading_memory) * np.dot(self.W_CH_CPU1b,
                                                                               1.0 - self.current_heading_memory)

        self.cpu1b = noisy_sigmoid(inputs, 5.0, 2.5, self.noise)

        self.cpu1 = np.hstack([self.cpu1b[-1], self.cpu1a, self.cpu1b[0]])

        motor = np.dot(self.W_CPU1a_motor, self.cpu1a)
        motor += np.dot(self.W_CPU1b_motor, self.cpu1b)
        self.motor = motor
        self.motor_value = (self.motor[0] - self.motor[1]) * 0.25

        return self.motor_value


class RingAttractorModel(object):
    """
    class for the implementation of ring attractor network for cue integration
    """
    def __init__(self):
        # time to get stable state
        self.time = 0.1 #s
        self.dt = 1e-4
        self.ti = 0.001 #s
        self.nt = np.floor(self.time/self.dt)
        self.ni = np.floor(self.ti/self.dt)

        self.num_neuron = 8
        self.neuron_pref = np.linspace(0, 360 - 360 / self.num_neuron, self.num_neuron).reshape(self.num_neuron, 1)
        # neuron connections
        self.W_E_E_K = 45 / self.num_neuron
        self.W_E_E = self._generate_w_e2e()
        self.W_I_E = 60 / self.num_neuron
        self.W_E_I = -6.0
        self.W_I_I = -1.0

        # other parameters
        self.gammaE = -1.5
        self.gammaI = -7.5
        self.tauE = 0.005
        self.tauI = 0.00025

        # neuron activations
        self.integration_neuron = np.zeros(16)
        self.inhibition_neuron = np.zeros(2)

    def _generate_w_e2e(self):
        wEE = np.zeros((self.num_neuron, self.num_neuron))
        sigma = 130
        for i in range(0, self.num_neuron):
            for j in range(0, self.num_neuron):
                diff = np.min([np.abs(self.neuron_pref[i] - self.neuron_pref[j]),
                               360 - np.abs(self.neuron_pref[i] - self.neuron_pref[j])])
                wEE[i, j] = np.exp((-diff ** 2) / (2 * sigma ** 2))
        return wEE * self.W_E_E_K

    def cue_integration_output(self, cue1, cue2):
        """
        update the ring attractor with cue1 and cue2 injected and return the stable state of the network as the
        integrated output
        :param cue1: the activation of cue1 - an array with size 16x1
        :param cue2: the activation of cue1 - an array with size 16x1
        :return: the optimal integration of cue1 and cue2, an array with the same size as cue1 and cue2
        """
        pi_left = cue1[0:8]

        pi_l = np.zeros((self.num_neuron, int(self.nt)))
        pi_l[:, int(self.ni):] = np.repeat(pi_left.reshape(self.num_neuron, 1), int(self.nt - self.ni), axis=1)

        pi_right = cue1[8:]
        pi_r = np.zeros((self.num_neuron, int(self.nt)))
        pi_r[:, int(self.ni):] = np.repeat(pi_right.reshape(self.num_neuron, 1), int(self.nt - self.ni), axis=1)

        v_left = cue2[0:8]
        v_l = np.zeros((self.num_neuron, int(self.nt)))
        v_l[:, int(self.ni):] = np.repeat(v_left.reshape(self.num_neuron, 1), int(self.nt - self.ni), axis=1)
        v_right = cue2[8:]
        v_r = np.zeros((self.num_neuron, int(self.nt)))
        v_r[:, int(self.ni):] = np.repeat(v_right.reshape(self.num_neuron, 1), int(self.nt - self.ni), axis=1)

        # generate the array for integration cells and the uniform inhibitory cell
        it_l = np.zeros((self.num_neuron, int(self.nt)))
        it_l[:, 0] = 0.1 * np.ones(self.num_neuron, )
        it_r = np.zeros((self.num_neuron, int(self.nt)))
        it_r[:, 0] = 0.1 * np.ones(self.num_neuron, )
        ul = np.zeros((1, int(self.nt)))
        ur = np.zeros((1, int(self.nt)))
        # iteration to the stable state
        for t in range(1, int(self.nt)):
            it_l[:, t] = it_l[:, t - 1] + (-it_l[:, t - 1] + np.max(
                [np.zeros((self.num_neuron,)),
                 self.gammaE + np.dot(self.W_E_E, it_l[:, t - 1]) + self.W_E_I * ul[:, t - 1] + pi_l[:, t - 1]
                 + v_l[:, t - 1]], axis=0)) * self.dt / self.tauE
            ul[:, t] = ul[:, t - 1] + (
                    -ul[:, t - 1] + np.max([0, self.gammaI + self.W_I_E * np.sum(it_l[:, t - 1]) + self.W_I_I * ul[:, t - 1]],
                                           axis=0)) * self.dt / self.tauI

            it_r[:, t] = it_r[:, t - 1] + (-it_r[:, t - 1] + np.max(
                [np.zeros((self.num_neuron,)),
                 self.gammaE + np.dot(self.W_E_E, it_r[:, t - 1]) + self.W_E_I * ur[:, t - 1] + pi_r[:, t - 1]
                 + v_r[:, t - 1]], axis=0)) * self.dt / self.tauE
            ur[:, t] = ur[:, t - 1] + (
                    -ur[:, t - 1] + np.max([0, self.gammaI + self.W_I_E * np.sum(it_r[:, t - 1]) + self.W_I_I * ur[:, t - 1]],
                                           axis=0)) * self.dt / self.tauI

        # get the final iteration as the output
        # update the neuron activation
        self.integration_neuron = np.hstack([it_l[:, -1], it_r[:, -1]])
        self.integration_neuron = noisy_sigmoid(self.integration_neuron, 5.0, 2.5, 0)
        self.inhibition_neuron = np.hstack([ul[-1], ur[-1]])

        return self.integration_neuron


def generate_pn2kc_weights(nb_pn, nb_kc, min_pn=10, max_pn=20, aff_pn2kc=None, nb_trials=100000, baseline=25000,
                           rnd=np.random.RandomState(2018), dtype=np.float32):
    """
    Create the synaptic weights among the Projection Neurons (PNs) and the Kenyon Cells (KCs).
    Choose the first sample that has dispersion below the baseline (early stopping), or the
    one with the lower dispersion (in case non of the samples' dispersion is less than the
    baseline).

    :param nb_pn:       the number of the Projection Neurons (PNs)
    :param nb_kc:       the number of the Kenyon Cells (KCs)
    :param min_pn:
    :param max_pn:
    :param aff_pn2kc:   the number of the PNs connected to every KC (usually 28-34)
                        if the number is less than or equal to zero it creates random values
                        for each KC in range [28, 34]
    :param nb_trials:   the number of trials in order to find a acceptable sample
    :param baseline:    distance between max-min number of projections per PN
    :param rnd:
    :type rnd: np.random.RandomState
    :param dtype:
    """

    dispersion = np.zeros(nb_trials)
    best_pn2kc = None

    for trial in range(nb_trials):
        pn2kc = np.zeros((nb_pn, nb_kc), dtype=dtype)

        if aff_pn2kc is None or aff_pn2kc <= 0:
            vaff_pn2kc = rnd.randint(min_pn, max_pn + 1, size=nb_pn)
        else:
            vaff_pn2kc = np.ones(nb_pn) * aff_pn2kc

        # go through every kenyon cell and select a nb_pn PNs to make them afferent
        for i in range(nb_pn):
            pn_selector = rnd.permutation(nb_kc)
            pn2kc[i, pn_selector[:vaff_pn2kc[i]]] = 1

        # This selections mechanism can be used to restrict the distribution of random connections
        #  compute the sum of the elements in each row giving the number of KCs each PN projects to.
        pn2kc_sum = pn2kc.sum(axis=0)
        dispersion[trial] = pn2kc_sum.max() - pn2kc_sum.min()
        # pn_mean = pn2kc_sum.mean()

        # Check if the number of projections per PN is balanced (min max less than baseline)
        #  if the dispersion is below the baseline accept the sample
        if dispersion[trial] <= baseline: return pn2kc

        # cache the pn2kc with the least dispersion
        if best_pn2kc is None or dispersion[trial] < dispersion[:trial].min():
            best_pn2kc = pn2kc

    # if non of the samples have dispersion lower than the baseline,
    # return the less dispersed one
    return best_pn2kc


def gen_tb_tb_weights(weight=1.):
    """Weight matrix to map inhibitory connections from TB1 to other neurons"""
    W = np.zeros([8, 8])
    sinusoid = -(np.cos(np.linspace(0, 2 * np.pi, 8, endpoint=False)) - 1) / 2
    for i in range(8):
        values = np.roll(sinusoid, i)
        W[i, :] = values
    return weight * W


def noisy_sigmoid(v, slope=1.0, bias=0.5, noise=0.01):
    """Takes a vector v as input, puts through sigmoid and
    adds Gaussian noise. Results are clipped to return rate
    between 0 and 1"""
    sig = expit(v * slope - bias)
    if noise > 0:
        sig += np.random.normal(scale=noise, size=len(v))
    return np.clip(sig, 0, 1)


def sigmoid(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))
