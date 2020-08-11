# Related Publication
[Sun, Xuelong, Yue, Shigang and Mangan, Michael (2020) A decentralised neural model explaining optimal integration of navigational strategies in insects. eLife, 9 . ISSN 2050-084X](https://elifesciences.org/articles/54026)

Any question, be free to contact me: xsun@lincoln.ac.uk, xuelongsun@hotmail.com

# Desciption
The modelling of insect navigation toolkit including:  
* sperated visual navigation-visual homing (VH) and route following (RF)  
* path integration (adapted from Stone et.al 2017)  
* optimal cue integration and the multi-guidance system coordination  
![](img/extended_navigation_toolkit.png 'extended insect navigation toolkit')  
* Modelling the specific brain regions.
![](img/brain_map.png 'Model mapped to brain regions')  

# Folder structure:
* __data__: stores all the materials used in the implementation  
  * _world.mat_ : simulated 3D world, a dictionary with keys 'X','Y','Z' stored the triangle pathes's 3D coordinates  
  * _ArcRouteMem.mat_ : the visual memory of specific route, a dictionary with keys 'pos','h', 'img' etc.
  * _HomeMemory_X0Y0.mat_ ： the visual memory of the nest (at (0,0)). having the same keys with route memory
  * (X)_MeshSampled_ZMs_fixedHeading.mat_ : the frequency coding (zernike moment of the panoramic skyline) of the view across the world with fixed heading 0deg. (Cannot upload, exceed 25M)
  * (X)_MeshSampled_ZMs_randomHeading.mat_ : the frequency coding (zernike moment of the panoramic skyline) of the view across the world with random headings. (Cannot upload, exceed 25M)
* __source__: stores all the source code of the implementation  
  * _image_process.py_ : useful functions for wrapping the image view, get the frequency encoding information, etc.
  * _zernike_moment.py_: the implementation of calculating the Zernike Moment coefficients of fixed size of image
  * _insect_brain_model.py_ : the model of brain regions like Central Complex (CX), Mushroom Body (MB) etc. 
  * _visual_homing.py_ : the class for visual homing  
  * _route following.py_ : the class for route following  
  * _path integration.py_ : the class for path integration  
  * _insect_navigation.py_ : the class for unified model of the navigation toolkit  
  * (X)___InsectNavigationGUI.exe___ : the GUI developed for running the simulations (Cannot upload, exceed 25M). Snapshot of the GUI:  
  
  ![](img/InsectNavigationGUI_ScreenShot.png 'GUI screenshot')

* __demo__: sample code for running the model and plot the figures in paper.

* __img__: images used for README.md  

# Running the simulation:

## 0. Dependance

### Packages and settings


```python
import scipy.io as sio
import numpy as np

%load_ext autoreload
%autoreload 2
```

### Data


```python
# the simulated 3D world
world = sio.loadmat('data/world.mat')
# the route memory
route_memory = sio.loadmat('data/ArcRouteMemory.mat')
# the home memory
home_memory = sio.loadmat('data/HomeMemory_X0Y0.mat')
# the max order of ZM
zm_n_max = 16
```

## 1. Basic simulation

### <font color='orange'>_path integration (PI)_</font>

#### * generate PI memory


```python
from path_integration import PathIntegrationAgent
# set parameters
initial_memory = 0.5
# create an agent
pi = PathIntegrationAgent(initial_memory)
# generate the PI memory
pi_len = 3.0 # m
pi_dir = 45 # deg
pi_memory = pi.generate_pi_memory(pi_len, pi_dir, initial_memory)
```

#### * homing


```python
# homing
start_pos = [-700,-700]
start_h = 0
time_out = 100
motor_k = 0.5
pos, h, velocity, pi_memory = pi.homing(start_pos, start_h, time_out, motor_k, step_size=8)
```

### <font color='red'>_visual homin (VH)_</font>

#### * training the MB network 


```python
from visual_homing import VisualHomingAgent
# set up parameters
num_pn = 81
num_kc = 4000
vh_learning_rate = 0.1
vh_kc_thr = 0.04
# create an instance
vh = VisualHomingAgent(world, route_memory, home_memory, zm_n_max, vh_learning_rate, vh_kc_thr, num_pn, num_kc)
# training
en = vh.train_mb_network()
```

#### * homing


```python
start_pos = [0,-700]
start_h = 0
time_out = 100
vh_k = 0.5
motor_k = 1.5 * 0.25
pos, h, velocity, mb_out, mb_delta = vh.homing(start_pos, start_h, time_out, vh_k, motor_k, step_size=8)
```

### <font color='blue'>_route following (RF)_</font>

#### * training the ANN


```python
from route_following import RouteFollowingAgent
# set parameters
rf_learning_rate = 0.1
rf_learning_step = 30000
# create an instance
rf = RouteFollowingAgent(world, route_memory, home_memory, zm_n_max, num_neurons=30)
# train the ANN
rf.train_nn_network(rf_learning_step, rf_learning_rate)
```

#### * homing


```python
start_pos = [-700,-700]
start_h = 0
time_out = 100
motor_k = 1.5
pos, h, velocity, nn_output = rf.homing(start_pos, start_h, time_out, motor_k, step_size=8)
```

### ___whole model___

#### * create the agent of insect navigation and train the networks


```python
from insect_navigation import InsectNavigationAgent

# set PI parameters
pi_initial_memory = 0.1
pi_len = 3.0 # m
pi_dir = 90 # deg

# set VH parameters
num_pn = 81
num_kc = 4000
vh_learning_rate = 0.1
vh_kc_thr = 0.04

# set RF parameters
rf_learning_rate = 0.1
rf_learning_step = 30000
ann_num_neurons = 30

# set SMP neuron parameters
tun_k = 0.0125
sn_thr = 5.0

# create the insect navigation agent
agent = InsectNavigationAgent(world, route_memory, home_memory, zm_n_max, 
                              vh_learning_rate, vh_kc_thr, num_pn, num_kc, 
                              tun_k, sn_thr,
                              ann_num_neurons,
                              pi_initial_memory)
# training the MB network
en = agent.train_mb_network()
# training the ANN network
err = agent.train_ann_network(rf_learning_step, rf_learning_rate)
# generate PI memory
pi = agent.generate_pi_memory(pi_len, pi_dir, pi_initial_memory)
```

#### * homing


```python
start_pos = [-200,-700]
start_h = 90
time_out = 10
motor_k = 1.5

# set PI parameters
pi_initial_memory = 0.1
pi_len = 3.0 # m
pi_dir = 225 # deg
# generate PI memory
pi = agent.generate_pi_memory(pi_len, pi_dir, pi_initial_memory)

# VH tuning scalar
vh_k = 0.5

# start homing
end_t, pos, h, velocity, mb_out, mb_delta, ann_out, vh_memory, pi_memory, ra_memory, tn, sn1, sn2 = agent.homing(start_pos, start_h, time_out,
                                                                                                               vh_k, sn_thr, tun_k, motor_k,
                                                                                                               step_size=8)
```

## 2. Reproducing behavioural data

Using the seperated model for VH and RF

### visual_navigation on and off route (Wystrach2012)

### <font color='red'>_visual homin (VH)_</font>


```python
from visual_homing import VisualHomingAgent
# set up parameters
num_pn = 81
num_kc = 4000
vh_learning_rate = 0.1
vh_kc_thr = 0.04
# create an instance
vh = VisualHomingAgent(world, route_memory, home_memory, zm_n_max, vh_learning_rate, vh_kc_thr, num_pn, num_kc)
# training MB
en = vh.train_mb_network()

# trials setting
start_pos = [0,-700]
start_h_s = np.linspace(0, 2 * np.pi, 12, endpoint=False)
time_out = 100
vh_k = 2.0
motor_k = 0.125

pos_s = []
h_s = []

# run the trial
for start_h in start_h_s:
    pos, h, velocity, mb_out, mb_delta = vh.homing(start_pos, start_h, time_out, vh_k, motor_k, step_size=4)
    # store the homing data
    pos_s.append(pos)
    h_s.append(h)
```

### <font color='blue'>_route following (RF)_</font>


```python
from route_following import RouteFollowingAgent
# set parameters
rf_learning_rate = 0.1
rf_learning_step = 30000
# create an instance
rf = RouteFollowingAgent(world, route_memory, home_memory, zm_n_max, num_neurons=30)
# train the ANN
rf.train_nn_network(rf_learning_step, rf_learning_rate)

start_pos = [-700,-700]
start_h_s = np.linspace(0, 2 * np.pi, 2, endpoint=False)
time_out = 5
motor_k = 0.125

pos_s = []
h_s = []

# run the trial
for start_h in start_h_s:
    pos, h, velocity, nn_output = rf.homing(start_pos, start_h, time_out, motor_k, step_size=4)
    # store the homing data
    pos_s.append(pos)
    h_s.append(h)
```

### <font color='gray'>__optimal cue integration__</font>

Using the whole model, but turn-off route following by setting ___sn_thr = 0.0___

#### _tuning PI uncertainty (Wystrach2015)_


```python
from insect_navigation import InsectNavigationAgent

# set PI parameters
pi_initial_memory = 0.1
pi_len = 3.0 # m
pi_dir = 90 # deg

# set VH parameters
num_pn = 81
num_kc = 4000
vh_learning_rate = 0.1
vh_kc_thr = 0.04

# set RF parameters
rf_learning_rate = 0.1
rf_learning_step = 30000
ann_num_neurons = 30

# set SMP neuron parameters
tun_k = 0.1
sn_thr = 0.0

# create the insect navigation agent
agent = InsectNavigationAgent(world, route_memory, home_memory, zm_n_max, 
                              vh_learning_rate, vh_kc_thr, num_pn, num_kc, 
                              tun_k, sn_thr,
                              ann_num_neurons,
                              pi_initial_memory)
# training the MB network
en = agent.train_mb_network()

# trials setting
# generate PI memory
pi_len_s = [0.1, 1.0, 3.0, 7.0] # m
pi_dir = 90 # deg

start_pos = [137.35, -50.]
start_h_s = np.linspace(0, 2 * np.pi, 2, endpoint=False)
time_out = 2
motor_k = 0.125

# VH tuning scalar
vh_k = 2.0

pos_s = []
h_s = []

# run the trials 
for start_h in start_h_s:
    for pi_len in pi_len_s:
        # generate different PI home vector length
        pi = agent.generate_pi_memory(pi_len, pi_dir, pi_initial_memory)
        # start homing
        end_t, pos, h, velocity, mb_out, mb_delta, ann_out, vh_memory, pi_memory, ra_memory, tn, sn1, sn2 = agent.homing(start_pos, start_h, time_out,
                                                                                                                   vh_k, sn_thr, tun_k, motor_k,
                                                                                                                   step_size=8)
        pos_s.append(pos)
        h_s.append(h)
```

#### _tuning VH uncertainty (Legge2014)_


```python
from insect_navigation import InsectNavigationAgent

# set PI parameters
pi_initial_memory = 0.1
pi_len = 3.0 # m
pi_dir = 90 # deg

# set VH parameters
num_pn = 81
num_kc = 4000
vh_learning_rate = 0.1
vh_kc_thr = 0.04

# set RF parameters
rf_learning_rate = 0.1
rf_learning_step = 30000
ann_num_neurons = 30

# set SMP neuron parameters
tun_k = 0.1
sn_thr = 0.0

# create the insect navigation agent
agent = InsectNavigationAgent(world, route_memory, home_memory, zm_n_max, 
                              vh_learning_rate, vh_kc_thr, num_pn, num_kc, 
                              tun_k, sn_thr,
                              ann_num_neurons,
                              pi_initial_memory)
# training the MB network
en = agent.train_mb_network()

# generate PI memory
pi = agent.generate_pi_memory(pi_len, pi_dir, pi_initial_memory)
        
# trials setting
# generate PI memory
pi_len_s = [0.1, 1.0, 3.0, 7.0] # m
pi_dir = 90 # deg

start_pos_s = [[137.35, -50.], [412.05, -150.], [686.75, -250.]]
start_h_s = np.linspace(0, 2 * np.pi, 2, endpoint=False)
time_out = 2
motor_k = 0.125

# VH tuning scalar
vh_k = 2.0

pos_s = []
h_s = []

# run the trials 
for start_h in start_h_s:
    for start_pos in start_pos_s:
        # start homing
        end_t, pos, h, velocity, mb_out, mb_delta, ann_out, vh_memory, pi_memory, ra_memory, tn, sn1, sn2 = agent.homing(start_pos, start_h, time_out,
                                                                                                                   vh_k, sn_thr, tun_k, motor_k,
                                                                                                                   step_size=8)
        pos_s.append(pos)
        h_s.append(h)
```

#### ___whole model for all the properties we want___



This can be done by using the ___whole model___ cell in the __Basic simulation__ section.

## 3. Other analysis tools

This section contains some code to generate some analysis data, such as the ZM encoding of the 3D world.

#### _Check the frequency phase tracking across the world_


```python
%%time
from insect_navigation import InsectNavigationAgent
from image_processing import visual_sense
# check the data for RF
# 1.RF memory , 2.the phase-tracking, 3.RF suggested

# set PI parameters
pi_initial_memory = 0.1
pi_len = 3.0 # m
pi_dir = 90 # deg

# set VH parameters
num_pn = 81
num_kc = 4000
vh_learning_rate = 0.1
vh_kc_thr = 0.04

# set RF parameters
rf_learning_rate = 0.1
rf_learning_step = 30000
ann_num_neurons = 30

# set SMP neuron parameters
tun_k = 0.0125
sn_thr = 5.0

# create the insect navigation agent
InsectNaviAgent = InsectNavigationAgent(world, route_memory, home_memory, zm_n_max, 
                                        vh_learning_rate, vh_kc_thr, num_pn, num_kc, 
                                        tun_k, sn_thr,
                                        ann_num_neurons,
                                        pi_initial_memory)
# training the MB network
en = agent.train_mb_network()
# training the ANN network
err = agent.train_ann_network(rf_learning_step, rf_learning_rate)

# sampled num
sample_num = 2

# sampled locations
pos_x = np.linspace(-10,2,sample_num)
pos_y = np.linspace(-8,2,sample_num)
# sampled heading
h = np.linspace(-np.pi,np.pi,2)

# stored data
ann_output = np.zeros([sample_num**2,len(h)])
current_zm_p = np.zeros([sample_num**2,len(h)])

vc_phase_prefs = np.linspace(-np.pi,np.pi,8,endpoint=False)

for i in range(sample_num**2):
    for k,h_i in enumerate(h):
        A,P = visual_sense(InsectNaviAgent.world, pos_x[i%sample_num],pos_y[i//sample_num],h_i,nmax=InsectNaviAgent.zm_n_max)
        current_zm_p[i,k] = P[16]
        nn_input = A.copy()
        nn_input = (nn_input - np.min(nn_input))/np.max(nn_input)
        nn_res = InsectNaviAgent.ann.nn_output(nn_input)
        ann_output[i,k] = np.arctan2(np.sum(nn_res*np.sin(vc_phase_prefs)), 
                                     np.sum(nn_res*np.cos(vc_phase_prefs)))          
        
# sio.savemat('QuiverPlotData_X-10_2_Y-8_2_SH20.mat',{'ann_output':ann_output, 'current_zm_p':current_zm_p})
```

#### _generate frequency coding memory of specific locations and heading defined by __pos and h___


```python
# generate visual memory along PI route
from image_processing import get_img_view
from zernike_moment import zernike_moment

home_pos_new = np.array([0, 0])
sample_len = 20
pos = np.array([[home_pos_new[0] - sample_len, home_pos_new[1]],
                [home_pos_new[0] - sample_len, home_pos_new[1] - sample_len],
                [home_pos_new[0], home_pos_new[1] - sample_len],
                [home_pos_new[0] + sample_len, home_pos_new[1] - sample_len],
                [home_pos_new[0] + sample_len, home_pos_new[1]],
                [home_pos_new[0] + sample_len, home_pos_new[1] + sample_len],
                [home_pos_new[0], home_pos_new[1] + sample_len],
                [home_pos_new[0] - sample_len, home_pos_new[1] + sample_len],
                [home_pos_new[0], home_pos_new[1]],
                ])
h = np.array([0, 0.25, 0.5, 0.75, 1.0, -0.75, -0.5, -0.25, 0.5]) * np.pi
        
memory_ZM_As = np.zeros([len(pos), 81])
memory_ZM_Ps = np.zeros([len(pos), 81])
memory_imgs =  np.zeros([len(pos), 208, 208])
for i in range(len(pos)):
    memory_imgs[i, :, :] = get_img_view(InsectNaviAgent.world_data, pos[i, 0] / 100.0, pos[i, 1] / 100.0, 0.01,
                                        h[i], hfov_d=360, wrap=True,
                                        blur=False, blur_kernel_size=3)
    index = 0
    for n in range(InsectNaviAgent.nmax + 1):
        for m in range(n + 1):
            if (n - m) % 2 == 0:
                M, memory_ZM_As[i, index], memory_ZM_Ps[i, index] = zernike_moment(255 - memory_imgs[i, :, :], n, m)
                index += 1
```

#### _compute the frequecny encoding of locations across the world_


```python
%%time
from image_processing import visual_sense
# generate the ZM coefficients for the world
sample_num = 20
map_x = np.linspace(-10,10,sample_num)
map_y = np.linspace(-10,10,sample_num)
h = np.zeros([len(map_x),len(map_y)])
world_zm_a = np.zeros([len(map_x),len(map_y),81])
world_zm_p = np.zeros([len(map_x),len(map_y),81])
for i,y in enumerate(map_y):
    for j,x in enumerate(map_x):
        A,P = visual_sense(InsectNaviAgent.world_data,x,y,h[i,j],nmax=16)
        world_zm_a[j,i] = A
        world_zm_p[j,i] = P
```
