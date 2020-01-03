"""
    # mobile environment that update channel and association
    """
import numpy as np
import random
import itertools
import math
from channel import *
from ue_mobility import *
import copy
import numpy as np
import sys
from collections import namedtuple

matplotlib.rcParams.update({'font.size': 14})

# defining the number of steps
MAXSTEP = 2000
UE_STEP = 1

N_ACT = 5   #number of actions of a single agent

MAX_UE_PER_GRID = 1 # Maximum number of UE per grid

# relative hight of the BSs to UEs in m assuming plain terrain
H_BS = 10
# min distance between BSs in Grids
MIN_BS_DIST = 2

R_BS = 50
#
BS_STEP = 2


class MobiEnvironment:
    
    def __init__(self, nBS, nUE, grid_n=200, mobility_model="group", test_mobi_file_name="", random_seed=None):
        self.nBS = nBS
        self.nUE = nUE
        self.bs_h = H_BS
        
        # x,y boundaries
        self.grid_n = grid_n

        if random_seed is not None:
            np.random.seed(seed=random_seed)

        [xMin, xMax, yMin, yMax] = [1, self.grid_n, 1, self.grid_n]
        boundaries = [xMin, xMax, yMin, yMax]
        base_station_x = []
        base_station_y = []
        for idx in range(self.nBS):
            # bs_x = int((idx * xMax) / self.nBS)
            # bs_y = np.random.randint(1, yMax)
            # base_station_x.append(bs_x)
            # base_station_y.append(bs_y)
            base_station_x.append(np.random.randint(1, xMax))
            base_station_y.append(np.random.randint(1, yMax))
            # xBS = np.array([int(xMax/4), int(xMax/4), int(xMax*3/4), int(xMax*3/4)])
            # yBS = np.array([int(yMax/4), int(yMax*3/4), int(yMax/4), int(yMax*3/4)])


        self.boundaries = boundaries
        self.mobility_model = mobility_model
        print("mobility model: ", mobility_model, " grid size ", grid_n)
        #       bsLoc is 3D, bsLocGrid is 2D heatmap
        #self.bsLoc, self.bsLocGrid = GetRandomLocationInGrid(self.grid_n, self.grid_n, self.nBS, H_BS, MIN_BS_DIST)
        self.initBsLoc = np.array([
            base_station_x,
            base_station_y,
            np.ones((np.size(base_station_x))) * self.bs_h
        ], dtype=int).T

        self.initBsLocGrid = GetGridMap(self.grid_n, self.grid_n, self.initBsLoc[:,:2])
        self.bsLoc = copy.deepcopy(self.initBsLoc)
        self.bsLocGrid = copy.deepcopy(self.initBsLocGrid)

#        self.ueLoc, self.ueLocGrid = GetRandomLocationInGrid(self.grid_n, self.grid_n, self.nUE)
        self.ueLoc = []
        self.ueLocGrid = []
        
        self.mm = []
        
        #mobility trace used for testing
        self.test_mobi_trace = []

        if self.mobility_model == "random_waypoint":
            self.mm = random_waypoint(nUE, dimensions=(self.grid_n, self.grid_n), velocity=(1, 1), wt_max=1.0)
        elif self.mobility_model == "group":
            # self.mm = reference_point_group([10,10,10,10], dimensions=(self.grid_n, self.grid_n), velocity=(0, 1), aggregation=0.8)
            self.mm = reference_point_group([self.nUE], dimensions=(self.grid_n, self.grid_n), velocity=(0, 1), aggregation=0.8)
            # for i in range(200):
            #     next(self.mm)
            #     i += 1 #repeat in reset
        elif self.mobility_model == "in_coverage":
            self.ueLoc, self.ueLocGrid = GetRandomLocationInCellCoverage(self.grid_n, self.grid_n, R_BS,  self.bsLoc, self.nUE)
        elif self.mobility_model == "read_trace":
            print("testing with mobility trace ", test_mobi_file_name)
            assert test_mobi_file_name
            self.ueLoc_trace = np.load(test_mobi_file_name)
            
            self.ueLoc = self.ueLoc_trace[0]
            self.ueLocGrid = GetGridMap(self.grid_n, self.grid_n, self.ueLoc)
        
        else:
            sys.exit("mobility model not defined")

        if (self.mobility_model == "random_waypoint") or (self.mobility_model == "group"):
            positions = next(self.mm)
            #2D to 3D
            user_height = np.zeros((np.shape(positions)[0], 1))
            self.ueLoc = np.append(positions, user_height, axis=1).astype(int)
            self.ueLocGrid = GetGridMap(self.grid_n, self.grid_n, self.ueLoc)

        self.channel = LTEChannel(self.nUE, self.nBS, self.boundaries, self.ueLoc, self.bsLoc)
        # self.association = self.channel.GetCurrentAssociationMap(self.ueLoc)

        
        self.action_space_dim = N_ACT**self.nBS
        self.observation_space_dim = 3 * (self.nBS + self.nUE)

        self.state = np.zeros((self.nBS + self.nUE, 3))
        self.step_n = 0
    

    def SetBsH(self, h):
        self.bs_h = h
    
    
    def reset(self):
        self.bsLoc = copy.deepcopy(self.initBsLoc)
        self.bsLocGrid = copy.deepcopy(self.initBsLocGrid)
        
        if (self.mobility_model == "random_waypoint") or (self.mobility_model == "group"):
            pass
            # positions = next(self.mm)
            # #2D to 3D
            # z = np.zeros((np.shape(positions)[0], 0))
            # self.ueLoc = np.concatenate((positions, z), axis=1).astype(int)
            # self.ueLocGrid = GetGridMap(self.grid_n, self.grid_n, self.ueLoc)
        elif self.mobility_model == "read_trace":
            print("reseting mobility trace ")
            self.ueLoc = self.ueLoc_trace[0]
            self.ueLocGrid = GetGridMap(self.grid_n, self.grid_n, self.ueLoc)
        else:
            self.ueLoc, self.ueLocGrid = GetRandomLocationInCellCoverage(self.grid_n, self.grid_n, R_BS,  self.bsLoc, self.nUE)

        self.channel.reset(self.ueLoc, self.bsLoc)
        # self.association = self.channel.GetCurrentAssociationMap(self.ueLoc)
        self.state = np.concatenate((self.bsLoc, self.ueLoc))
        self.step_n = 0
        
        return np.array(self.state)
    
    def step(self, action, ifrender=False): #(step)
        
        # positions = next(self.mm)
        # #2D to 3D
        # z = np.zeros((np.shape(positions)[0],0))
        # self.ueLoc = np.concatenate((positions, z), axis=1).astype(int)

        self.bsLoc, bsActions = BS_move(self.bsLoc, self.boundaries, action, BS_STEP, MIN_BS_DIST + BS_STEP, N_ACT)
        self.association_map, meanSINR, nOut = self.channel.UpdateDroneNet(self.ueLoc, self.bsLoc, ifrender, self.step_n)
        
        self.bsLocGrid = GetGridMap(self.grid_n, self.grid_n, self.bsLoc)
        self.ueLocGrid = GetGridMap(self.grid_n, self.grid_n, self.ueLoc)
        
        # r_dissect = []
        #
        # r_dissect.append(meanSINR/20)

        # r_dissect.append(-1.0 * nOut/self.nUE)
        
        self.state = np.concatenate((self.bsLoc, self.ueLoc))

        done = False

        self.step_n += 1

        if self.step_n >= MAXSTEP:
            done = True

        # reward = max(sum(r_dissect), -1)
        reward = meanSINR/200

        info_tup = namedtuple('info_tup', ['r_dissect', 'step_n', 'ue_loc', 'bs_loc', 'outage_fraction', 'bs_actions'])
        info = info_tup(reward, self.step_n, self.ueLoc, self.bsLoc, (1.0 * nOut) / self.nUE, bsActions)
        return np.array(self.state), reward, done, info
    
    def step_test(self, action, ifrender=False): #(step)
        """
            similar to step(), but write here an individual function to
            avoid "if--else" in the original function to reduce training
            time cost
            """
        if self.mobility_model == "read_trace":
            self.ueLoc = self.ueLoc_trace[self.step_n]
        elif self.mobility_model == "group":
            pass
            # positions = next(self.mm)
            # # 2D to 3D
            # z = np.zeros((np.shape(positions)[0], 0))
            # self.ueLoc = np.concatenate((positions, z), axis=1).astype(int)
        self.bsLoc, bsActions = BS_move(self.bsLoc, self.boundaries, action, BS_STEP, MIN_BS_DIST + BS_STEP, N_ACT)
        self.association_map, meanSINR, nOut = self.channel.UpdateDroneNet(self.ueLoc, self.bsLoc, ifrender, self.step_n)
        
        self.bsLocGrid = GetGridMap(self.grid_n, self.grid_n, self.bsLoc)
        self.ueLocGrid = GetGridMap(self.grid_n, self.grid_n, self.ueLoc)
        
        # r_dissect = []
        #
        # r_dissect.append(meanSINR/20)
        #
        # r_dissect.append(-1.0 * nOut/self.nUE)

        self.state = np.concatenate((self.bsLoc, self.ueLoc))
        
        done = False
        
        self.step_n += 1
        
        if self.step_n >= MAXSTEP:
            done = True
    
        # reward = max(sum(r_dissect), -1)
        reward = meanSINR/200

        info_tup = namedtuple('info_tup', ['r_dissect', 'step_n', 'ue_loc', 'bs_loc', 'outage_fraction', 'bs_actions'])
        info = info_tup(reward, self.step_n, self.ueLoc, self.bsLoc, (1.0 * nOut) / self.nUE, bsActions)
        return np.array(self.state), reward, done, info


    def render(self):
        fig = figure(1, figsize=(20,20))
        for bs in range(self.nBS):
            subplot(self.nBS +1, 2, bs+1)
            title("bs " + str(bs) + "ue distribution")
            imshow(self.association[bs], interpolation='nearest', origin='lower')
            
            subplot(self.nBS +1, 2, self.nBS+ bs+1)
            title("bs " + str(bs) + "ue sinr distribution")
            imshow(self.association_sinr[bs], interpolation='nearest', origin='lower')

    def plot_sinr_map(self):
        fig = figure(1, figsize=(100,100))
        subplot(1, 2, 1)
        
        sinr_all = self.channel.GetSinrInArea(self.bsLoc)
        imshow(sinr_all, interpolation='nearest', origin='lower', vmin= -50, vmax = 100)
        colorbar()
        xlabel('x[m]')
        ylabel('y[m]')
        title('DL SINR [dB]')
        
        subplot(1,2,2)
        hist(sinr_all, bins=100, fc='k', ec='k')
        ylim((0,20))
        xlabel("SINR")
        ylabel("number of UEs")
        xlim(-100, 100)
        
        show()
        fig.savefig("sinr_map.pdf")
        
        np.save("sinr_map",sinr_all)
