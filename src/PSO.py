"""
    PSO.py:
    __author__   = "nathanrooy"
    __source__   = "https://github.com/nathanrooy/particle-swarm-optimization/blob/master/setup.py"
"""
from __future__ import division
import time
import random
import copy
from multiprocessing import Process, Manager
import os


def evaluate_in_parallel(swarms, costFunc, j, parameter, sim_cfg, ret_value):
    ret_value[j] = costFunc(swarms[j].position_i, parameter, sim_cfg)

class Particle:
    def __init__(self,x0):
        self.position_i=[]          # particle position
        self.velocity_i=[]          # particle velocity
        self.pos_best_i=[]          # best position individual
        self.err_best_i=-1          # best error individual
        self.err_i=-1               # error individual

        for i in range(0,num_dimensions):
            self.velocity_i.append(random.uniform(-1,1))
            self.position_i.append(x0[i])

    # evaluate current fitness
    def evaluate(self, err_i):
        self.err_i = err_i
        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i==-1:
            self.pos_best_i = copy.copy(self.position_i)
            self.err_best_i = self.err_i

    # update new particle velocity
    def update_velocity(self,pos_best_g, w=0.5, c1=1, c2=2):
        # w=0.5       # constant inertia weight (how much to weigh the previous velocity)
        # c1=1        # cognative constant
        # c2=2        # social constant

        for i in range(0,num_dimensions):
            r1=random.random()
            r2=random.random()
            
            vel_cognitive=c1*r1*(self.pos_best_i[i]-self.position_i[i])
            vel_social=c2*r2*(pos_best_g[i]-self.position_i[i])
            self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social

    # update the particle position based off new velocity updates
    def update_position(self,bounds):
        for i in range(0,num_dimensions):
            self.position_i[i]=self.position_i[i]+self.velocity_i[i]
            
            # adjust maximum position if necessary
            if self.position_i[i]>bounds[i][1]:
                self.position_i[i]=bounds[i][1]

            # adjust minimum position if neseccary
            if self.position_i[i]<bounds[i][0]:
                self.position_i[i]=bounds[i][0]

class PSO():
    def __init__(self, costFunc, algo_cfg, initial, parameter, sim_cfg, verbose=False):
        global num_dimensions
        num_dimensions=6 # TODO...

        bounds = []
        num_particles = algo_cfg['hyperparameter']['number_particles']
        maxiter = algo_cfg['number_iters']

        for veh_type in parameter:
            for param_type in parameter[veh_type]:
                bounds.append(parameter[veh_type][param_type])

        err_best_g=-1                   # best error for group
        pos_best_g=[]                   # best position for group

        # establish the swarms
        manager = Manager()
        swarms = []
        for i in range(0,num_particles):
            if initial == 'default':
                swarms.append(Particle([2.5, 1, 0.5, 0.1, 2.6, 4.5]))
            # if i < len(initial):
            #     swarms.append(Particle(initial[i]))
            # else:
            elif initial == 'random':
                random_solution = []
                for i in range(num_dimensions):
                    random_solution.append(random.uniform(bounds[i][0], bounds[i][1]))
                swarms.append(Particle(random_solution))

        checkpoint_folder = sim_cfg['log_folder'] + '/checkpoint'
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        # begin optimization loop
        i=0
        while i<maxiter:            
            if verbose:
                print(f'iter: {i:>4d}, best solution: {err_best_g:10.6f}')
                for idx, sw in enumerate(swarms):
                    print(f'Particle: {idx:>3d}, position: {sw.position_i}')
                print()
            
            # cycle through particles in swarms and evaluate fitness
            Processes = {}
            ret_value = manager.list([0]*num_particles)

            start_idx = 0
            while True:
                for j in range(algo_cfg['number_processors']):
                    if start_idx + j >= num_particles:
                        break
                    Processes[start_idx + j] = Process(target=evaluate_in_parallel, args=(swarms,costFunc, j, parameter, sim_cfg, ret_value))
                    Processes[start_idx + j].start()
                for j in range(algo_cfg['number_processors']):
                    if start_idx + j >= num_particles:
                        break
                    Processes[start_idx + j].join()
                start_idx += algo_cfg['number_processors']
                if start_idx >= num_particles:
                    break 
                
            # for j in range(0,num_particles):
            #     Processes[j] = Process(target=evaluate_in_parallel, args=(swarms,costFunc, j, parameter, sim_cfg, ret_value))
            #     Processes[j].start()

            # for j in range(0,num_particles):
            #     Processes[j].join()
            
            for j in range(0,num_particles):
                swarms[j].evaluate(ret_value[j])
                                            
            # determine if current particle is the best (globally)
            for j in range(0,num_particles):
                if swarms[j].err_i < err_best_g or err_best_g==-1:
                    pos_best_g=list(swarms[j].position_i)
                    err_best_g=float(swarms[j].err_i)
            
            f = open('{}/iteration-{}.txt'.format(checkpoint_folder, i), 'w')
            for j in range(0,num_particles):
                f.write("%s,%s\n" % (str(swarms[j].position_i), str(swarms[j].err_i)))
            f.write("%s,%s\n" % (str(pos_best_g), str(err_best_g)))
            f.close()

            # cycle through swarm and update velocities and position
            for j in range(0,num_particles):
                swarms[j].update_velocity(pos_best_g, w=algo_cfg['hyperparameter']['w'], c1=algo_cfg['hyperparameter']['c1'], c2=algo_cfg['hyperparameter']['c2'])
                swarms[j].update_position(bounds)
            i+=1
        
        f = open('{}/iteration-{}.txt'.format(checkpoint_folder, i), 'w')
        for j in range(0,num_particles):
            f.write("%s,%s\n" % (str(swarms[j].position_i), str(swarms[j].err_i)))
        f.write("%s,%s\n" % (str(pos_best_g), str(err_best_g)))
        f.close()
        
        # print final results
        print('\nFINAL SOLUTION:')
        print(f'   > {pos_best_g}')
        print(f'   > {err_best_g}\n')