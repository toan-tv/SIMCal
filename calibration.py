import pickle as pkl
import random
from src.PSO import PSO
from src.GA import GA
from src.SA import minimize
import yaml
import sys
import os
import argparse
from src.FA import FireflyAlgorithm


parser = argparse.ArgumentParser()
parser.add_argument('--config_file', action='store', type=str, dest='config_file')
args = vars(parser.parse_args())

cfg = yaml.safe_load(open(args['config_file'], 'r'))
print(cfg)

#--- COST FUNCTION ------------------------------------------------------------+
def cost_func(x, parameter=cfg['parameter'], sim_cfg=cfg['simulation']):
    import traci
    import datetime as dt
    import numpy as np
    import pandas as pd
    from dataset.Corridor.tlProgram import tlPrograms

    # Actuated control ----------------------------------------------------------------
    for intersection_idx in range(len(tlPrograms['traffic_lights'])):
        phase_plans = tlPrograms['traffic_lights'][intersection_idx]['phase_plans']
        for phase_idx in range(len(phase_plans)):
            phase_plans[phase_idx]['idx'] = phase_idx
    # ----------------------------------------------------------------------------------

    sumo_cmd = ["/usr/bin/sumo"]
    sumo_config = ["-c", sim_cfg['sumo_cfg']]
    sumo_cmd.extend(sumo_config)
    
    df_obs = pd.read_csv(sim_cfg['obs_data'])
    df_obs['measurement_tstamp'] = pd.to_datetime(df_obs['time'])
    df_obs['speed'] = df_obs['speed']*0.44704
    df_obs = df_obs[df_obs['speed'] > 0]
    detector_ids = df_obs.detector.unique()

    traci.start(sumo_cmd, label=str(random.randint(10000, 50000)))

    for veh_type in parameter:
        for parameter_type in parameter[veh_type]:
            if parameter_type == 'MinGap':
                traci.vehicletype.setMinGap(veh_type, x[0])
            elif parameter_type == 'Tau':
                traci.vehicletype.setTau(veh_type, x[1])
            elif parameter_type == 'Imperfection':
                traci.vehicletype.setImperfection(veh_type, x[2])
            elif parameter_type == 'SpeedDeviation':
                traci.vehicletype.setSpeedDeviation(veh_type, x[3])
            elif parameter_type == 'Accel':
                traci.vehicletype.setAccel(veh_type, x[4])
            elif parameter_type == 'Decel':
                traci.vehicletype.setDecel(veh_type, x[5])
            else:
                print("<<< Not support {} >>>".format(parameter_type))
                sys.exit(0)

    historical_speed_data = {}
    speed_diffs = {}
    for detector_id in detector_ids:
        historical_speed_data[detector_id] = []

    count = int(traci.simulation.getTime())
    num_steps = int((sim_cfg['end'] - sim_cfg['start']).total_seconds()) + count
    jump_step = int(sim_cfg['step'])*60
    while count < num_steps:
        # Actuated control ---------------------------------------------------------------------
        for intersection_idx in range(len(tlPrograms['traffic_lights'])):
            phase_plans = tlPrograms['traffic_lights'][intersection_idx]['phase_plans']
            if len(phase_plans) <= 0:
                continue
            if count >= phase_plans[0]['time']:
                traci.trafficlight.setProgram(tlPrograms['traffic_lights'][intersection_idx]['node_id'], str(int(phase_plans[0]['idx'])))
                del phase_plans[0]
        # --------------------------------------------------------------------------------------

        traci.simulationStep()
        if count % jump_step == 0:
            if (count / jump_step) > 0:
                count_idx = int((count / jump_step) - 1)
                speed_diffs[count_idx] = {}
                for detector_id in detector_ids:
                    if len(historical_speed_data[detector_id]) <= 0:
                        continue
                    # get obs speed
                    df_obs_detector = df_obs[(df_obs['detector'] == detector_id) & (df_obs['measurement_tstamp'] \
                                                         == np.datetime64(sim_cfg['start'])+np.timedelta64(jump_step*count_idx, 's'))]
                    if len(df_obs_detector) == 0:
                        continue
                    obs_speed = df_obs_detector['speed'].values[0]
                    # get sim speed
                    sim_speed = np.mean(historical_speed_data[detector_id])
                    speed_diffs[count_idx][detector_id] = (obs_speed, sim_speed)
                    historical_speed_data[detector_id] = []

        for detector_id in detector_ids:
            try:
                sim_speed_ = traci.inductionloop.getLastStepMeanSpeed(detector_id)
            except:
                sim_speed = -1
            if sim_speed_ != -1:
                historical_speed_data[detector_id].append(sim_speed_)
        count += 1 
    traci.close()

    arr_MAPE = []
    for step in speed_diffs.keys():
        arr_err = []
        for dectector_id in speed_diffs[step].keys():
            mape = abs(speed_diffs[step][dectector_id][0] - speed_diffs[step][dectector_id][1]) / \
                        speed_diffs[step][dectector_id][0]
            arr_err.append(mape)
        if len(arr_err) > 0:
            arr_MAPE.append(np.mean(mape))
    MAPE = np.mean(arr_MAPE)
    
    result_folder = sim_cfg['log_folder'] + '/result/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    obj = {'errors': speed_diffs, 'MAPE': MAPE, 'x': x}
    file_name = result_folder + str(x)
    pkl.dump(obj, open(file_name, 'wb'))
    
    return MAPE
#--- MAIN ---------------------------------------------------------------------+

if cfg['algorithm']['name'] == 'PSO':
    PSO(cost_func, cfg['algorithm'], cfg['initial'], cfg['parameter'], cfg['simulation'], verbose=True)
elif cfg['algorithm']['name'] == 'GA':
    GA(cost_func, cfg['algorithm'], cfg['initial'], cfg['parameter'], cfg['simulation'], verbose=True)
elif cfg['algorithm']['name'] == 'SA':
    if cfg['initial'] == 'default':
        opt = minimize(cost_func, [2.5, 1, 0.5, 0.1, 2.6, 4.5], opt_mode='continuous', step_max=50, t_max=1, t_min=0,
                    bounds=[[1, 4], [0, 2], [0, 1], [0.01, 0.2], [0.6, 4.6], [2.5, 6.5]])    
        opt.results()
elif cfg['algorithm']['name'] == 'FA':
    best, MAPE = FireflyAlgorithm().minimizer(function=cost_func, dim=6, parameter=cfg['parameter'], 
            sim_cfg=cfg['simulation'], lb=[1, 0, 0, 0.01, 0.6, 2.5], ub=[4, 2, 1, 0.2, 4.6, 6.5], max_evals=50)
    print(best, MAPE)