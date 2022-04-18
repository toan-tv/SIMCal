# _SIMCal_: A High-Performance Toolkit For Calibrating Traffic Simulation

### _SIMCal_'s framework:

![alt text](/imgs/simcal-framework-1.png)


### Configuration examples:
* FireFly Algorithm: /Corridor/FA_cfg.yaml
* Genetic Algorithm: /Corridor/GA_cfg.yaml
* Particle Swarm Optimization: /Corridor/PSO_cfg.yaml
* Simulated Annealing: /Corridor/SA_cfg.yaml

### To calibrate:
> python3 --config_file=/Corridor/GA_cfg.yaml

### Visualization:
Please refer to jupyter notebook files

### Some results:
#### Calibration for MLK Corridor (10 intersections) at Chattanooga, TN, USA

<table>
  <tr>
     <td> Performance of algorithms</td>
     <td> Performance when using different population sizes</td>
  </tr>
  <tr>
    <td valign="top"><img src="/imgs/algo_bar_chart-1.png"></td>
    <td valign="top"><img src="/imgs/pop_size_line_result-1.png"></td>
  </tr>
 </table>

#### Calibration for a city-level network -- Chattanooga, TN, USA
<table>
  <tr>
     <td> Simulation in SUMO, having around 12,000 KM of roads</td>
     <td> Took around 2.5 days to get the optimal solution</td>
  </tr>
  <tr>
    <td valign="top"><img src="/imgs/Chatt.png"></td>
    <td valign="top"><img src="/imgs/network_size_result-1.png"></td>
  </tr>
 </table>
