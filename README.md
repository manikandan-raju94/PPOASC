**PPOASC: PPO-Based Adaptive Signal Control**

**Overview**
PPOASC (PPO-Based Adaptive Signal Control) is a reinforcement learning framework designed for adaptive traffic signal control in urban transportation networks using the Simulation of Urban MObility (SUMO) environment. The framework employs Proximal Policy Optimization (PPO) to dynamically optimize traffic signal operations under varying traffic demand conditions.

This repository provides two implementations of the proposed framework:
•	PASC-WComm: PPO-based adaptive signal control without inter-agent communication, where each intersection operates independently using local traffic observations.
•	PASC-Comm: PPO-based adaptive signal control with inter-agent communication, where neighboring intersections exchange traffic information to improve coordination and overall network performance.
The objective is to reduce traffic congestion, minimize vehicle waiting time, improve throughput, and enhance urban traffic efficiency under light, medium, and heavy traffic conditions.

## Repository Structure

- `pasc_wcomm.py` – PPO Adaptive Signal Control (Without Communication)
- `pasc_comm.py` – PPO Adaptive Signal Control (With Communication)
- `traffic_env_wcomm.py` – Environment for PASC-WComm
- `traffic_env_comm.py` – Environment for PASC-Comm
- `logger.py` – Logging utilities
- `runsimulation.sumocfg` – SUMO configuration file
- `Network/` – SUMO network files
- `Demand/` – Traffic demand and route files
- `models/` – Trained models
- `checkpoints/` – PPO checkpoints
- `logs/` – Training logs
- `tb_logs/` – TensorBoard logs

  
**Features**
•	PPO-based adaptive traffic signal control
•	Multi-intersection traffic management
•	Dynamic traffic demand generation
•	SUMO-based simulation environment
•	Support for light, medium, and heavy traffic scenarios
•	Performance logging and model checkpointing
•	Communication-enabled and communication-free configurations
•	Reproducible research implementation

**Requirements**
Software
•	Python 3.10 or later
•	SUMO (Simulation of Urban MObility)

**Python Packages**
Install all required packages using:
pip install -r requirements.txt
or
pip install numpy pandas matplotlib gymnasium stable-baselines3 torch tensorboard traci sumolib lxml

SUMO Installation
Install SUMO from:
https://sumo.dlr.de/
Verify installation:
sumo --version
Ensure SUMO is available in the system PATH.

**Running PASC-WComm (Without Communication)**
To execute PPO-based adaptive signal control without inter-agent communication:
python pasc_wcomm.py
In this version, each intersection operates independently using only local traffic information.

**Running PASC-Comm (With Communication)**
To execute PPO-based adaptive signal control with inter-agent communication:
python pasc_comm.py
In this version, neighboring intersections exchange traffic-related information to improve coordination and network-wide traffic management.

**Training Output**
During execution, the framework will:
•	Launch the SUMO simulation environment
•	Generate traffic demand scenarios
•	Train PPO agents
•	Record traffic performance metrics
•	Save checkpoints periodically
•	Store trained models
•	Generate TensorBoard logs

**Generated outputs include:**
logs/
models/
checkpoints/
tb_logs/

**Traffic Scenarios**
The framework supports multiple traffic demand levels:
•	Light Traffic
•	Medium Traffic
•	Heavy Traffic

Traffic demand files are located in the Demand directory and can be modified according to specific experimental requirements.

**Research Objective**
The proposed PPOASC framework aims to:
•	Minimize vehicle waiting times
•	Reduce queue lengths
•	Improve traffic throughput
•	Reduce congestion and spillback effects
•	Adapt signal timings dynamically
•	Enhance urban traffic efficiency through intelligent control

**Citation**
If you use this repository in your research, please cite the corresponding publication describing the PPOASC framework.
Data Availability
The source code, simulation configurations, traffic demand files, and network models used in this study are publicly available in this repository to support transparency, reproducibility, and future research in adaptive traffic signal control and intelligent transportation systems.
Repository:
https://github.com/manikandan-raju94/PPOASC

**Author**
R. Manikandan
Ph.D. Research Scholar
Vellore Institute of Technology (VIT)
Tamil Nadu, India

**License**
This repository is provided for academic and research purposes.

