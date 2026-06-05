import numpy as np
import gymnasium as gym
from gymnasium import spaces

import os
import sys
import time
import random
import glob
import xml.etree.ElementTree as ET

from datetime import datetime
from contextlib import suppress
from collections import deque

import traci
import traci.constants as tc


# ============================================================
# =============== GLOBAL CONSTANTS (CLEAN) ====================
# ============================================================

DELTA_TIME = 15
MAX_SIM_TIME = 3600
QUEUE_TH = 6
SPEED_STOP = 0.3
MAX_VEHICLES_SAMPLED = 200
SAMPLE_INTERVAL = 10


# ETC THRESHOLDS (Research Mode)
ETC_QUEUE_LIMIT = 12
ETC_SPILLBACK_LIMIT = 4
ETC_SPEED_DROP = 3.0   # m/s threshold for speed drop penalty

# REWARD WEIGHTS (Research Mode)
WEIGHTS = {
    "queue":  -0.35,
    "pressure": -0.25,
    "spillback": -0.50,
    "wait": -0.20,
    "speed_drop": -0.15,
    "neighbor_pressure": -0.18,
    "transition": -0.10,

    # positives
    "flow": +1.00
}


# ===================================================================================
# =============== ADJACENCY MAP for MULTI AGENT COMMUNICATION ======================
# ===================================================================================

def build_adjacency(junction_ids, net_file):
    """
    Build neighbor relationships between junctions
    based on SUMO edge connectivity.
    """

    tree = ET.parse(net_file).getroot()

    edge_to_from = {}
    edge_to_to = {}

    # map edge → junctions
    for edge in tree.findall("edge"):
        if "from" in edge.attrib and "to" in edge.attrib:
            edge_to_from[edge.attrib["id"]] = edge.attrib["from"]
            edge_to_to[edge.attrib["id"]] = edge.attrib["to"]

    adjacency = {jid: [] for jid in junction_ids}

    for jid in junction_ids:

        try:
            controlled_lanes = traci.trafficlight.getControlledLanes(jid)
        except:
            continue

        inbound_edges = {lane.split("_")[0] for lane in controlled_lanes}

        for edge in inbound_edges:

            if edge not in edge_to_from:
                continue

            upstream = edge_to_from[edge]
            downstream = edge_to_to[edge]

            if upstream in adjacency and upstream != jid:
                adjacency[jid].append(upstream)

            if downstream in adjacency and downstream != jid:
                adjacency[jid].append(downstream)

        adjacency[jid] = list(set(adjacency[jid]))

    return adjacency
# ======================================================================
# =============== CORRECTED TRAFFIC LIGHT PHASE MANAGER =================
# ======================================================================

class TrafficLightPhaseManager:

    PHASE_GREEN = 0
    PHASE_YELLOW = 1
    PHASE_RED = 2

    def __init__(self):

        self.junction_phases = {}
        self.phase_types = {}
        self.phase_durations = {}
        self.green_indices = {}

        self.transition_map = {}

        self.current = {}
        self.timer = {}

        self.target_green = {}
        self.in_transition = {}
       

    # -------------------------------------------------
    # Detect GREEN / YELLOW / RED
    # -------------------------------------------------
    def classify_phase(self, state_str):

        s = state_str.lower()

        if "y" in s:
            return self.PHASE_YELLOW

        if "g" in s:
            return self.PHASE_GREEN

        return self.PHASE_RED


    # -------------------------------------------------
    # Load phases for a junction
    # -------------------------------------------------
    def load_junction_phases(self, jid, phases):

        self.junction_phases[jid] = phases
        self.phase_types[jid] = []
        self.phase_durations[jid] = []

        for ph in phases:

            ptype = self.classify_phase(ph.state)

            self.phase_types[jid].append(ptype)
            self.phase_durations[jid].append(ph.duration)

        greens = []

        for idx, t in enumerate(self.phase_types[jid]):
            if t == self.PHASE_GREEN:
                greens.append(idx)

        self.green_indices[jid] = greens

        # build transition map
        self.transition_map[jid] = self._build_transition(jid)

        # initialize state
        if len(greens) == 0:
            raise ValueError(f"No green phases found for junction {jid}")

        self.current[jid] = greens[0]
        self.timer[jid] = 0
        self.target_green[jid] = greens[0] if greens else 0
        self.in_transition[jid] = False


    # -------------------------------------------------
    # Build transition sequence
    # -------------------------------------------------
    def _build_transition(self, jid):

        mapping = {}

        phases = self.junction_phases[jid]
        types = self.phase_types[jid]

        N = len(phases)

        for idx in self.green_indices[jid]:

            yellow = None
            red = None

            # next yellow
            if idx + 1 < N and types[idx + 1] == self.PHASE_YELLOW:
                yellow = idx + 1

                # optional red
                if idx + 2 < N and types[idx + 2] == self.PHASE_RED:
                    red = idx + 2

            # find next green
            next_green = None

            for j in range(idx + 1, idx + 1 + N):

                k = j % N

                if types[k] == self.PHASE_GREEN:
                    next_green = k
                    break

            if next_green is None:
                next_green = idx

            mapping[idx] = {
                "yellow": yellow,
                "red": red,
                "next": next_green
            }

        return mapping


    # -------------------------------------------------
    # Set new target green phase
    # -------------------------------------------------
    def set_target(self, jid, target_green):

        self.target_green[jid] = target_green

        if self.current[jid] != target_green:

            self.in_transition[jid] = True
            self.timer[jid] = 0


    # -------------------------------------------------
    # Update phase (called every DELTA_TIME)
    # -------------------------------------------------
    def update(self, jid, dt, traffic_mode, queue):

        self.timer[jid] += dt
        cur = self.current[jid]

        phase_type = self.phase_types[jid][cur]

        # --------------------------------
        # Adaptive green durations
        # --------------------------------
        if traffic_mode == "light":
            MIN_GREEN, MAX_GREEN = 6, 25

        elif traffic_mode in ["medium_low", "medium_mid", "medium"]:
            MIN_GREEN, MAX_GREEN = 8, 35

        elif traffic_mode in ["heavy_low", "heavy_mid", "heavy"]:
            MIN_GREEN, MAX_GREEN = 10, 45

        else:
            MIN_GREEN = 8
            MAX_GREEN = 35

        if queue > 40:
            MAX_GREEN = min(MAX_GREEN + 15, 60)
        # --------------------------------
        # GREEN phase logic
        # --------------------------------
        if phase_type == self.PHASE_GREEN and not self.in_transition[jid]:

            if self.timer[jid] < MIN_GREEN:
                return cur

            if self.timer[jid] >= MAX_GREEN:
                self.in_transition[jid] = True
                self.timer[jid] = 0

            else:
                return cur

        # --------------------------------
        # Transition logic
        # --------------------------------
        if self.in_transition[jid]:

            T = self.transition_map[jid]
            target = self.target_green[jid]

            mapping = T.get(cur, None)

            if mapping is None:
                self.current[jid] = target
                self.in_transition[jid] = False
                self.timer[jid] = 0
                return target

            yellow = mapping["yellow"]
            red = mapping["red"]

            if yellow is not None and cur != yellow and cur != red:
                self.current[jid] = yellow
                self.timer[jid] = 0
                return yellow

            if red is not None and cur == yellow:
                self.current[jid] = red
                self.timer[jid] = 0
                return red

            if cur == red or red is None:
                self.current[jid] = target
                self.in_transition[jid] = False
                self.timer[jid] = 0
                return target

        return cur

       

    # -------------------------------------------------
    # Get current phase
    # -------------------------------------------------
    def get(self, jid):

        return self.current.get(jid, 0)


    # -------------------------------------------------
    # Check transition state
    # -------------------------------------------------
    def is_transitioning(self, jid):

        return self.in_transition.get(jid, False)


    # -------------------------------------------------
    # Force phase change (used at reset)
    # -------------------------------------------------
    def force(self, jid, phase_idx):

        self.current[jid] = phase_idx
        self.timer[jid] = 0
        self.in_transition[jid] = False
    # ===================================================================================
# =======================  GRID MARL ENVIRONMENT (CORE)  ============================
# ===================================================================================

class GridMarlEnv(gym.Env):

    metadata = {"render_modes": []}

    reward_profiles = {
        "light": {
            "queue_weight": 0.12, "pressure_weight": 0.08, 
            "spillback_weight": 0.20, "waiting_weight": 0.02,
            "neighbor_weight": 0.10, "occupancy_weight": 0.05,
            "speed_weight": 0.08, "teleport_weight": 0.15,
            "stuck_weight": 0.20, "transition_penalty": 0.05,
            "flow_bonus": 0.5, "etc_bonus": 0.3,
            "queue_threshold": 15, "waiting_threshold": 30,
            "occupancy_threshold": 0.25, "speed_threshold": 8.33
        },
        "medium": {
            "queue_weight": 0.15, "pressure_weight": 0.10, 
            "spillback_weight": 0.25, "waiting_weight": 0.03,
            "neighbor_weight": 0.12, "occupancy_weight": 0.06,
            "speed_weight": 0.10, "teleport_weight": 0.20,
            "stuck_weight": 0.25, "transition_penalty": 0.05,
            "flow_bonus": 0.3, "etc_bonus": 0.2,
            "queue_threshold": 25, "waiting_threshold": 45,
            "occupancy_threshold": 0.35, "speed_threshold": 6.94
        },
        "heavy": {
            "queue_weight": 0.18, "pressure_weight": 0.12, 
            "spillback_weight": 0.30, "waiting_weight": 0.04,
            "neighbor_weight": 0.15, "occupancy_weight": 0.07,
            "speed_weight": 0.12, "teleport_weight": 0.25,
            "stuck_weight": 0.30, "transition_penalty": 0.05,
            "flow_bonus": 0.1, "etc_bonus": 0.1,
            "queue_threshold": 40, "waiting_threshold": 60,
            "occupancy_threshold": 0.45, "speed_threshold": 5.56
        }
    }

   

    def __init__(self, net_file, sumo_cfg, seed=42, pre_generated_route_file=None):
        super().__init__()
        self.seed_value = seed
        np.random.seed(seed)
        random.seed(seed)

        self.net_file = net_file
        self.sumo_cfg = sumo_cfg
        self.pre_generated_route_file = pre_generated_route_file

        self.mode = "light"
        
        # load junctions from NET.xml
        self.junctions = self._extract_junctions(net_file)
        self.num_agents = len(self.junctions)

        # adjacency (multi-agent)
        self.adjacency = None

        # observation dims
        self.obs_dim = 10   # updated: includes queue, press., spill, phase, phase_type, neighbor pressure
        self.observation_space = spaces.Box(
            low=-5, high=5, shape=(self.num_agents, self.obs_dim), dtype=np.float32
        )

        # dynamic: action space later replaced after loading phases
        self.action_space = spaces.MultiDiscrete([1] * self.num_agents)

        # ---- performance caches ----
        self.phase_manager = TrafficLightPhaseManager()
        self.lane_to_junc = {}
        self.lanes = []
        self.lane_list = []

        # tracking
        self.obs = None
        self.queue_j = None
        self.pressure_j = None
        self.spill_j = None
        self.phase_time = None

        # global vehicle tracking
        self.vehicle_wait = {}
        self.wait_samples = []
        self.total_seen = set()
        self.active_vehicles = set()
        self.completed = 0
        self.prev_completed = 0
        self.stuck = {}

        self.episode = 0
        self.step_count = 0
        self._episode_logged = False

        # replay buffers
        self.deque_queue = deque(maxlen=2000)
        self.deque_pressure = deque(maxlen=2000)
        self.deque_spill = deque(maxlen=2000)
        self.deque_rewards = deque(maxlen=2000)

    def build_action_space(self):
        phase_counts = []
        for jid in self.junctions:
            greens = self.phase_manager.green_indices[jid]
            phase_counts.append(len(greens))
        self.action_space = spaces.MultiDiscrete(phase_counts)
        print("Action space:", phase_counts)

    
    def _pressure_correction(self, actions):

        corrected = actions.copy()

        for i, jid in enumerate(self.junctions):

            greens = self.phase_manager.green_indices[jid]

            if len(greens) <= 1:
                continue

            pressure = self.pressure_j[i]

            # if congestion high, override RL
            if pressure > 30:

                best_action = actions[i]
                best_queue = -1

                lanes = traci.trafficlight.getControlledLanes(jid)

                for idx in range(len(greens)):

                    queue = 0

                    for lane in lanes:
                        queue += traci.lane.getLastStepHaltingNumber(lane)

                    if queue > best_queue:
                        best_queue = queue
                        best_action = idx

                corrected[i] = best_action

        return corrected

    def _apply_actions(self, actions):

        for i, jid in enumerate(self.junctions):

            # prevent too frequent switching
            if self.phase_time[i] < 15:
                continue

            green_list = self.phase_manager.green_indices[jid]

            action = int(actions[i])

            if action >= len(green_list):
                action = 0

            target_green = green_list[action]

            cur = self.phase_manager.get(jid)

            if cur != target_green:
                self.phase_manager.set_target(jid, target_green)
                self.phase_time[i] = 0

    
    # ================================================================
    # Extract all SUMO traffic-light junctions from NET file
    # ================================================================
    def _extract_junctions(self, net_file):
        root = ET.parse(net_file).getroot()
        arr = [tl.attrib["id"] for tl in root.findall("tlLogic")]
        return arr

     

    # ================================================================
    # RESET: full SUMO episode reset
    # ================================================================
    def reset(self, seed=None, options=None):
        self.episode += 1

        if seed is not None:
            self.seed_value = seed
            np.random.seed(seed)
            random.seed(seed)

        if self.episode % 10 == 0:
            print(f"[SEED DEBUG] Episode {self.episode}, Seed: {self.seed_value}")

        self._episode_logged = False
        self.step_count = 0
        

        self.vehicle_wait.clear()
        self.wait_samples.clear()
        self.total_seen.clear()
        self.active_vehicles.clear()
        self.completed = 0
        self.prev_completed = 0
        self.stuck.clear()

        self.deque_queue.clear()
        self.deque_pressure.clear()
        self.deque_spill.clear()
        self.deque_rewards.clear()
        
        scale = self._get_demand_scale()

        # ------------------------------------------------
        # ROUTE SELECTION
        # ------------------------------------------------

        # If evaluation forces a route file
        if hasattr(self, "route_file") and self.route_file is not None:

            route_file = self.route_file
            print("Using forced route file:", route_file)

        else:

            scale = self._get_demand_scale()

            print("Episode:", self.episode)
            print("Demand scale:", scale)

            # ------------------------------------------------
            # Dynamic Demand Scaling
            # ------------------------------------------------

            if self.mode == "light":
                route_file = os.path.join(os.getcwd(), "Demand", "light_traffic.rou.xml")

            elif self.mode == "medium_low":
                route_file = os.path.join(os.getcwd(), "Demand", "medium_low_traffic.rou.xml")

            elif self.mode == "medium_mid":
                route_file = os.path.join(os.getcwd(), "Demand", "medium_mid_traffic.rou.xml")

            elif self.mode == "medium":
                route_file = os.path.join(os.getcwd(), "Demand", "medium_traffic.rou.xml")

            elif self.mode == "heavy_low":
                route_file = os.path.join(os.getcwd(), "Demand", "heavy_low_traffic.rou.xml")

            elif self.mode == "heavy_mid":
                route_file = os.path.join(os.getcwd(), "Demand", "heavy_mid_traffic.rou.xml")

            elif self.mode == "heavy":
                route_file = os.path.join(os.getcwd(), "Demand", "heavy_traffic.rou.xml")
            else:
                raise ValueError(f"Unknown traffic mode: {self.mode}")

        if not os.path.exists(route_file):
            raise FileNotFoundError(f"Route file not found: {route_file}")

        print("Using route file:", route_file)

        # teleport timing
       # teleport timing
        if self.mode == "light":
            teleport = "120"     # vehicles teleport after 2 minutes

        elif self.mode in ["medium_low", "medium_mid", "medium"]:
            teleport = "600"     # teleport after 10 minutes

        elif self.mode in ["heavy_low", "heavy_mid", "heavy"]:
            teleport = "-1"      # disable teleport (vehicles never teleport)

        else:
            teleport = "600"     # safe default

        # close previous simulation
        with suppress(Exception):
            if traci.isLoaded():
                traci.close()

        # launch SUMO
        cmd = [
        "sumo",
        "-c", self.sumo_cfg,
        "--route-files", route_file,
        "--seed", str(self.seed_value),
        "--no-step-log",
        "--no-warnings",
        "--time-to-teleport", teleport
        ]
        traci.start(cmd)
        time.sleep(0.2)

        # load updated junctions from SUMO
        self.junctions = list(traci.trafficlight.getIDList())
        self.num_agents = len(self.junctions)

        # adjacency ON (recommended)
        self.adjacency = build_adjacency(self.junctions, self.net_file)

        # allocate arrays
       # allocate arrays
        self.obs = np.zeros((self.num_agents, self.obs_dim), dtype=np.float32)

        self.queue_j = np.zeros(self.num_agents)
        self.pressure_j = np.zeros(self.num_agents)

        # ADD THIS LINE
        self.prev_pressure = np.zeros(self.num_agents)

        self.spill_j = np.zeros(self.num_agents)
        self.phase_time = np.zeros(self.num_agents)

        # mapping arrays
        self.j2i = {j: i for i, j in enumerate(self.junctions)}
        self.i2j = {i: j for i, j in enumerate(self.junctions)}

        # load light phases
        self.green_map = {}
        for j in self.junctions:
            try:
                logic = traci.trafficlight.getAllProgramLogics(j)[0]
                phases = logic.phases
                self.phase_manager.load_junction_phases(j, phases)

                greens = self.phase_manager.green_indices[j]
                self.green_map[j] = greens

                # force initial green
                traci.trafficlight.setPhase(j, greens[0])
                self.phase_manager.force(j, greens[0])

            except Exception as e:
                print(f"[WARN] Phase load fail at {j}: {e}")
                self.green_map[j] = [0]

        # lane → junction
        self.build_action_space()
        self.lane_to_junc.clear()
        self.lanes.clear()

        for j in self.junctions:
            for lane in traci.trafficlight.getControlledLanes(j):
                if lane not in self.lane_to_junc:
                    self.lane_to_junc[lane] = j
                    self.lanes.append(lane)

        self.lane_list = list(self.lanes)

        
        # subscribe lanes
        for lane in self.lane_list:
            with suppress(Exception):
                traci.lane.subscribe(lane, [
                    tc.LAST_STEP_VEHICLE_NUMBER,
                    tc.LAST_STEP_HALTING_NUMBER,
                    tc.LAST_STEP_MEAN_SPEED
                ])

        # advance simulation so subscriptions become active
        traci.simulationStep()

       

        return self._get_obs(), {}
    
    def set_logger(self, logger):
        self.logger = logger

    def set_mode(self, mode):
        valid_modes = [
            "light",
            "medium_low",
            "medium_mid",
            "medium",
            "heavy_low",
            "heavy_mid",
            "heavy"
        ]

        if mode not in valid_modes:
            raise ValueError(f"Invalid traffic mode: {mode}")

        self.mode = mode

    def _get_demand_scale(self):
        return 1.0

    # ================================================================
    # OBSERVATION BUILDER
    # ================================================================
    def _get_obs(self):

        # reset metrics
        self.queue_j.fill(0)
        self.pressure_j.fill(0)
        self.spill_j.fill(0)

        speed_drop = 0

        lane_to_junc = self.lane_to_junc
        j2i = self.j2i

        debug_lane_sample = None
        debug_queue_sum = 0
        debug_lane_counter = 0

        # -----------------------------
        # Lane aggregation
        # -----------------------------
        for lane in self.lane_list:

            try:
                # Direct SUMO queries (more reliable than subscriptions)
                q = traci.lane.getLastStepHaltingNumber(lane)
                s = traci.lane.getLastStepMeanSpeed(lane)

                if debug_lane_sample is None:
                    debug_lane_sample = (lane, q, s)

                j = lane_to_junc[lane]
                idx = j2i[j]

                self.queue_j[idx] += q
                debug_queue_sum += q
                debug_lane_counter += 1

                # pressure calculation
                pressure = q * (1 - min(s/13.9,1))
                self.pressure_j[idx] += pressure

                # spillback detection
                if q > QUEUE_TH and s < SPEED_STOP:
                    self.spill_j[idx] += 1

                # speed drop metric
                if s < ETC_SPEED_DROP:
                    speed_drop += 1

            except Exception as e:
                print("Lane error:", lane, e)

        # # -----------------------------
        # # DEBUG PRINT (every 100 steps)
        # # -----------------------------
        # if self.step_count % 100 == 0:

        #     print("\n--- DEBUG OBS ---")
        #     print("Step:", self.step_count)
        #     print("Total lanes:", len(self.lane_list))
        #     print("Processed lanes:", debug_lane_counter)

        #     if debug_lane_sample:
        #         lane, q, s = debug_lane_sample
        #         print("Sample lane:", lane)
        #         print("Sample halting vehicles:", q)
        #         print("Sample speed:", s)

        #     print("Total queue detected:", debug_queue_sum)
        #     print("Queue per junction:", self.queue_j[:min(5, len(self.queue_j))])
        #     print("Pressure per junction:", self.pressure_j[:min(5, len(self.pressure_j))])
        #     print("-----------------\n")

        # normalize speed drop metric
        self.speed_drop_metric = speed_drop / max(len(self.lane_list), 1)

        # -----------------------------
        # Pre-calc shared values
        # -----------------------------
        active_ratio = len(self.active_vehicles) / 1000.0

        if self.wait_samples:
            last_wait = min(self.wait_samples[-1], 600) / 600.0
        else:
            last_wait = 0.0

        # -----------------------------
        # Build observation
        # -----------------------------
        for idx, j in enumerate(self.junctions):

            ph = self.phase_manager.get(j)
            ptype = self.phase_manager.phase_types[j][ph]

            neigh = self.adjacency.get(j, [])

            if neigh:
                neigh_press = np.mean([self.pressure_j[j2i[n]] for n in neigh])
            else:
                neigh_press = 0.0

            self.obs[idx] = np.array([
                self.queue_j[idx] / 20.0,
                self.pressure_j[idx] / 50.0,
                self.spill_j[idx] / 5.0,
                ph / 10.0,
                ptype / 2.0,
                self.phase_time[idx] / 50.0,
                neigh_press / 30.0,
                active_ratio,
                last_wait,
                1.0 if self.phase_manager.is_transitioning(j) else 0.0
            ], dtype=np.float32)

        return self.obs
    # ===================================================================================
# =========================   REWARD FUNCTION (RESEARCH MODE)  ======================
# ===================================================================================

    def _get_context_weights(self):

        avg_queue = np.mean(self.queue_j)

        # network congestion index
        NCI = avg_queue / 80.0

        if NCI < 0.2:
            context = "light"

        elif NCI < 0.5:
            context = "medium"

        else:
            context = "heavy"


        if context == "light":

            weights = {
                "queue": -0.2,
                "pressure": -0.1,
                "spillback": -0.2,
                "wait": -0.05,
                "flow": 3.0
            }

        elif context == "medium":

            weights = {
                "queue": -0.8,
                "pressure": -0.6,
                "spillback": -0.5,
                "wait": -0.2,
                "flow": 2.0
            }

        else:

            weights = {
                "queue": -1.0,
                "pressure": -0.8,
                "spillback": -1.2,
                "wait": -0.4,
                "flow": 2.0
            }

        return weights
    

    def _sample_wait(self):

        try:

            current = traci.vehicle.getIDList()
            curset = set(current)

            self.active_vehicles = curset
            self.total_seen.update(curset)

            pick = (
                random.sample(current, min(len(current), MAX_VEHICLES_SAMPLED))
                if len(current) > MAX_VEHICLES_SAMPLED else current
            )

            for v in pick:

                try:
                    w = traci.vehicle.getWaitingTime(v)

                    self.vehicle_wait[v] = w
                    self.wait_samples.append(w)
                    if len(self.wait_samples) > 2000:
                        self.wait_samples.pop(0)

                    if w > 300:
                        self.stuck[v] = self.stuck.get(v, 0) + 1

                except:
                    pass

            # detect completed vehicles
            done_veh = set(self.vehicle_wait.keys()) - curset

            self.completed += len(done_veh)

            for d in done_veh:
                self.vehicle_wait.pop(d, None)
                self.stuck.pop(d, None)

            # -----------------------------
            # DEBUG WAIT METRICS
            # -----------------------------
            if self.step_count % 200 == 0:

                if self.wait_samples:
                    avg_wait = np.mean(self.wait_samples[-50:])
                    max_wait = np.max(self.wait_samples[-50:])
                else:
                    avg_wait = 0
                    max_wait = 0

                print("\n--- WAIT DEBUG ---")
                print("Active vehicles:", len(self.active_vehicles))
                print("Completed vehicles:", self.completed)
                print("Avg sampled wait:", avg_wait)
                print("Max sampled wait:", max_wait)
                print("------------------\n")

        except Exception as e:
            print("Wait sampling error:", e)


    
    def _compute_reward(self):

        # =====================================================
        # 1. Core traffic metrics
        # =====================================================

        avg_queue = np.mean(self.queue_j) / 20.0
        avg_pressure = np.mean(self.pressure_j) / 20.0
        avg_spill = np.mean(self.spill_j) / 10.0

        # =====================================================
        # 2. Waiting time
        # =====================================================

        if self.wait_samples:
            avg_wait = np.mean(self.wait_samples[-50:]) / 50.0
        else:
            avg_wait = 0.0


        # =====================================================
        # 3. Speed drop penalty (slow traffic detection)
        # =====================================================

        speed_drop_penalty = 0

        for lane in self.lane_list:

            try:
                sp = traci.lane.getLastStepMeanSpeed(lane)

                if sp < ETC_SPEED_DROP:
                    speed_drop_penalty += 1

            except:
                pass

        speed_drop_penalty /= max(len(self.lane_list), 1)


        # =====================================================
        # 4. Neighbor pressure (multi-agent coordination)
        # =====================================================

        neigh_penalty = 0

        for j in self.junctions:

            neigh = self.adjacency.get(j, [])

            if neigh:

                neigh_pressure = np.mean([
                    self.pressure_j[self.j2i[n]]
                    for n in neigh
                ])

                neigh_penalty += neigh_pressure

        neigh_penalty /= max(self.num_agents, 1)
        neigh_penalty /= 100.0


        # =====================================================
        # 5. Transition penalty (avoid too frequent switching)
        # =====================================================

        transition_pen = sum(
            1 for j in self.junctions
            if self.phase_manager.is_transitioning(j)
        )

        transition_pen /= max(self.num_agents, 1)


        # =====================================================
        # 6. Throughput reward (IMPORTANT FIX)
        # =====================================================

        new_completed = self.completed - self.prev_completed
        self.prev_completed = self.completed

        flow_bonus = new_completed / DELTA_TIME


        # =====================================================
        # 7. Pressure reduction reward
        # =====================================================

        pressure_diff = np.mean(self.prev_pressure - self.pressure_j)

        pressure_reward = np.clip(pressure_diff / 10.0, -1.0, 1.0)

        self.prev_pressure = self.pressure_j.copy()


        # =====================================================
        # 8. Stuck vehicle penalty
        # =====================================================

        stuck_count = sum(
            1 for v in self.vehicle_wait
            if self.vehicle_wait[v] > 120
        )

        stuck_penalty = stuck_count / 50.0


        # =====================================================
        # 9. Context-aware reward weights
        # =====================================================

        weights = self._get_context_weights()


        cars_reward = (
            weights["queue"] * avg_queue +
            weights["pressure"] * avg_pressure +
            weights["spillback"] * avg_spill +
            weights["wait"] * avg_wait +
            weights["flow"] * flow_bonus
        )


        # =====================================================
        # 10. Final reward combination
        # =====================================================

        reward = cars_reward

        reward += pressure_reward * 0.5

        reward -= 0.3 * speed_drop_penalty
        reward -= 0.2 * neigh_penalty
        reward -= 0.1 * transition_pen
        reward -= 0.2 * stuck_penalty


        # scale reward to stable PPO range
        reward = np.clip(reward, -10, 10) / 5.0

        return float(reward)
# ===================================================================================
# ================================   STEP() FUNCTION   ===============================
# ===================================================================================

    def step(self, actions):

        info = {}

        # Debug
        if self.step_count % 200 == 0:
            print("Sample actions:", actions)

        # ------------------------------------------------
        # 1. Observe state first
        # ------------------------------------------------
        obs = self._get_obs()

        # ------------------------------------------------
        # 2. Pressure override (safety)
        # ------------------------------------------------
        actions = self._pressure_correction(actions)

        # ------------------------------------------------
        # 3. Apply RL actions
        # ------------------------------------------------
        self._apply_actions(actions)

        # ------------------------------------------------
        # 4. Update traffic signals
        # ------------------------------------------------
        for jid in self.junctions:

            idx = self.j2i[jid]
            queue = self.queue_j[idx]

            phase = self.phase_manager.update(
                jid,
                DELTA_TIME,
                self.mode,
                queue
            )

            traci.trafficlight.setPhase(jid, int(phase))

        # ------------------------------------------------
        # 5. Run SUMO simulation
        # ------------------------------------------------
        for _ in range(DELTA_TIME):
            traci.simulationStep()

        self.step_count += 1
        self.phase_time += DELTA_TIME

        # ------------------------------------------------
        # 6. Waiting time sampling
        # ------------------------------------------------
        if self.step_count % SAMPLE_INTERVAL == 0:
            self._sample_wait()

        # ------------------------------------------------
        # 7. New observation
        # ------------------------------------------------
        obs = self._get_obs()

        # ------------------------------------------------
        # 8. Reward
        # ------------------------------------------------
        reward = self._compute_reward()
        self.deque_rewards.append(reward)

        # ------------------------------------------------
        # 9. Episode termination
        # ------------------------------------------------
        sim_time = traci.simulation.getTime()

        terminated = False
        truncated = sim_time >= MAX_SIM_TIME

        if truncated or terminated:

            avg_queue = float(np.mean(self.queue_j))
            avg_pressure = float(np.mean(self.pressure_j))
            avg_spill = float(np.mean(self.spill_j))

            if self.wait_samples:
                avg_wait = float(np.mean(self.wait_samples))
                max_wait = float(np.max(self.wait_samples))
                p95_wait = float(np.percentile(self.wait_samples, 95))
            else:
                avg_wait = 0.0
                max_wait = 0.0
                p95_wait = 0.0

            summary = {
                "episode": self.episode,
                "traffic_mode": self.mode,
                "episode_total_reward": float(sum(self.deque_rewards)),
                "episode_avg_queue": avg_queue,
                "episode_avg_wait_time": avg_wait,
                "episode_avg_pressure": avg_pressure,
                "episode_avg_spillback": avg_spill,
                "total_vehicles": len(self.total_seen),
                "completed_vehicles": self.completed,
                "active_vehicles": len(self.active_vehicles),
                "max_wait_time": max_wait,
                "p95_wait_time": p95_wait,
                "total_steps": self.step_count
            }

            if hasattr(self, "logger"):
                self.logger.log_episode(summary)

            info["episode_summary"] = summary

        return obs, reward, terminated, truncated, info
# ===================================================================================
# ================================   CLOSE()   ======================================
# ===================================================================================

    def close(self):
        try:
            if traci.isLoaded():
                traci.close()
        except:
            pass