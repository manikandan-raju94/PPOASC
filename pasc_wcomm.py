import os
import sys
import random
import numpy as np
import torch
from datetime import datetime
import time
import gymnasium as gym
import glob

# =============================================================
# PATH SETUP
# =============================================================
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

NET_FILE = os.path.join(ROOT, "Network", "network_signalized.net.xml")
SUMO_CFG = os.path.join(ROOT, "runsimulation.sumocfg")

print("ROOT =", ROOT)
print("NET_FILE =", NET_FILE)
print("SUMO_CFG =", SUMO_CFG)

from Demand.traffic_generator_dynamic import generate_traffic
from traffic_env_wcomm import GridMarlEnv
from logger import TrafficLogger, create_logger

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# =============================================================
# GLOBAL SEED
# =============================================================
SEED = 42
DDS_MODES = [
    "light",
    "medium",
    "heavy"
]
# Global logger instance (will be created once)
GLOBAL_LOGGER = None


# =============================================================
# SCENARIO STATS
# =============================================================
class ScenarioStats:
    def __init__(self):
       
        self.total = 0
        DDS_MODES = [
            "light",
            "medium",
            "heavy"
        ]

        self.counts = {m: 0 for m in DDS_MODES}
        self.rewards = {m: [] for m in DDS_MODES}
        self.queues = {m: [] for m in DDS_MODES}

    def add_scenario(self, mode, reward=None, queue=None):
        if mode in self.counts:
            self.counts[mode] += 1
            self.total += 1
            if reward is not None:
                self.rewards[mode].append(reward)
            if queue is not None:
                self.queues[mode].append(queue)

    def print_stats(self):
        if self.total == 0:
            return
        print("\n========== SCENARIO DISTRIBUTION ==========")
        for mode in DDS_MODES:
            pct = (self.counts[mode] / self.total) * 100
            reward_str = ""
            if self.rewards[mode]:
                avg_reward = np.mean(self.rewards[mode][-10:])
                reward_str = f", Avg Reward: {avg_reward:8.2f}"
            print(f"{mode.capitalize():7}: {self.counts[mode]:3d} ({pct:4.1f}%){reward_str}")
        print("===========================================\n")


scenario_stats = ScenarioStats()


# =============================================================
# MODE BALANCER
# =============================================================
class ModeBalancer:
    def __init__(self):
        self.counts = {m: 0 for m in DDS_MODES}
        self.performance = {m: [] for m in DDS_MODES}
        self.total = 0
        self.last_mode = None
        
    def update_performance(self, mode, reward):
        self.performance[mode].append(reward)
        if len(self.performance[mode]) > 20:
            self.performance[mode].pop(0)
    
    def get_next_mode(self):
        self.total += 1
        
        # First 3 episodes: ensure each mode once
        if self.total <= 3:
            unseen = [m for m in DDS_MODES if self.counts[m] == 0]
            if unseen:
                selected = random.choice(unseen)
                self.counts[selected] += 1
                self.last_mode = selected
                print(f"[Balancer] First episodes - forcing {selected}")
                return selected
        
        # Check for imbalance
        max_count = max(self.counts.values())
        min_count = min(self.counts.values())
        
        if max_count - min_count >= 2:
            candidates = [m for m, c in self.counts.items() if c == min_count]
            selected = random.choice(candidates)
            self.counts[selected] += 1
            self.last_mode = selected
            print(f"[Balancer] Imbalance - forcing {selected}")
            return selected
        
        # Weighted random selection
        weights = []
        for mode in DDS_MODES:
            count_weight = 1.0 / (self.counts[mode] + 1)
            
            if len(self.performance[mode]) >= 3:
                recent_avg = np.mean(self.performance[mode][-3:])
                # Normalize based on typical reward ranges
                if mode == "light":
                    norm_reward = np.clip((recent_avg + 2000) / 2000, 0.5, 1.5)
                elif mode == "medium":
                    norm_reward = np.clip((recent_avg + 1000) / 1500, 0.5, 1.5)
                else:  # heavy
                    norm_reward = np.clip((recent_avg + 2000) / 2000, 0.5, 1.5)
                
                perf_weight = 2.0 - norm_reward
                perf_weight = np.clip(perf_weight, 0.5, 1.5)
            else:
                perf_weight = 1.0
            
            weights.append(count_weight * perf_weight)
        
        total_weight = sum(weights)
        weights = [w/total_weight for w in weights]
        
        selected = random.choices(DDS_MODES, weights=weights)[0]
        self.counts[selected] += 1
        return selected
    
    def print_stats(self):
        if self.total == 0:
            return
        print("\n========== MODE SELECTION DISTRIBUTION ==========")
        for mode in DDS_MODES:
            pct = (self.counts[mode] / self.total) * 100
            bar_length = int(pct / 2)
            bar = "█" * bar_length
            perf_str = ""
            if self.performance[mode]:
                avg_perf = np.mean(self.performance[mode][-5:])
                perf_str = f", Recent Avg: {avg_perf:8.2f}"
            print(f"{mode.capitalize():7}: {self.counts[mode]:3d} ({pct:5.1f}%) {bar}{perf_str}")
        print("================================================\n")


mode_balancer = ModeBalancer()


# =============================================================
# TRAFFIC FILE CACHE
# =============================================================
class TrafficFileCache:
    def __init__(self):
        self.files = {}
        self.generated = False
        
    def pre_generate_all(self):
        if not self.generated:
            print("\n=== Pre-generating all traffic modes ===\n")
            for mode in DDS_MODES:
                self.get_traffic_file(mode, force_generate=True)
            print("\n=== All traffic modes cached ===\n")
            self.generated = True
        
    def get_traffic_file(self, mode, force_generate=False):
        if mode not in self.files or force_generate:
            print(f"\n=== GENERATING {mode.upper()} TRAFFIC ===")
            veh, route_cnt, xml_file, traffic_mode = generate_traffic(mode)
            self.files[mode] = {
                'file': xml_file,
                'mode': traffic_mode,
                'vehicles': veh,
                'routes': route_cnt
            }
            print(f"Cached: {xml_file} ({veh} vehicles)")
        else:
            cached = self.files[mode]
            print(f"Using cached {mode}: {cached['file']} ({cached['vehicles']} vehicles)")
        
        return self.files[mode]['file'], self.files[mode]['mode']


traffic_cache = TrafficFileCache()


def pick_scenario_balanced():
    selected_mode = mode_balancer.get_next_mode()
    xml_file, traffic_mode = traffic_cache.get_traffic_file(selected_mode)
    return xml_file, traffic_mode


# =============================================================
# UPDATE SUMO CONFIG
# =============================================================
def update_sumo_cfg(route_file):
    try:
        with open(SUMO_CFG, "r") as f:
            lines = f.readlines()
        
        with open(SUMO_CFG, "w") as f:
            for line in lines:
                if "<route-files" in line:
                    f.write(f'        <route-files value="{route_file}"/>\n')
                else:
                    f.write(line)
        print(f"Updated SUMO config with: {route_file}")
    except Exception as e:
        print(f"SUMO config update failed: {e}")


# =============================================================
# ENV FACTORY - WITH GLOBAL LOGGER
# =============================================================
def create_env(experiment_name, mode=None, loss_callback=None):
    global GLOBAL_LOGGER
    
    dynamic_seed = int(time.time() * 1000) % 10000 + random.randint(0, 1000)
    
    if mode is None:
        route_file, actual_mode = pick_scenario_balanced()
    else:
        route_file, actual_mode = traffic_cache.get_traffic_file(mode)

    # update_sumo_cfg(route_file)

    env = GridMarlEnv(NET_FILE, SUMO_CFG, seed=dynamic_seed, pre_generated_route_file=route_file)
    env.set_mode(actual_mode)
    
    # Create logger only ONCE for the entire training run
    if GLOBAL_LOGGER is None:
        GLOBAL_LOGGER = create_logger("logs", experiment_name)
        print(f"[Global Logger] Created for experiment: {experiment_name}")
    
    # Use the same global logger for all environments
    env.set_logger(GLOBAL_LOGGER)

    if loss_callback:
        loss_callback.update_logger(GLOBAL_LOGGER)

    print(f"Created environment: {actual_mode}")
    return env, actual_mode



# =============================================================
# LOSS CALLBACK
# =============================================================
class PPOLossCallback(BaseCallback):
    def __init__(self, logger_obj=None, env_wrapper=None):
        super().__init__()
        global GLOBAL_LOGGER
        self.logger_obj = logger_obj or GLOBAL_LOGGER
        self.env_wrapper = env_wrapper
        self.last_log = 0
        self.log_interval = None
        self.episode_count = 0

    def update_logger(self, new_logger):
        self.logger_obj = new_logger

    def update_env_wrapper(self, env_wrapper):
        self.env_wrapper = env_wrapper

    def _init_callback(self):
        self.log_interval = self.model.n_steps
        print(f"Log interval: {self.log_interval}")

    def _on_step(self):
        if self.log_interval is None:
            self._init_callback()

        # Track episode completions
        if self.env_wrapper and hasattr(self.env_wrapper, 'current_info'):
            info = self.env_wrapper.current_info
            if info and "episode_summary" in info:
                self.episode_count += 1
                summary = info["episode_summary"]
                mode = summary["traffic_mode"]
                reward = summary["episode_total_reward"]
                queue = summary["episode_avg_queue"]
                
                mode_balancer.update_performance(mode, reward)
                scenario_stats.add_scenario(mode, reward, queue)

        # Log training losses
        if self.logger_obj is None:
            return True

        if self.num_timesteps - self.last_log < self.log_interval:
            return True
        def _on_rollout_end(self):

            if self.logger_obj is None:
                return
    
        try:
            # Get logs from SB3
            logs = self.model.logger.name_to_value
            
            # Debug: Print available keys occasionally
            if self.num_timesteps == self.log_interval:
                print("\n[DEBUG] Available log keys:")
                for key in logs.keys():
                    print(f"  {key}: {logs[key]}")
            
            current_mode = self.env_wrapper.current_mode if self.env_wrapper else "unknown"
            
            # FIXED: Policy loss might be under different names
            # Try multiple possible keys
            policy_loss = 0.0
            possible_policy_keys = [
                "train/policy_loss",
                "policy_loss",
                "loss/policy_loss",
                "train/policy_gradient_loss",
                "policy_gradient_loss"
            ]
            
            for key in possible_policy_keys:
                if key in logs:
                    policy_loss = logs[key]
                    break
            
            # If still zero, try to find any key containing 'policy'
            if policy_loss == 0.0:
                for key in logs.keys():
                    if 'policy' in key.lower():
                        policy_loss = logs[key]
                        print(f"[DEBUG] Found policy loss at: {key} = {policy_loss}")
                        break
            
            # Get other losses (these keys are usually correct)
            value_loss = logs.get("train/value_loss", 0)
            entropy_loss = logs.get("train/entropy_loss", 0)
            approx_kl = logs.get("train/approx_kl", 0)
            clip_fraction = logs.get("train/clip_fraction", 0)
            learning_rate = logs.get("train/learning_rate", 0)
            
            # Log to CSV
            self.logger_obj.log_losses(
                timestep=self.num_timesteps,
                episode=self.episode_count,
                traffic_mode=current_mode,
                policy_loss=policy_loss,
                value_loss=value_loss,
                entropy_loss=entropy_loss,
                approx_kl=approx_kl,
                clip_fraction=clip_fraction,
                learning_rate=learning_rate,
            )
            
            # Print occasionally
            if self.num_timesteps % 5000 == 0:
                print(f"[LOSS] Step {self.num_timesteps}: P={policy_loss:.4f}, V={value_loss:.2f}, KL={approx_kl:.4f}")

        except Exception as e:
            print(f"[Warning] Loss log error: {e}")

        self.last_log = self.num_timesteps
        return True


# =============================================================
# ENV WRAPPER - FIXED WITH PROPER EPISODE COUNTING
# =============================================================
class EnvWrapper(gym.Env):
    def __init__(self, experiment_name, loss_callback=None):
        super().__init__()
        self.exp_name = experiment_name
        self.loss_callback = loss_callback
        
        # Create first environment
        env_result = create_env(experiment_name, mode=None, loss_callback=loss_callback)
        self.current_env = env_result[0]
        self._current_mode = env_result[1]
        
        # Track episodes at wrapper level
        self._wrapper_episode = 0
        
        # Store rewards and queues with episode numbers
        self.episode_rewards = {m: [] for m in DDS_MODES}
        self.episode_queues = {m: [] for m in DDS_MODES}
        
        self.current_obs, self.current_info = self.current_env.reset()
        self.action_space = self.current_env.action_space
        self.observation_space = self.current_env.observation_space

    @property
    def episode(self):
        return self._wrapper_episode

    @property
    def current_mode(self):
        return self._current_mode

    def reset(self, **kwargs):
        self.current_obs, self.current_info = self.current_env.reset()
        return self.current_obs, self.current_info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.current_env.step(action)
        
        self.current_obs = obs
        self.current_info = info

        if terminated or truncated:
            # Increment wrapper episode counter when episode ends
            self._wrapper_episode += 1
            final_reward = reward
            final_info = info
            
            # Initialize summary to None
            summary = None
            
            if "episode_summary" in final_info:
                summary = final_info["episode_summary"]

                summary["episode"] = self._wrapper_episode

                reward = summary["episode_total_reward"]
                queue = summary["episode_avg_queue"]

                self.episode_rewards[self._current_mode].append({
                    "wrapper_episode": self._wrapper_episode,
                    "env_episode": summary["episode"],
                    "reward": reward
                })

                self.episode_queues[self._current_mode].append({
                    "wrapper_episode": self._wrapper_episode,
                    "queue": queue
                })

            # Safe print - check if summary exists
            env_episode = summary.get('episode', '?') if summary else '?'
            print(f"\n--- Episode {self._wrapper_episode} (Env Episode {env_episode}) finished ---")
            print(f"Mode: {self._current_mode}, Reward: {final_reward:.2f}")

            self.current_env.close()
            
            # Create new environment
            env_result = create_env(self.exp_name, mode=None, loss_callback=self.loss_callback)
            self.current_env = env_result[0]
            self._current_mode = env_result[1]
            
            self.action_space = self.current_env.action_space
            self.observation_space = self.current_env.observation_space
            self.current_obs, self.current_info = self.current_env.reset()
            
            return self.current_obs, final_reward, False, False, final_info

        return self.current_obs, reward, terminated, truncated, info

    def close(self):
        self.current_env.close()


# =============================================================
# MAIN TRAIN LOOP
# =============================================================
def main():
    global GLOBAL_LOGGER
    
    experiment_name = "PPO_MARL_20260421_144607"

    print("\n================ TRAINING START ================")
    print("Experiment :", experiment_name)
    print("Seed       :", SEED)
    print("Total timesteps : 500,000")
    print("================================================\n")

    # Pre-generate all traffic modes
    traffic_cache.pre_generate_all()

    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path=os.path.join(ROOT, "checkpoints"),
        name_prefix="ppo_checkpoint"
    )

    # Create loss callback first
    loss_callback = PPOLossCallback()

    # Create environment wrapper
    env_wrapper = EnvWrapper(experiment_name, loss_callback)
    
    # Update loss callback with environment wrapper reference
    loss_callback.update_env_wrapper(env_wrapper)
    
    # Store reference before Monitor wrapping
    original_wrapper = env_wrapper

    # Wrap for SB3
    env_wrapper = Monitor(env_wrapper)
    env = DummyVecEnv([lambda: env_wrapper])

    # =========================================================
    # PPO - OPTIMIZED CONFIGURATION
    # =========================================================
    checkpoint_dir = os.path.join(ROOT, "checkpoints")

    checkpoint_files = glob.glob(
        os.path.join(checkpoint_dir, "ppo_checkpoint_*_steps.zip")
    )

    if checkpoint_files:
        latest_checkpoint = max(
            checkpoint_files,
            key=os.path.getctime
        )

        print(f"Loading latest checkpoint: {latest_checkpoint}")
        model = PPO.load(latest_checkpoint, env=env)
        print("Current PPO timesteps:", model.num_timesteps)
    else:
        print("No checkpoint found. Creating new PPO model")
        model = PPO(
            "MlpPolicy",
            env,

            learning_rate=lambda f: 3e-4 * f,
            n_steps=2048,
            batch_size=512,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.98,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.25,
            max_grad_norm=0.5,
            target_kl=0.02,
            policy_kwargs=dict(
                net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
                activation_fn=torch.nn.Tanh,
            ),
            verbose=1,
            seed=SEED,
            tensorboard_log=os.path.join("tb_logs", experiment_name),
        )
    t0 = time.time()
    try:
        model.learn(
            total_timesteps=500000,
            callback=[checkpoint_callback, loss_callback],
            reset_num_timesteps=False,
            tb_log_name=experiment_name
        )

        # Save final model
        save_path = os.path.join(ROOT, "models", f"{experiment_name}_wcomm.zip")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        print("\nMODEL SAVED:", save_path)
        
        
       # ------------------------------------------------
        # Save DDS mode-specific models
        # ------------------------------------------------
      

    except Exception as e:
        print("\nTRAINING ERROR:", e)

        # 🔥 SAVE MODEL EVEN IF CRASH HAPPENS
        try:
            crash_path = os.path.join(ROOT, "models", f"{experiment_name}_wcomm.zip")
            os.makedirs(os.path.dirname(crash_path), exist_ok=True)
            model.save(crash_path)
            print("Emergency model saved at:", crash_path)
        except Exception as save_error:
            print("Failed to save emergency model:", save_error)

        import traceback
        traceback.print_exc()

    finally:
        env.close()
        total = time.time() - t0

        print("\n================ TRAINING COMPLETE ================")
        print(f"Time taken: {total:.2f}s ({total/3600:.2f} hours)")
        
        if GLOBAL_LOGGER:
            print(f"\nLog files created:")
            print(f"  Episode metrics: {GLOBAL_LOGGER.episode_file}")
            print(f"  Loss metrics: {GLOBAL_LOGGER.loss_file}")
        
        scenario_stats.print_stats()
        mode_balancer.print_stats()
        
        print("\n========== FINAL PERFORMANCE SUMMARY ==========")
        for mode in DDS_MODES:
            if mode in original_wrapper.episode_rewards and original_wrapper.episode_rewards[mode]:
                # Extract rewards and queues
                rewards = [item["reward"] for item in original_wrapper.episode_rewards[mode][-10:]]
                queues = [item["queue"] for item in original_wrapper.episode_queues[mode][-10:]]
                
                if rewards:
                    avg_reward = np.mean(rewards)
                    avg_queue = np.mean(queues)
                    print(f"{mode.capitalize():7}: Reward={avg_reward:8.2f}, Queue={avg_queue:5.1f} (n={len(original_wrapper.episode_rewards[mode])})")
            else:
                print(f"{mode.capitalize():7}: No episodes yet")
        print("================================================\n")


if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    main()


