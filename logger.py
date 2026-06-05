import os
import pandas as pd
from datetime import datetime


class TrafficLogger:
    def __init__(self, log_dir="logs", experiment_name="PPO_baseline"):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.global_episode = 0

        # ---------------------------------------------------------
        # Fixed experiment folder
        # ---------------------------------------------------------
        self.exp_dir = os.path.join(self.log_dir, self.experiment_name)
        os.makedirs(self.exp_dir, exist_ok=True)

        # Always reuse the same run folder
        self.run_folder = "run_001"
        self.run_dir = os.path.join(self.exp_dir, self.run_folder)
        os.makedirs(self.run_dir, exist_ok=True)

        # CSV file paths
        self.episode_file = os.path.join(self.run_dir, "episodes.csv")
        self.loss_file = os.path.join(self.run_dir, "losses.csv")

        # ---------------------------------------------------------
        # Only create files if they do not already exist
        # ---------------------------------------------------------
        if not os.path.exists(self.episode_file):
            self._init_episode_csv()
        else:
            try:
                existing = pd.read_csv(self.episode_file)
                if len(existing) > 0 and "episode" in existing.columns:
                    self.global_episode = int(existing["episode"].max())
            except Exception:
                self.global_episode = 0

        if not os.path.exists(self.loss_file):
            self._init_loss_csv()

        print(f"[Logger] Experiment: {self.experiment_name}")
        print(f"[Logger] Reusing folder: {self.run_dir}")
        print(f"[Logger] Episode CSV: {self.episode_file}")
        print(f"[Logger] Loss CSV: {self.loss_file}")
        print(f"[Logger] Starting episode count from: {self.global_episode}")

    # =========================================================
    # CSV Initialization
    # =========================================================
    def _init_episode_csv(self):
        headers = [
            "episode",
            "timestamp",
            "traffic_mode",
            "total_reward",
            "avg_queue",
            "avg_wait_time",
            "avg_pressure",
            "avg_spillback",
            "total_steps",
            "total_vehicles",
            "completed_vehicles",
            "active_vehicles",
            "max_wait_time",
            "p95_wait_time"
        ]

        pd.DataFrame(columns=headers).to_csv(
            self.episode_file,
            index=False
        )

    def _init_loss_csv(self):
        headers = [
            "timestamp",
            "timestep",
            "episode",
            "traffic_mode",
            "policy_loss",
            "value_loss",
            "entropy_loss",
            "approx_kl",
            "clip_fraction",
            "learning_rate"
        ]

        pd.DataFrame(columns=headers).to_csv(
            self.loss_file,
            index=False
        )

    # =========================================================
    # Episode Logging
    # =========================================================
    def log_episode(self, metrics):

        def get_val(keys, default=0):
            for key in keys:
                if key in metrics:
                    return metrics[key]
            return default

        active_vehicles_val = metrics.get("active_vehicles", 0)

        if hasattr(active_vehicles_val, "__len__") and not isinstance(
            active_vehicles_val, (int, float)
        ):
            active_vehicles = len(active_vehicles_val)
        else:
            active_vehicles = int(active_vehicles_val)

        self.global_episode += 1

        data = {
            "episode": self.global_episode,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "traffic_mode": str(get_val(["traffic_mode"], "unknown")),
            "total_reward": float(
                get_val(["episode_total_reward", "total_reward"], 0)
            ),
            "avg_queue": float(
                get_val(["episode_avg_queue", "avg_queue"], 0)
            ),
            "avg_wait_time": float(
                get_val(["episode_avg_wait_time", "avg_wait_time"], 0)
            ),
            "avg_pressure": float(
                get_val(["episode_avg_pressure", "avg_pressure"], 0)
            ),
            "avg_spillback": float(
                get_val(["episode_avg_spillback", "avg_spillback"], 0)
            ),
            "total_steps": int(get_val(["total_steps"], 0)),
            "total_vehicles": int(get_val(["total_vehicles"], 0)),
            "completed_vehicles": int(
                get_val(["completed_vehicles"], 0)
            ),
            "active_vehicles": active_vehicles,
            "max_wait_time": float(get_val(["max_wait_time"], 0)),
            "p95_wait_time": float(get_val(["p95_wait_time"], 0))
        }

        pd.DataFrame([data]).to_csv(
            self.episode_file,
            mode="a",
            header=False,
            index=False
        )

        print("\n" + "=" * 60)
        print(f"EPISODE {data['episode']} | Mode: {data['traffic_mode']}")
        print(f"  Total Reward: {data['total_reward']:.2f}")
        print(f"  Avg Queue: {data['avg_queue']:.2f}")
        print(f"  Avg Wait Time: {data['avg_wait_time']:.2f}s")
        print(f"  Max Wait Time: {data['max_wait_time']:.2f}s")
        print(f"  P95 Wait Time: {data['p95_wait_time']:.2f}s")
        print(
            f"  Vehicles: {data['completed_vehicles']}/"
            f"{data['total_vehicles']} completed, "
            f"{data['active_vehicles']} active"
        )
        print(f"  Avg Pressure: {data['avg_pressure']:.2f}")
        print(f"  Avg Spillback: {data['avg_spillback']:.2f}")
        print("=" * 60)

    # =========================================================
    # Loss Logging
    # =========================================================
    def log_losses(
        self,
        timestep,
        episode,
        traffic_mode,
        policy_loss,
        value_loss,
        entropy_loss,
        approx_kl,
        clip_fraction,
        learning_rate
    ):

        data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "timestep": int(timestep),
            "episode": int(episode),
            "traffic_mode": str(traffic_mode),
            "policy_loss": float(policy_loss),
            "value_loss": float(value_loss),
            "entropy_loss": float(entropy_loss),
            "approx_kl": float(approx_kl),
            "clip_fraction": float(clip_fraction),
            "learning_rate": float(learning_rate)
        }

        pd.DataFrame([data]).to_csv(
            self.loss_file,
            mode="a",
            header=False,
            index=False
        )


def create_logger(log_dir="logs", experiment_name="PPO_baseline"):
    return TrafficLogger(log_dir, experiment_name)