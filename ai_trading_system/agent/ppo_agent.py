import os
from stable_baselines3 import PPO
from loguru import logger
import pandas as pd
from agent.trading_env import TradingEnv
from config import settings

class PPOAgent:
    """
    Phase 6: PPO Agent (stable-baselines3) Wrapper
    """
    def __init__(self, data: pd.DataFrame, model_dir=os.path.join(settings.BASE_DIR, "models", "saved")):
        self.data = data
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.env = TradingEnv(data)
        self.model = None

    def create_model(self):
        """Initialize a new PPO model."""
        logger.info("Initializing PPO Agent...")
        # device="cuda" allows GPU training
        self.model = PPO("MlpPolicy", self.env, verbose=1, device="cuda")

    def train(self, timesteps=100000):
        """Train the agent for specified timesteps."""
        if not self.model:
            self.create_model()
        logger.info(f"Training PPO for {timesteps} timesteps...")
        self.model.learn(total_timesteps=timesteps)
        logger.info("PPO Training complete.")

    def predict(self, obs):
        """Predict action for a given observation."""
        if not self.model:
            logger.error("Model not loaded or created.")
            return None
        action, _states = self.model.predict(obs, deterministic=True)
        return action

    def save(self, name="ppo_agent.zip"):
        path = os.path.join(self.model_dir, name)
        if self.model:
            self.model.save(path)
            logger.info(f"PPO Agent saved to {path}")

    def load(self, name="ppo_agent.zip"):
        path = os.path.join(self.model_dir, name)
        if os.path.exists(path):
            self.model = PPO.load(path, env=self.env, device="cuda")
            logger.info(f"PPO Agent loaded from {path}")
            return True
        else:
            logger.error(f"PPO Model not found at {path}")
            return False
