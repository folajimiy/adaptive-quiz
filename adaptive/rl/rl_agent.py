class RLAgent:
    def __init__(self):
        self.base_rewards = {
            "remember": 0.1,
            "understand": 0.2,
            "apply": 0.3,
            "analyze": 0.4,
            "evaluate": 0.5,
            "create": 0.6
        }

    def get_reward(self, bloom_level, correct):
        base = self.base_rewards.get(bloom_level, 0.3)
        return base + 0.5 if correct else base - 0.2
