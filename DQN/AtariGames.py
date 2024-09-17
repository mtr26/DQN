import gym

# * The game ids

PONG = 0
BREAKOUT = 1


game_index = {
    0 : 'ALE/Pong-v5',
    1 : 'ALE/Breakout-v5'
}


class Game_ATARI:
    def __init__(self, game_id):
        self.env = gym.make(game_index[game_id], render_mode = "rgb_array")
        self.rewards = []
        self.inputs = self.env.action_space.n
        self.first_obs, info = self.env.reset(seed=42)
        
        
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.env.render()
        if terminated or truncated:
            self.first_obs, info = self.env.reset()
        self.rewards.append(reward)
        return observation, reward

    def close(self):
        self.env.close()

    def reset(self):
        self.first_obs, info = self.env.reset()

