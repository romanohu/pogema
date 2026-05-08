import functools

import numpy as np
from pogema import GridConfig
from pogema.envs import _make_pogema


def parallel_env(grid_config: GridConfig = GridConfig()):
    return PogemaParallel(grid_config)


class PogemaParallel:
    @staticmethod
    def _format_observation(observation):
        if hasattr(observation, "astype"):
            return observation.astype(np.float32)
        return observation

    def state(self):
        return self.pogema.get_state()

    def __init__(self, grid_config: GridConfig, render_mode='ansi'):
        self.metadata = {'render_modes': ['ansi'], "name": "pogema"}
        self.render_mode = render_mode
        self.pogema = _make_pogema(grid_config)
        self.possible_agents = ["player_" + str(r) for r in range(self.pogema.get_num_agents())]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.agents = None
        self.num_moves = None

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        assert agent in self.possible_agents
        return self.pogema.observation_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        assert agent in self.possible_agents
        return self.pogema.action_space

    def render(self, mode="human"):
        assert mode == 'human'
        return self.pogema.render()

    def reset(self, seed=None, options=None):
        observations, info = self.pogema.reset(seed=seed, options=options)
        self.agents = self.possible_agents[:]
        self.num_moves = 0
        observations = {
            agent: self._format_observation(observations[self.agent_name_mapping[agent]])
            for agent in self.agents
        }
        infos = {agent: info[self.agent_name_mapping[agent]] for agent in self.agents}
        return observations, infos

    def step(self, actions):
        anm = self.agent_name_mapping

        wait_action = self.pogema.grid_config.get_wait_action()
        actions = [actions[agent] if agent in actions else wait_action for agent in self.possible_agents]
        observations, rewards, terminated, truncated, infos = self.pogema.step(actions)
        d_observations = {agent: self._format_observation(observations[anm[agent]]) for agent in
                          self.agents}
        d_rewards = {agent: rewards[anm[agent]] for agent in self.agents}
        d_terminated = {agent: terminated[anm[agent]] for agent in self.agents}
        d_truncated = {agent: truncated[anm[agent]] for agent in self.agents}
        d_infos = {agent: infos[anm[agent]] for agent in self.agents}

        for agent, idx in anm.items():
            if (not self.pogema.grid.is_active[idx] or all(truncated) or all(terminated)) and agent in self.agents:
                self.agents.remove(agent)

        return d_observations, d_rewards, d_terminated, d_truncated, d_infos

    @property
    def unwrapped(self):
        return self

    def close(self):
        pass
