from pogema import GridConfig, pogema_v0


def test_grid_config_defaults_to_one_agent_and_infers_custom_agents():
    assert GridConfig().num_agents == 1
    assert GridConfig(
        agents_xy=[(0, 0), (1, 1)],
        targets_xy=[(0, 1), (1, 2)],
    ).num_agents == 2


def test_random_agv_environment_resets_and_steps_with_four_actions():
    env = pogema_v0(GridConfig(num_agents=2, size=8, seed=1))

    observations, infos = env.reset()
    actions = env.sample_actions()
    observations, rewards, terminated, truncated, infos = env.step(actions)

    assert env.action_space.n == 4
    assert len(observations) == 2
    assert len(rewards) == 2
    assert len(terminated) == 2
    assert len(truncated) == 2
    assert all(0 <= int(action) < 4 for action in actions)


def test_gymnasium_single_agent_wrapper_supports_reset_seed_and_step():
    env = pogema_v0(GridConfig(
        integration="gymnasium",
        num_agents=2,
        size=4,
        density=0,
        agents_xy=[(1, 1), (2, 1)],
        targets_xy=[(1, 3), (2, 3)],
    ))

    observation, info = env.reset(seed=123)
    observation, reward, terminated, truncated, info = env.step(3)

    assert env.action_space.n == 4
    assert observation.shape == (3, 11, 11)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_pettingzoo_parallel_reset_returns_observations_and_infos():
    env = pogema_v0(GridConfig(
        integration="PettingZoo",
        num_agents=2,
        size=4,
        density=0,
        agents_xy=[(1, 1), (2, 1)],
        targets_xy=[(1, 3), (2, 3)],
    ))

    observations, infos = env.reset(seed=123)

    assert sorted(observations) == ["player_0", "player_1"]
    assert sorted(infos) == ["player_0", "player_1"]
    assert observations["player_0"].shape == (3, 11, 11)
    assert isinstance(infos["player_0"], dict)
