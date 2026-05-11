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


def test_default_action_scheme_uses_five_cardinal_actions():
    env = pogema_v0(
        GridConfig(
            action_scheme="default",
            num_agents=1,
            observation_type="MAPF",
            on_target="nothing",
            agents_xy=[(1, 1)],
            targets_xy=[(1, 2)],
            map="....\n....\n....\n....",
            density=0.0,
            obs_radius=1,
        )
    )

    observations, _infos = env.reset()
    core_env = env.unwrapped

    assert env.action_space.n == 5
    assert "heading" not in observations[0]
    assert "global_heading" not in observations[0]

    env.step([1])
    assert core_env.get_agents_xy(ignore_borders=True)[0] == [0, 1]


def test_oriented_action_scheme_stays_four_actions_with_heading():
    env = pogema_v0(
        GridConfig(
            action_scheme="oriented_v1",
            num_agents=1,
            observation_type="MAPF",
            on_target="nothing",
            initial_headings=[0],
            agents_xy=[(1, 1)],
            targets_xy=[(1, 2)],
            map="....\n....\n....\n....",
            density=0.0,
            obs_radius=1,
        )
    )

    observations, _infos = env.reset()

    assert env.action_space.n == 4
    assert observations[0]["heading"] == 0
    assert observations[0]["global_heading"] == 0


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
