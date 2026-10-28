"""Run one placement scenario across users and days with isolated feedback memory."""

import gzip
import json
import random
import re
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from habitat_llm.personalization.core import FeedbackMemory, PROFILES, placement_pair

# Register Habitat schemas and the file_stem resolver before Hydra composes config.
from habitat_llm.utils import fix_config, setup_config
from habitat.config.default_structured_configs import (
    HabitatConfigPlugin,
    register_hydra_plugin,
)
from habitat_baselines.config.default_structured_configs import (
    HabitatBaselinesConfigPlugin,
)

# Import custom action ConfigStore entries before Hydra resolves the defaults.
# In particular, actions.py registers humanoid_base_velocity at import time.
from habitat_llm.agent.env import (
    EnvironmentInterface,
    register_actions,
    register_measures,
    register_sensors,
    remove_visual_sensors,
)

register_hydra_plugin(HabitatConfigPlugin)
register_hydra_plugin(HabitatBaselinesConfigPlugin)


@hydra.main(
    version_base="1.3",
    config_path="../conf",
    config_name="baselines/personalized_takeover",
)
def main(config):
    # Keep inspection usable before simulator initialization / model loading.
    cfg = config.personalization
    data_path = Path(
        config.habitat.dataset.data_path.format(split=config.habitat.dataset.split)
    )
    opener = gzip.open if data_path.suffix == ".gz" else open
    with opener(data_path, "rt") as stream:
        raw = json.load(stream)
    candidates = []
    for episode in raw["episodes"]:
        try:
            pairs = placement_pair(episode)
        except ValueError:
            continue
        candidates.append((episode, pairs))
    if cfg.inspect_only:
        print(
            json.dumps(
                [
                    {
                        "episode_id": ep["episode_id"],
                        "instruction": ep["instruction"],
                        "placements": pairs,
                    }
                    for ep, pairs in candidates
                ],
                indent=2,
            )
        )
        return
    if cfg.episode_id is not None:
        candidates = [
            (ep, pairs)
            for ep, pairs in candidates
            if str(ep["episode_id"]) == str(cfg.episode_id)
        ]
    if not candidates:
        raise ValueError(
            "No compatible episode. Use personalization.inspect_only=true to list candidates."
        )
    episode, pairs = candidates[0]
    if cfg.first_object_handle is not None:
        if cfg.first_object_handle not in [pair[0] for pair in pairs]:
            raise ValueError(
                "first_object_handle is not one of the selected goal objects"
            )
        pairs.sort(key=lambda pair: pair[0] != cfg.first_object_handle)
    conditions = list(cfg.conditions)
    if (
        not conditions
        or len(set(conditions)) != len(conditions)
        or set(conditions) - {"no_memory", "learned_memory", "oracle_profile"}
    ):
        raise ValueError("conditions must be distinct known conditions")
    users = list(cfg.users)
    if not users or len({u.id for u in users}) != len(users):
        raise ValueError("Need nonempty, unique user IDs")
    for user in users:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", user.id) or user.profile not in PROFILES:
            raise ValueError("Invalid user ID or profile")
    if cfg.days < 1 or cfg.pause_steps < 1:
        raise ValueError("days and pause_steps must be positive")
    if config.world_model.partial_obs:
        raise ValueError("This controlled pilot requires full observation")
    if (
        config.num_proc != 1
        or config.robot_agent_uid != 0
        or config.human_agent_uid != 1
    ):
        raise ValueError("This pilot requires num_proc=1, robot=0, human=1")

    from habitat_llm.agent.env.dataset import CollaborationDatasetV0
    from habitat_llm.evaluation.takeover_evaluation_runner import (
        TakeoverEvaluationRunner,
    )
    from habitat_llm.llm import instantiate_llm

    fix_config(config)
    config = setup_config(config, int(cfg.seed))
    dataset = CollaborationDatasetV0(config.habitat.dataset)
    selected = [
        ep
        for ep in dataset.episodes
        if str(ep.episode_id) == str(episode["episode_id"])
    ]
    dataset = CollaborationDatasetV0(config.habitat.dataset, episodes=selected)
    if not config.evaluation.save_video:
        remove_visual_sensors(config)
    register_sensors(config)
    register_actions(config)
    register_measures(config)
    out = Path(HydraConfig.get().runtime.output_dir) / "personalization"
    out.mkdir(parents=True, exist_ok=False)  # Never silently contaminate a rerun.
    OmegaConf.save(config, out / "config.yaml")
    (out / "scenario.json").write_text(
        json.dumps(
            {
                "episode_id": episode["episode_id"],
                "instruction": episode["instruction"],
                "placements": pairs,
                "observation": "full",
                "simulated_feedback": True,
            },
            indent=2,
        )
    )
    llm = instantiate_llm(
        str(cfg.llm_name), generation_params={"max_tokens": int(cfg.max_tokens)}
    )
    env = EnvironmentInterface(config, dataset=dataset, init_wg=False)
    try:
        env.initialize_perception_and_world_graph()
        runner = TakeoverEvaluationRunner(config.evaluation, env)
        for condition in conditions:
            for user in users:
                memory = FeedbackMemory(out / condition / user.id / "memory.json")
                for day in range(1, int(cfg.days) + 1):
                    # Paired initial conditions across users, days and baselines.
                    random.seed(int(cfg.seed))
                    np.random.seed(int(cfg.seed))
                    torch.manual_seed(int(cfg.seed))
                    env.env.env.env._env.seed(int(cfg.seed))
                    env.reset_environment(episode_id=str(episode["episode_id"]))
                    runner.reset()
                    graph = env.full_world_graph
                    first = graph.get_node_from_sim_handle(pairs[0][0]).name
                    remaining = graph.get_node_from_sim_handle(pairs[1][0]).name
                    target = graph.get_node_from_sim_handle(pairs[0][1]).name
                    for handle, _ in pairs:
                        node = graph.get_node_from_sim_handle(handle)
                        furniture = graph.find_furniture_for_object(node)
                        if furniture is not None and furniture.name == target:
                            raise ValueError(
                                "Selected goal already satisfied at reset; choose another episode"
                            )
                    history = "No previous interaction feedback is available."
                    if condition == "learned_memory":
                        history = memory.context()
                    elif condition == "oracle_profile":
                        history = "Known preference: " + PROFILES[user.profile]["WAIT"]
                    runner.configure_trial(
                        (first, remaining, target), history, llm, int(cfg.pause_steps)
                    )
                    name = f"{condition}_{user.id}_day_{day}"
                    row = dict(
                        condition=condition,
                        user_id=user.id,
                        day=day,
                        episode_id=str(episode["episode_id"]),
                    )
                    try:
                        info = runner.run_instruction(output_name=name)
                        row.update(runner.protocol.result(user.profile))
                        row.update(
                            {
                                key: info.get(key)
                                for key in (
                                    "task_state_success",
                                    "task_percent_complete",
                                    "runtime",
                                )
                            }
                        )
                        row["completed"] = row["phase"] == "done" and bool(
                            row["task_state_success"]
                        )
                        # Do not learn preference feedback from failed physical rollouts.
                        if row["completed"]:
                            memory.record(day, row["decision"], row["feedback"])
                    except Exception as exc:
                        row.update(
                            completed=False, error=f"{type(exc).__name__}: {exc}"
                        )
                        # Preserve failed trials and abort instead of silently reporting successes only.
                        with (out / "results.jsonl").open("a") as stream:
                            stream.write(json.dumps(row) + "\n")
                        raise
                    row["decision_prompt"] = runner.decision_prompt
                    row["decision_response"] = runner.decision_response
                    with (out / "results.jsonl").open("a") as stream:
                        stream.write(json.dumps(row) + "\n")
                    print(
                        json.dumps(
                            {
                                k: v
                                for k, v in row.items()
                                if k not in ("decision_prompt", "decision_response")
                            }
                        )
                    )
    finally:
        env.env.close()


if __name__ == "__main__":
    main()
