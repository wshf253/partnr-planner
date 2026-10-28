"""Collect frozen support interactions, then diagnose partner-specific failures."""

import gzip
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import hydra
import numpy as np
import torch
from habitat.config.default_structured_configs import (
    HabitatConfigPlugin,
    register_hydra_plugin,
)
from habitat_baselines.config.default_structured_configs import (
    HabitatBaselinesConfigPlugin,
)
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

# Custom ConfigStore entries must exist BEFORE Hydra composes defaults.
from habitat_llm.agent.env import (
    EnvironmentInterface,
    register_actions,
    register_measures,
    register_sensors,
    remove_visual_sensors,
)
from habitat_llm.agent.env.dataset import CollaborationDatasetV0
from habitat_llm.coordination.core import (
    CONDITIONS,
    PROFILE_DESCRIPTIONS,
    Goal,
    choose_split,
    extract_goals,
    frozen_context,
    layout_fingerprint,
)
from habitat_llm.evaluation.coordination_evaluation_runner import (
    CoordinationEvaluationRunner,
)
from habitat_llm.utils import fix_config, setup_config

register_hydra_plugin(HabitatConfigPlugin)
register_hydra_plugin(HabitatBaselinesConfigPlugin)


def read_candidates(config):
    cfg = config.coordination
    path = Path(
        config.habitat.dataset.data_path.format(split=config.habitat.dataset.split)
    )
    with (gzip.open if path.suffix == ".gz" else open)(path, "rt") as stream:
        episodes = json.load(stream)["episodes"]
    valid = []
    for ep in episodes:
        try:
            extract_goals(ep, cfg.min_objects, cfg.min_groups)
            layout_fingerprint(ep)
        except ValueError:
            continue
        valid.append(ep)
    return valid


def run_trial(runner, env, raw, profile, context, cfg, seed, directory):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.env.env.env._env.seed(seed)
    env.reset_environment(episode_id=str(raw["episode_id"]))
    runner.reset()
    graph = env.full_world_graph
    handles = extract_goals(raw, cfg.min_objects, cfg.min_groups)
    goals = [
        Goal(
            graph.get_node_from_sim_handle(g.object).name,
            graph.get_node_from_sim_handle(g.target).name,
        )
        for g in handles
    ]
    # The diagnostic has shared goals, not corpus-specific textual role assignments.
    instruction = (
        "Together with your partner, complete all of these placements: "
        + "; ".join(f"put {g.object} on {g.target}" for g in goals)
        + ". Either agent may perform any placement. Coordinate efficiently."
    )
    runner.configure_trial(goals, profile, context, cfg, directory)
    # Preserve enough unfinished work for the hypothesized coordination conflict.
    pending = [g for g in goals if not runner.scene["objects"][g.object]["at_goal"]]
    groups = Counter(g.target for g in pending)
    if len(pending) < int(cfg.min_objects) or sum(
        n >= 2 for n in groups.values()
    ) < int(cfg.min_groups):
        raise ValueError(
            "Too many goals already satisfied at reset; select a different episode"
        )
    try:
        info = runner.run_instruction(
            instruction=instruction, output_name=Path(directory).name
        )
    except Exception:
        # Keep the actual prompt and observed partial trajectory for diagnosis.
        runner.finish_trial(
            instruction, raw["episode_id"], {"task_state_success": False}
        )
        (Path(directory) / "failed_prompt.txt").write_text(
            runner.planner[0].curr_prompt
        )
        raise
    trace, metrics = runner.finish_trial(instruction, raw["episode_id"], info)
    result = {
        key: info.get(key)
        for key in ("task_state_success", "task_percent_complete", "runtime")
    }
    result.update(metrics)
    result["context_sha256"] = hashlib.sha256(context.encode()).hexdigest()
    result["initial_observation_sha256"] = hashlib.sha256(
        json.dumps(trace["initial_observation"], sort_keys=True).encode()
    ).hexdigest()
    return trace, result


@hydra.main(
    version_base="1.3",
    config_path="../conf",
    config_name="baselines/partner_coordination",
)
def main(config):
    cfg = config.coordination
    candidates = read_candidates(config)
    if cfg.inspect_only:
        print(
            json.dumps(
                [
                    {
                        "episode_id": ep["episode_id"],
                        "instruction": ep["instruction"],
                        "scene_id": ep["scene_id"],
                        "layout_fingerprint": layout_fingerprint(ep),
                        "goals": [
                            g.__dict__
                            for g in extract_goals(ep, cfg.min_objects, cfg.min_groups)
                        ],
                    }
                    for ep in candidates
                ],
                indent=2,
            )
        )
        return
    if config.world_model.partial_obs or config.num_proc != 1:
        raise ValueError("This diagnostic requires full observation and num_proc=1")
    if config.robot_agent_uid != 0 or config.human_agent_uid != 1:
        raise ValueError("Requires robot UID 0, partner UID 1")
    if (
        len(set(cfg.conditions)) != len(cfg.conditions)
        or not cfg.conditions
        or set(cfg.conditions) - set(CONDITIONS)
    ):
        raise ValueError("Invalid or duplicate conditions")
    if len({p.id for p in cfg.partners}) != len(cfg.partners) or not cfg.partners:
        raise ValueError("Need unique partner IDs")
    for partner in cfg.partners:
        if (
            not re.fullmatch(r"[A-Za-z0-9_-]+", partner.id)
            or partner.profile not in PROFILE_DESCRIPTIONS
        ):
            raise ValueError("Invalid partner ID/profile")
    if not cfg.seeds or len(set(cfg.seeds)) != len(cfg.seeds):
        raise ValueError("Need distinct seeds")
    for key in (
        "max_sim_steps",
        "human_poll_steps",
        "observation_interval",
        "max_history_chars",
        "max_prompt_chars",
    ):
        if int(cfg[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    support, tests = choose_split(
        candidates, cfg.support_ids, cfg.test_ids, cfg.support_count, cfg.test_count
    )
    fix_config(config)
    config = setup_config(config, int(cfg.seeds[0]))
    full = CollaborationDatasetV0(config.habitat.dataset)
    ids = {str(ep["episode_id"]) for ep in support + tests}
    dataset = CollaborationDatasetV0(
        config.habitat.dataset,
        episodes=[ep for ep in full.episodes if str(ep.episode_id) in ids],
    )
    if not config.evaluation.save_video:
        remove_visual_sensors(config)
    register_sensors(config)
    register_actions(config)
    register_measures(config)
    out = Path(HydraConfig.get().runtime.output_dir) / "coordination"
    out.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(config, out / "config.yaml")
    (out / "split.json").write_text(
        json.dumps(
            {
                "support": [
                    {"episode_id": ep["episode_id"], "layout": layout_fingerprint(ep)}
                    for ep in support
                ],
                "test": [
                    {"episode_id": ep["episode_id"], "layout": layout_fingerprint(ep)}
                    for ep in tests
                ],
                "dataset": str(config.habitat.dataset.data_path),
            },
            indent=2,
        )
    )
    env = EnvironmentInterface(config, dataset=dataset, init_wg=False)
    try:
        env.initialize_perception_and_world_graph()
        runner = CoordinationEvaluationRunner(config.evaluation, env)
        for seed in cfg.seeds:
            for partner in cfg.partners:
                support_traces = []
                plans = [("support", "no_history", ep) for ep in support]
                plans += [
                    ("test", condition, ep)
                    for ep in tests
                    for condition in cfg.conditions
                ]
                frozen = None
                initial_hashes = {}
                for phase, condition, ep in plans:
                    if phase == "test" and frozen is None:
                        # Serialize once: test trials can neither append nor alter support.
                        frozen = json.loads(json.dumps(support_traces))
                    context = frozen_context(
                        condition,
                        frozen or [],
                        partner.profile,
                        int(cfg.max_history_chars),
                    )
                    name = f"{phase}_{condition}_{partner.id}_ep{ep['episode_id']}_seed{seed}"
                    row = {
                        "phase": phase,
                        "condition": condition,
                        "partner_id": partner.id,
                        "profile": partner.profile,
                        "episode_id": str(ep["episode_id"]),
                        "seed": int(seed),
                    }
                    try:
                        trace, metrics = run_trial(
                            runner,
                            env,
                            ep,
                            partner.profile,
                            context,
                            cfg,
                            int(seed),
                            out / name,
                        )
                        row.update(metrics)
                        if phase == "support":
                            support_traces.append(trace)
                        else:
                            key = str(ep["episode_id"])
                            baseline_hash = initial_hashes.setdefault(
                                key, row["initial_observation_sha256"]
                            )
                            row["paired_initial_state_matches"] = (
                                row["initial_observation_sha256"] == baseline_hash
                            )
                    except Exception as exc:
                        row["error"] = f"{type(exc).__name__}: {exc}"
                        with (out / "results.jsonl").open("a") as stream:
                            stream.write(json.dumps(row) + "\n")
                        raise
                    with (out / "results.jsonl").open("a") as stream:
                        stream.write(json.dumps(row) + "\n")
                    print(json.dumps(row))
    finally:
        env.env.close()


if __name__ == "__main__":
    main()
