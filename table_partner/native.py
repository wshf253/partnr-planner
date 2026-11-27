"""Generate CollaborationEpisode data and execute the protocol with real PARTNR skills."""
from collections import deque
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
import random

from .scenario import Day, Dish, DINING, KITCHEN, SCENE, make_day


def generate(path, days, seed):
    import numpy as np
    from dataset_generation.benchmark_generation.generate_episodes import (
        default_gen_config, default_metadata_dict, generate_episode,
        initialize_generator, save_ep_dataset,
    )
    from habitat_llm.agent.env.evaluation.evaluation_functions import (
        EvaluationProposition, EvaluationPropositionDependency, TemporalConstraint,
        TerminalSatisfactionConstraint,
    )

    random.seed(seed)
    np.random.seed(seed)
    generator = initialize_generator(dict(default_gen_config), default_metadata_dict)
    episodes = []
    for index in range(days):
        day = make_day(index, seed)
        proposal = dict(
            scene_id=SCENE, episode_id=str(index),
            instruction="Set all plates and clean cups on the dining table, washing dirty cups first. "
                        "Afterwards clear every dish to the kitchen table.",
            initial_state=[dict(number=1, object_classes=[d.kind], furniture_names=[d.location],
                                object_states={"is_clean": d.clean}) for d in day.dishes],
        )
        for _ in range(5):
            ep, details = generate_episode(generator, proposal)
            if ep is not None:
                break
        if ep is None:
            raise RuntimeError(f"Cannot place dishes on day {index + 1}: {details}")
        objects = generator.ep_sampled_objects
        if len(objects) != len(day.dishes):
            raise RuntimeError("Generated dish count differs from proposal")
        handles = {d.name: obj.handle for d, obj in zip(day.dishes, objects)}
        furniture = generator.metadata_interface.recobj_semname_to_handle
        tables = {name: furniture[name] for name in {DINING, KITCHEN, *(d.location for d in day.dishes)}}
        ep.episode_id = str(index)
        ep.instruction = proposal["instruction"]
        ep.info["table_partner"] = dict(day=asdict(day), objects=handles, tables=tables)
        # Use a separate predicate per object, including dishes already at either table.
        # The runner independently checks the whole table at the meal boundary.
        ep.evaluation_propositions = [
            EvaluationProposition("is_on_top", dict(
                object_handles=[handle], receptacle_handles=[tables[DINING]]))
            for handle in handles.values()
        ]
        clean_indices = []
        for d in day.dishes:
            if d.kind == "cup":
                clean_indices.append(len(ep.evaluation_propositions))
                ep.evaluation_propositions.append(EvaluationProposition(
                    "is_clean", {"object_handles": [handles[d.name]]}))
        prepared = list(range(len(ep.evaluation_propositions)))
        cleared = []
        for handle in handles.values():
            cleared.append(len(ep.evaluation_propositions))
            ep.evaluation_propositions.append(EvaluationProposition("is_on_top", dict(
                object_handles=[handle], receptacle_handles=[tables[KITCHEN]])))
        ep.evaluation_proposition_dependencies = [
            EvaluationPropositionDependency(cleared.copy(), prepared, "after_satisfied")]
        ep.evaluation_constraints = [
            TemporalConstraint([(i, j) for i in prepared for j in cleared]),
            TerminalSatisfactionConstraint(clean_indices + cleared)]
        episodes.append(ep)
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as tmp:
        target = Path(tmp) / "episodes.json.gz"
        save_ep_dataset(episodes, str(target))
        target.replace(path)


class HabitatSession:
    def __init__(self, path, seed):
        from habitat_llm.agent.env import (
            CollaborationDatasetV0, EnvironmentInterface, register_actions, register_measures,
            register_sensors, remove_visual_sensors,
        )
        from habitat_llm.planner.planner import Planner
        from habitat_llm.utils.core import get_config, setup_config
        from habitat_llm.utils.sim import init_agents
        from omegaconf import open_dict

        from habitat.config.default_structured_configs import HabitatConfigPlugin, register_hydra_plugin
        from habitat_baselines.config.default_structured_configs import HabitatBaselinesConfigPlugin

        register_hydra_plugin(HabitatConfigPlugin)
        register_hydra_plugin(HabitatBaselinesConfigPlugin)
        cfg = get_config("examples/skill_runner_default_config")
        with open_dict(cfg.habitat.dataset):
            cfg.habitat.dataset.metadata = {"metadata_folder": "data/hssd-hab/metadata"}
            cfg.habitat.dataset.data_path = str(path.resolve())
        cfg.evaluation.save_video = False
        cfg = setup_config(cfg, seed)
        remove_visual_sensors(cfg)
        register_sensors(cfg)
        register_actions(cfg)
        register_measures(cfg)
        dataset = CollaborationDatasetV0(cfg.habitat.dataset)
        self.episodes = dataset.episodes
        self.env = EnvironmentInterface(cfg, dataset=dataset)
        self.skills = Planner({}, self.env)
        self.skills.agents = init_agents(cfg.evaluation.agents, self.env)

    def start(self, index):
        ep = self.episodes[index]
        self.env.reset_environment(episode_id=ep.episode_id)
        if self.env.sim.ep_info.episode_id != ep.episode_id:
            raise RuntimeError(f"Habitat failed to select episode {ep.episode_id}")
        for agent in self.skills.agents:
            agent.reset()
        meta = ep.info["table_partner"]
        spec = meta["day"]
        day = Day(spec["index"], [Dish(**d) for d in spec["dishes"]], spec["deadline"], spec["probe"])
        self.objects = meta["objects"]
        self.tables = meta["tables"]
        graph = self.env.world_graph[1]
        self.names = {k: graph.get_node_from_sim_handle(v).name
                      for k, v in {**self.objects, **self.tables}.items()}
        faucets = sorted(f.name for f in graph.get_all_furnitures()
                         if "faucet" in f.properties.get("components", []))
        self.faucet = next(iter(faucets), None)
        if not self.faucet and any(not d.clean for d in day.dishes):
            raise RuntimeError("Selected scene has no accessible faucet in the world graph")
        self.obs = self.env.get_observations()
        self.steps = 0
        for d in day.dishes:
            self.check_placement(d, d.location, d.clean)
        return day

    def check_placement(self, dish, target, clean):
        from habitat_llm.agent.env.evaluation.predicate_wrappers import SimBasedPredicates
        handle = self.objects[dish.name]
        sim = self.env.sim
        on_table = SimBasedPredicates.is_on_top(sim, [handle], [self.tables[target]])
        if not on_table.is_satisfied:
            raise RuntimeError(f"{dish.name} is not on {target} at step {self.steps}")
        if dish.kind == "cup":
            actual = sim.object_state_machine.get_snapshot_dict(sim)["is_clean"][handle]
            if bool(actual) != clean:
                raise RuntimeError(f"Unexpected cleanliness of {dish.name} at step {self.steps}")

    def execute(self, jobs, target):
        queues = {}
        for agent, dish in jobs.items():
            name = self.names[dish.name]
            dest = self.names[target]
            queue = deque()
            if not dish.clean:
                if agent != 1:
                    raise ValueError("Only the human can wash a cup")
                queue.extend([("Rearrange", f"{name},on,{self.faucet},none,none", None),
                              ("Clean", name, None)])
            queue.append(("Rearrange", f"{name},on,{dest},none,none", None))
            queues[agent] = queue
        # Each agent advances its own skill queue when that skill returns.
        # The other agent may still be navigating.
        for _ in range(6000):
            actions = {agent: q[0] for agent, q in queues.items() if q}
            if not actions:
                break
            low, responses = self.skills.process_high_level_actions(actions, self.obs)
            for agent, response in responses.items():
                if response:
                    if response.strip() != "Successful execution!":
                        raise RuntimeError(f"step {self.steps}, agent {agent}, {actions[agent]}: {response}")
                    queues[agent].popleft()
            if low:
                obs, _, done, _ = self.env.step(low)
                self.obs = self.env.parse_observations(obs)
                self.steps += 1
                if done:
                    raise RuntimeError(f"Habitat episode ended at step {self.steps}")
            elif not any(responses.values()):
                raise RuntimeError("PARTNR skills returned no action and no response")
        else:
            raise RuntimeError("Concurrent assignment exceeded its skill budget")
        for dish in jobs.values():
            self.check_placement(dish, target, True)

    def verify_ready(self, day):
        for dish in day.dishes:
            self.check_placement(dish, DINING, True)

    def score(self, day):
        return float(self.env.env.env.env._env.get_metrics()["task_state_success"])

    def close(self):
        self.env.env.close()
