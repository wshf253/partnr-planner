"""Full-observation graph adapter. No partner intentions or chosen goals exposed."""


def snapshot(env, goals, previous_robot_position=None):
    graph = env.full_world_graph

    def position(node):
        return [round(float(x), 4) for x in node.properties["translation"]]

    robot = position(graph.get_spot_robot())
    human = position(graph.get_human())
    objects = {}
    for goal in goals:
        node = graph.get_node_from_name(goal.object)
        furniture = graph.find_furniture_for_object(node)
        holder = (
            0
            if graph.is_object_with_robot(node)
            else 1 if graph.is_object_with_human(node) else None
        )
        location = furniture.name if furniture is not None else None
        objects[goal.object] = {
            "position": position(node),
            "held_by": holder,
            "location": location,
            "target": goal.target,
            # Graph proxy for scheduling; Habitat task_state_success remains the
            # authoritative outcome (furniture membership is not an on-top test).
            "at_goal": holder is None and location == goal.target,
        }
    return {
        "robot_position": robot,
        "human_position": human,
        "previous_robot_position": previous_robot_position or robot,
        "objects": objects,
    }


def public_observation(scene, step):
    return {
        "sim_step": step,
        "robot_position": scene["robot_position"],
        "human_position": scene["human_position"],
        "objects": {
            name: {
                **{k: v for k, v in state.items() if k != "at_goal"},
                "held_by_role": {0: "self", 1: "partner", None: None}[state["held_by"]],
            }
            for name, state in scene["objects"].items()
        },
    }
