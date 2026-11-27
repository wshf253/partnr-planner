"""Shared high-level protocol for the symbolic and Habitat runners."""
from dataclasses import asdict, dataclass
import random


DINING = "table_1"    # living room in the selected HSSD scene
KITCHEN = "table_0"
SIDE = "table_2"
SCENE = "103997919_171031233"


@dataclass
class Dish:
    name: str
    kind: str
    location: str
    clean: bool


@dataclass
class Day:
    index: int
    dishes: list
    deadline: int
    probe: bool


def make_day(index, seed):
    rng = random.Random(f"table-{seed}-{index}")
    # One neutral day, two matched intervention probes, then alternating test loads.
    if index == 0:
        plates, cups, preset = 2, 0, 0
    else:
        busy = index % 2 == 0
        preset = rng.randrange(2)
        plates = 1 + preset + (rng.randint(3, 4) if busy else 2)
        cups = rng.randint(2, 3) if busy else rng.randint(1, 2)
    dishes = []
    for kind, count in (("plate", plates), ("cup", cups)):
        for i in range(count):
            loc = rng.choice((KITCHEN, SIDE))
            if kind == "plate" and i < preset:
                loc = DINING
            # Keep one clean cup available as the robot's independent task.
            clean = kind == "plate" or i == 0 or rng.random() < 0.35
            dishes.append(Dish(f"{kind}_{i}", kind, loc, clean))
    return Day(index, dishes, rng.choice((3, 4, 5)), index in (1, 2))


class Symbolic:
    """One clock tick per concurrent assignment, plus a tick for washing."""
    def __init__(self):
        self.steps = 0

    def execute(self, jobs, target):
        self.steps += 1 + int(any(not dish.clean for dish in jobs.values()))

    def verify_ready(self, day):
        if not all(d.clean and d.location == DINING for d in day.dishes):
            raise RuntimeError("The complete table setting was not observed")

    def score(self, day):
        return float(all(d.clean and d.location == KITCHEN for d in day.dishes))


class Partner:
    def __init__(self, profile):
        self.profile = profile
        self.lane = "plate"

    def select(self, remaining, load, joined):
        if joined and (self.profile == "B" or (self.profile == "C" and load >= 5)):
            self.lane = "cup"
        ordered = sorted(remaining, key=lambda d: (d.kind != self.lane, d.clean))
        return ordered[0] if ordered else None


def run_day(day, profile, decide, history, backend, deadline=None):
    human = Partner(profile)
    events = []
    evidence = []
    deadline = day.deadline if deadline is None else deadline

    def complete(jobs, target):
        # Backends report failure before this ledger is advanced.
        backend.execute(jobs, target)
        for agent, dish in jobs.items():
            events.append(dict(agent=agent, object=dish.name, target=target,
                               washed=not dish.clean, end=backend.steps))
            dish.location = target
            dish.clean = True

    initial = [asdict(d) for d in day.dishes]
    first = next(d for d in day.dishes if d.kind == "plate" and d.location != DINING)
    complete({1: first}, DINING)
    decision = None
    observation = None
    while True:
        pending = [d for d in day.dishes if d.location != DINING]
        if not pending:
            break
        plates = [d for d in pending if d.kind == "plate"]
        cups = [d for d in pending if d.kind == "cup" and d.clean]
        diagnostic = decision is None and len(plates) >= 2 and bool(cups)
        if diagnostic:
            observation = dict(plates_remaining=len(plates), cups_remaining=len(pending)-len(plates),
                               dirty_cups=sum(not d.clean for d in pending),
                               available_dishes=[asdict(d) for d in pending],
                               human_last_action=events[-1]["object"],
                               time_left=deadline-backend.steps, history=list(history))
            decision = "join" if day.probe else decide(observation)
            if decision not in ("join", "separate"):
                raise ValueError(f"Invalid robot decision {decision!r}")
            robot_dish = plates[0] if decision == "join" else cups[0]
        else:
            # After observing the partner's last action, take the complementary lane.
            last_kind = next((e["object"].split("_")[0] for e in reversed(events)
                              if e["agent"] == 1), "plate")
            options = [d for d in pending if d.clean and d.kind != last_kind]
            # Leave a sole remaining dish for the person if it is in their lane.
            if not options and len(pending) > 1:
                options = [d for d in pending if d.clean]
            robot_dish = options[0] if options else None
        joined = robot_dish is not None and robot_dish.kind == "plate" and human.lane == "plate"
        human_dish = human.select([d for d in pending if d is not robot_dish], len(pending), joined)
        jobs = {i: d for i, d in ((0, robot_dish), (1, human_dish)) if d is not None}
        complete(jobs, DINING)
        if diagnostic and joined:
            evidence.append(dict(remaining=len(pending),
                                 response="switch" if human_dish.kind == "cup" else "stay"))

    backend.verify_ready(day)
    ready_at = backend.steps
    for offset in range(0, len(day.dishes), 2):
        complete(dict(enumerate(day.dishes[offset:offset+2])), KITCHEN)
    state_success = backend.score(day)
    return dict(day=day.index + 1, initial=initial, probe=day.probe,
                decision=decision, observation=observation, evidence=evidence,
                ready_at=ready_at, deadline=deadline, on_time=ready_at <= deadline,
                final_step=backend.steps, task_state_success=state_success,
                success=bool(state_success and ready_at <= deadline), events=events)
