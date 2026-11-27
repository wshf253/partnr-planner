"""Protocol regressions, HTTP contract, and concurrent skill scheduling."""
import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import Mock, patch

from .native import HabitatSession
from .qwen import MODEL, choose_qwen, choose_reference
from .scenario import DINING, KITCHEN, Symbolic, make_day, run_day


class ProtocolTests(unittest.TestCase):
    def test_paired_trials_and_end_states(self):
        for seed in range(20):
            trials = {}
            for profile in "ABC":
                history = []
                trials[profile] = []
                for index in range(7):
                    result = run_day(make_day(index, seed), profile, choose_reference, history, Symbolic())
                    self.assertTrue(result["task_state_success"])
                    self.assertEqual(result["on_time"], result["ready_at"] <= result["deadline"])
                    for event in result["events"]:
                        if event["washed"]:
                            self.assertEqual(event["agent"], 1)
                        if event["target"] == KITCHEN:
                            self.assertGreater(event["end"], result["ready_at"])
                    self.assertEqual(result["events"][0]["target"], DINING)
                    history.extend(result["evidence"])
                    trials[profile].append(result)
            self.assertEqual(trials["A"][0], trials["B"][0])
            self.assertEqual(trials["A"][0], trials["C"][0])
            for index in range(7):
                self.assertEqual(trials["A"][index]["initial"], trials["C"][index]["initial"])
            for profile, expected in {"A": ["stay", "stay"], "B": ["switch", "switch"],
                                      "C": ["stay", "switch"]}.items():
                observed = [r["evidence"][0]["response"] for r in trials[profile][1:3]]
                self.assertEqual(observed, expected)

    def test_history_changes_heldout_decision(self):
        rows = []
        memory = []
        for index in range(7):
            row = run_day(make_day(index, 17), "C", choose_reference, memory, Symbolic())
            memory.extend(row["evidence"])
            rows.append(row)
        self.assertEqual([r["decision"] for r in rows[3:]], ["separate", "join"] * 2)
        cold = run_day(make_day(4, 17), "C", choose_reference, [], Symbolic())
        self.assertEqual(cold["decision"], "separate")

    def test_late_meal_still_clears(self):
        r = run_day(make_day(1, 17), "A", choose_reference, [], Symbolic(), deadline=1)
        self.assertFalse(r["success"])
        self.assertTrue(r["task_state_success"])

    def test_http_decision_and_retry(self):
        requests = []
        class Endpoint(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                requests.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
                reply = '{"decision":"join"}' if len(requests) == 2 else 'invalid'
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({"choices": [{"message": {"content": reply}}]}).encode())

        with HTTPServer(("127.0.0.1", 0), Endpoint) as server:
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                env = {"QWEN_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
                       "QWEN_MODEL": MODEL}
                with patch.dict(os.environ, env):
                    trial = run_day(make_day(4, 17), "B", choose_qwen,
                                    [{"remaining": 6, "response": "switch"}], Symbolic())
            finally:
                server.shutdown()
                thread.join()
        self.assertEqual(trial["decision"], "join")
        self.assertEqual(len(requests), 2)
        path, payload = requests[0]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(payload["model"], "qwen3.8-27B")
        obs = json.loads(payload["messages"][1]["content"])
        self.assertEqual(obs, trial["observation"])
        self.assertNotIn("profile", obs)
        self.assertNotIn("partner", obs)

    def test_native_waits_for_both_agents(self):
        from .scenario import Dish
        session = HabitatSession.__new__(HabitatSession)
        session.names = {"p": "plate_8", "c": "cup_9", DINING: "table_9"}
        session.steps, session.obs, session.faucet = 0, {}, "counter_2"
        session.check_placement = Mock()
        session.env = Mock()
        session.env.step.return_value = ({}, 0, False, {})
        session.skills = Mock()
        calls = []
        def respond(actions, obs):
            calls.append({i: a[0] for i, a in actions.items()})
            return {i: 1 for i in actions}, {i: "Successful execution!" for i in actions}
        session.skills.process_high_level_actions.side_effect = respond
        session.execute({0: Dish("p", "plate", KITCHEN, True),
                         1: Dish("c", "cup", KITCHEN, False)}, DINING)
        self.assertEqual(calls, [{0: "Rearrange", 1: "Rearrange"}, {1: "Clean"}, {1: "Rearrange"}])
        self.assertEqual(session.steps, 3)

if __name__ == "__main__":
    unittest.main()
