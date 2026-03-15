import csv
import tempfile
import unittest
from pathlib import Path

from cvrp_aco_solver import ACOConfig, solve_all_aco, solve_instance_aco
from cvrp_utils import parse_vrp


BASE_DIR = Path(__file__).resolve().parent


class TestCVRPAntColony(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = ACOConfig(
            ants=35,
            iterations=140,
            alpha=1.0,
            beta=4.2,
            rho=0.15,
            q0=0.55,
            xi=0.08,
            stall_iterations=35,
            time_limit_s=2.0,
        )

    def test_small_euc_instance(self) -> None:
        path = str(BASE_DIR / "P" / "P-n16-k8.vrp")
        routes, cost, feasible, _ = solve_instance_aco(path, self.cfg, seed=42)
        inst = parse_vrp(path)

        self.assertTrue(feasible)
        self.assertLessEqual(cost, 470)
        self.assertLessEqual(len(routes), inst.k_hint)

        served = [c for r in routes for c in r]
        self.assertEqual(set(served), set(inst.customers))
        self.assertEqual(len(served), len(inst.customers))

    def test_small_explicit_instance(self) -> None:
        path = str(BASE_DIR / "E" / "E-n13-k4.vrp")
        routes, cost, feasible, _ = solve_instance_aco(path, self.cfg, seed=42)
        inst = parse_vrp(path)

        self.assertTrue(feasible)
        self.assertLessEqual(cost, 285)
        self.assertLessEqual(len(routes), inst.k_hint)

    def test_batch_creates_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_csv = str(Path(tmp) / "aco_results.csv")
            rows = solve_all_aco(
                base_dir=str(BASE_DIR),
                output_csv=out_csv,
                cfg=ACOConfig(ants=22, iterations=70, stall_iterations=20, time_limit_s=1.2),
                seed=11,
                limit=4,
            )

            self.assertEqual(len(rows), 4)
            self.assertTrue(Path(out_csv).exists())

            with open(out_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                data = list(reader)

            self.assertEqual(len(data), 4)
            self.assertTrue(all(r["instance"] for r in data))


if __name__ == "__main__":
    unittest.main()
