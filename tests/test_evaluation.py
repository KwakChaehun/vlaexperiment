from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from vla_eval.contracts import Observation, validate_chunk
from vla_eval.dataset import PairedDataset
from vla_eval.flow import EulerFlowPolicy
from vla_eval.horizon_statistics import horizon_comparisons
from vla_eval.offline import evaluate, train_small
from vla_eval.openpi import OpenPiNativePolicy
from vla_eval.runner import Trial, check_config, execute_trial, trial_grid
from vla_eval.small import RidgeSmall, fit_ridge, observation_features
from vla_eval.statistics import paired_comparisons, wilson
from vla_eval.toy import ToyHeavy, ToyPrecisionEnv, ToySmall, plan


ROOT = Path(__file__).resolve().parents[1]


def base_config():
    return json.loads((ROOT / "configs/comparison.json").read_text())


class ConstantEngine:
    name = "constant_test_engine"

    def __init__(self):
        self.times = []
        self.shapes = []

    def condition(self, obs, horizon):
        return float(obs.state[0])

    def noise(self, horizon, seed, condition):
        self.shapes.append((1, horizon, 1))
        return np.zeros((1, horizon, 1), dtype=np.float32)

    def velocity(self, latent, t, condition):
        self.times.append(t)
        return np.full_like(latent, condition)

    def decode(self, latent, condition):
        return latent[0]

    def synchronize(self):
        pass


class FlowTests(unittest.TestCase):
    def test_split_preserves_time_and_uses_fresh_condition(self):
        engine = ConstantEngine()
        policy = EulerFlowPolicy(engine)
        cache = policy.prefill(Observation(np.array([1.])), 3, 4, 2, 0)
        result = policy.resume(Observation(np.array([3.])), cache)
        np.testing.assert_allclose(result, 2.0)
        self.assertEqual(engine.times, [0., .25, .5, .75])
        self.assertEqual(cache.completed, 2)
        np.testing.assert_allclose(cache.latent, .5)

    def test_same_condition_split_matches_full(self):
        policy = EulerFlowPolicy(ConstantEngine())
        obs = Observation(np.array([2.]))
        np.testing.assert_allclose(policy.predict(obs, 7, 8, 0),
                                   policy.resume(obs, policy.prefill(obs, 7, 8, 3, 0)))

    def test_requested_generation_length_changes_actual_tensor(self):
        engine = ConstantEngine()
        policy = EulerFlowPolicy(engine)
        for h in (1, 4, 16):
            policy.predict(Observation(np.array([1.])), h, 2, 0)
        self.assertEqual(engine.shapes, [(1, 1, 1), (1, 4, 1), (1, 16, 1)])

    def test_invalid_flow_split_fails(self):
        with self.assertRaises(ValueError):
            EulerFlowPolicy(ConstantEngine()).prefill(Observation(np.zeros(1)), 1, 4, 5, 0)


class RunnerTests(unittest.TestCase):
    def test_invalid_horizon_and_nonfinite_actions_rejected(self):
        for value in (np.zeros((3, 2)), np.full((4, 2), np.nan)):
            with self.assertRaises(ValueError):
                validate_chunk(value, 4)

    def test_grid_never_executes_more_than_generated(self):
        config = base_config()
        config.update(horizons=[1, 4], execute=[1, 4, 16], seeds=[0], methods=["A"])
        grid = list(trial_grid(config))
        self.assertTrue(all(t.execute <= t.horizon for t, _ in grid))
        self.assertEqual({(t.horizon, t.execute) for t, _ in grid}, {(1, 1), (4, 1), (4, 4)})

    def test_configuration_rejects_bad_inputs(self):
        for update in ({"execute": [0]}, {"partial_fraction": 1}, {"seeds": [0, 0]},
                       {"flow_steps": [True]}, {"environment": {"control_hz": 0}}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                check_config({**base_config(), **update})

    def test_forecast_does_not_read_arrival_perturbation(self):
        a = ToyPrecisionEnv(3, {"arrival_position_mm": 0})
        b = ToyPrecisionEnv(3, {"arrival_position_mm": 10})
        np.testing.assert_equal(a.forecast().observation.state, b.forecast().observation.state)
        self.assertFalse(np.array_equal(a.observe().state, b.observe().state))
        with self.assertRaises(RuntimeError):
            b.forecast()

    def test_forecasts_vary_across_trials_for_shuffle_control(self):
        a = ToyPrecisionEnv(0, {}).forecast().observation.state
        b = ToyPrecisionEnv(1, {}).forecast().observation.state
        self.assertFalse(np.array_equal(a, b))

    def test_latency_residual_and_generation_accounting(self):
        config = base_config()
        config.update(latency_mode="model", modeled_latency_ms={"heavy_pre": 800,
                      "correction_post": 20, "small_post": 10})
        config["environment"].update(max_actions=4, reach_duration_s=.5)
        trial = Trial("test", "mismatch", "C", 0, 16, 4, 8, 4)
        row = execute_trial(trial, {"name": "mismatch"}, config,
                            ToyHeavy(), ToySmall(), ToyPrecisionEnv)
        self.assertEqual(row["first_action_delay_ms"], 320)
        self.assertEqual(row["generated_actions"], 16)
        self.assertEqual(row["used_actions"], 4)
        self.assertEqual(row["wasted_actions"], 12)
        self.assertEqual(row["generation_utilization"], .25)
        self.assertEqual(row["generated_actions_per_executed_action"], 4.)
        self.assertAlmostEqual(row["post_inference_per_executed_action_ms"],
                               sum(c["measured_ms"] for c in row["trace"]) / row["used_actions"])

    def test_D_includes_remaining_heavy_work(self):
        config = base_config()
        config.update(latency_mode="model", modeled_latency_ms={"partial_pre": 200,
                      "resume_and_correction_post": 150, "small_post": 10})
        config["environment"]["max_actions"] = 4
        row = execute_trial(Trial("test", "x", "D", 2, 16, 4, 8, 4), {"name": "x"},
                            config, ToyHeavy(), ToySmall(), ToyPrecisionEnv)
        self.assertEqual(row["first_action_delay_ms"], 150)

    def test_resume_only_never_calls_small_and_includes_resume_cost(self):
        class ForbiddenSmall:
            def predict(self, *args):
                raise AssertionError("resume_only must use Heavy on later replans")

            def correct(self, *args):
                raise AssertionError("resume_only must isolate Heavy reuse")

            def synchronize(self):
                raise AssertionError("resume_only must not synchronize Small")

        config = base_config()
        config.update(latency_mode="model", modeled_latency_ms={"partial_pre": 200,
                      "resume_post": 40, "heavy_post": 90})
        config["environment"].update(max_actions=8, tolerance_mm=0.00001,
                                     rotation_tolerance_deg=0.00001)
        row = execute_trial(Trial("test", "x", "resume_only", 2, 16, 4, 8, 4),
                            {"name": "x"}, config, ToyHeavy(), ForbiddenSmall(), ToyPrecisionEnv)
        self.assertEqual(row["first_action_delay_ms"], 40)
        self.assertEqual(row["trace"][0]["stage"], "resume_post")
        self.assertEqual(row["trace"][1]["stage"], "heavy_post")


def write_dataset(path, leak=False, missing_forecast=False):
    rng = np.random.default_rng(51)
    observations = [ToyPrecisionEnv(i, {}).observe() for i in range(8)]
    forecasts = [ToyPrecisionEnv(i, {}).forecast().observation for i in range(8)]
    arrays = dict(episode_id=np.array([f"e{i}" for i in range(8)]),
                  split=np.array(["train"] * 4 + ["test"] * 4),
                  state=np.stack([o.state for o in observations]),
                  forecast_state=np.stack([o.state for o in forecasts]),
                  actions=np.stack([plan(o, 4) for o in observations]),
                  action_mask=np.ones((8, 4), dtype=bool),
                  forecast_source=np.array(["stale_observation"] * 8))
    if leak:
        arrays["episode_id"][4] = "e0"
    if missing_forecast:
        arrays.pop("forecast_state")
    np.savez_compressed(path, **arrays)


class OfflineTests(unittest.TestCase):
    @staticmethod
    def repeated_dataset(path, repeats=5):
        write_dataset(path)
        with np.load(path, allow_pickle=False) as original:
            arrays = {k: np.repeat(original[k], repeats, axis=0) for k in original.files}
        np.savez_compressed(path, **arrays)

    def test_subset_caps_episodes_and_uniformly_samples_each(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.npz"
            self.repeated_dataset(path)
            dataset = PairedDataset(path)
            try:
                self.assertEqual(dataset.select_indices("test", max_episodes=2,
                                 max_samples_per_episode=3), [20, 22, 24, 25, 27, 29])
                self.assertEqual(dataset.select_indices("test", episode_ids=["e5"],
                                 sample_stride=2, max_samples_per_episode=2), [25, 29])
            finally:
                dataset.close()

    def test_subset_rejects_invalid_limits_and_missing_episodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.npz"
            write_dataset(path)
            dataset = PairedDataset(path)
            try:
                for selection in ({"max_episodes": 0}, {"sample_stride": -1},
                                  {"episode_ids": ["e0"]}, {"max_samples_per_episode": True}):
                    with self.subTest(selection=selection), self.assertRaises(ValueError):
                        dataset.select_indices("test", **selection)
            finally:
                dataset.close()

    def test_subset_is_fixed_across_methods_and_K(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.npz"
            self.repeated_dataset(path, 3)
            config = base_config()
            config.update(dataset=str(path), seeds=[0], horizons=[4], execute=[1, 4],
                          methods=["A", "B"], flow_steps=[4], scenarios=[{"name": "arrival"}],
                          selection={"max_episodes": 2, "max_samples_per_episode": 2})
            rows = []
            evaluate(config, rows.append)
            self.assertEqual(len(rows), 16)
            for method in ("A", "B"):
                for k in (1, 4):
                    self.assertEqual([r["sample_index"] for r in rows
                                      if r["method"] == method and r["execute"] == k], [12, 14, 15, 17])

    def test_train_diagnostics_are_explicit_A_only_and_labeled(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.npz"
            write_dataset(path)
            config = base_config()
            config.update(dataset=str(path), seeds=[0], horizons=[4], execute=[1], methods=["A"],
                          flow_steps=[4], scenarios=[{"name": "arrival"}], split="train",
                          selection={"max_episodes": 1})
            with self.assertRaises(ValueError):
                evaluate(config, lambda row: None)
            config["allow_train_diagnostics"] = True
            rows = []
            evaluate(config, rows.append)
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0]["diagnostic_only"])
            config["methods"] = ["A", "B"]
            with self.assertRaises(ValueError):
                evaluate(config, lambda row: None)

    def test_episode_split_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.npz"
            write_dataset(path, leak=True)
            with self.assertRaises(ValueError):
                PairedDataset(path)

    def test_missing_forecast_does_not_fall_back_to_arrival(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.npz"
            write_dataset(path, missing_forecast=True)
            dataset = PairedDataset(path)
            try:
                with self.assertRaises(ValueError):
                    dataset.observation(0, forecast=True)
            finally:
                dataset.close()

    def test_joint_small_training_and_all_four_methods(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.npz"
            checkpoint = Path(tmp) / "small.npz"
            write_dataset(path)
            config = base_config()
            config.update(dataset=str(path), seeds=[0], horizons=[4], execute=[1, 4],
                          flow_steps=[4], scenarios=[{"name": "arrival"}])
            train_small(config, checkpoint, 1.)
            self.assertTrue(checkpoint.exists())
            small = RidgeSmall(checkpoint)
            self.assertEqual(small.train_episode_ids, {"e0", "e1", "e2", "e3"})
            config["small_checkpoint"] = str(checkpoint)
            rows = []
            evaluate(config, rows.append)
            self.assertEqual(len(rows), 48)
            self.assertTrue(all(row["success"] is None for row in rows))
            self.assertTrue(all(np.isfinite(row["action_mse"]) for row in rows))
            comparisons = paired_comparisons(rows, 40)
            self.assertTrue(comparisons)

    def test_predicted_forecast_requires_generation_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.npz"
            write_dataset(path)
            with np.load(path, allow_pickle=False) as original:
                arrays = {k: original[k] for k in original.files}
            arrays["forecast_source"] = np.array(["predicted_observation"] * 8)
            np.savez_compressed(path, **arrays)
            dataset = PairedDataset(path)
            try:
                with self.assertRaises(ValueError):
                    dataset.forecast_latency_ms(0)
            finally:
                dataset.close()

    def test_ridge_training_uses_same_weights_for_B_C(self):
        rng = np.random.default_rng(5)
        features = rng.normal(size=(20, 2))
        priors = rng.normal(size=(20, 3, 1))
        targets = np.broadcast_to(features[:, :1, None], (20, 3, 1)).copy()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "small.npz"
            fit_ridge(features, priors, targets, [f"e{i}" for i in range(20)], path)
            small = RidgeSmall(path)
            obs = Observation(np.array([.1, -.2]))
            self.assertEqual(small.predict(obs, 3, 4, 0).shape, (3, 1))
            self.assertEqual(small.correct(obs, priors[0], obs).shape, (3, 1))

    def test_padding_is_not_scored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.npz"
            write_dataset(path)
            with np.load(path, allow_pickle=False) as original:
                arrays = {k: original[k] for k in original.files}
            arrays["action_mask"][:, 2:] = False
            arrays["actions"][:, 2:] = np.nan
            np.savez_compressed(path, **arrays)
            config = base_config()
            config.update(dataset=str(path), seeds=[0], horizons=[4], execute=[4], methods=["A"],
                          flow_steps=[4], scenarios=[{"name": "arrival"}])
            rows = []
            evaluate(config, rows.append)
            self.assertTrue(all(row["used_actions"] == 2 for row in rows))
            self.assertTrue(all(np.isfinite(row["action_mse"]) for row in rows))

    def test_offline_reuse_probe_compares_against_A_without_B(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.npz"
            write_dataset(path)
            config = base_config()
            config.update(dataset=str(path), seeds=[0], horizons=[4], execute=[1],
                          methods=["A", "resume_only"], flow_steps=[4],
                          scenarios=[{"name": "arrival"}])
            rows = []
            evaluate(config, rows.append)
            self.assertEqual(len(rows), 8)
            comparisons = paired_comparisons(rows, 20)
            self.assertTrue(comparisons)
            self.assertEqual({c["comparison"] for c in comparisons}, {"resume_only-A"})


class StatisticsTests(unittest.TestCase):
    def test_wilson_is_not_degenerate_at_zero_success(self):
        low, high = wilson(0, 10)
        self.assertAlmostEqual(low, 0)
        self.assertGreater(high, .2)

    def test_paired_differences_cluster_correlated_frames(self):
        rows = []
        for i in range(3):
            for method, mse in (("B", 2.), ("C", 1.)):
                rows.append(dict(suite="offline", scenario="x", method=method, horizon=4,
                                 execute=1, flow_steps=4, partial_steps=2, latency_mode="wall",
                                 seed=i, episode_id="same_episode", action_mse=mse, success=None))
        result = paired_comparisons(rows, 20)
        self.assertEqual(result[0]["pairs"], 3)
        self.assertEqual(result[0]["independent_units"], 1)
        self.assertEqual(result[0]["mean_delta"], -1.)

    def test_unpaired_trials_are_not_used(self):
        rows = []
        for method, seed in (("B", 0), ("C", 1)):
            rows.append(dict(suite="x", scenario="x", method=method, horizon=1, execute=1,
                             flow_steps=4, partial_steps=2, latency_mode="wall", seed=seed,
                             action_mse=1., success=None))
        self.assertEqual(paired_comparisons(rows, 20), [])


class OpenPiBindingTests(unittest.TestCase):
    def test_native_binding_passes_images_state_and_deterministic_noise(self):
        from types import SimpleNamespace
        calls = []

        def infer(inputs, *, noise):
            calls.append((inputs, noise.copy()))
            return {"actions": np.zeros((10, 7), dtype=np.float32)}

        policy = OpenPiNativePolicy.__new__(OpenPiNativePolicy)
        policy.horizon, policy.action_dim, policy.flow_steps = 10, 32, 10
        policy.policy = SimpleNamespace(infer=infer)
        obs = Observation(np.zeros(8, dtype=np.float32),
                          {k: np.zeros((1, 4, 4, 3), dtype=np.uint8)
                           for k in ("image", "wrist_image")}, instruction="place the bowl")
        self.assertEqual(policy.predict(obs, 10, 10, 2).shape, (10, 7))
        policy.predict(obs, 10, 10, 2)
        np.testing.assert_array_equal(calls[0][1], calls[1][1])
        self.assertEqual(calls[0][1].shape, (10, 32))
        self.assertEqual(calls[0][0]["prompt"], obs.instruction)
        self.assertEqual(calls[0][0]["observation/image"].shape, (4, 4, 3))
        with self.assertRaises(ValueError):
            policy.predict(obs, 4, 10, 2)
        with self.assertRaises(ValueError):
            policy.predict(obs, 10, 4, 2)


class HorizonStatisticsTests(unittest.TestCase):
    @staticmethod
    def row(h, k, seed=0, n=4, suite="horizon"):
        return dict(suite=suite, scenario="x", method="A", horizon=h, execute=k,
                    flow_steps=n, partial_steps=n // 2, latency_mode="zero", seed=seed,
                    success=h < 4, position_error_mm=h * (100 if n == 8 else 1),
                    first_action_mse=h / 10, action_mse=h * k)

    def test_axes_hold_the_other_length_fixed(self):
        rows = [self.row(h, k) for h in (1, 2, 4) for k in (1, 2, 4) if k <= h]
        comparisons = horizon_comparisons(rows, 20)
        self.assertEqual({r["axis"] for r in comparisons}, {"generation", "execution", "joint"})
        for r in comparisons:
            a, b = r["candidate"], r["reference"]
            if r["axis"] == "generation":
                self.assertEqual(a["K"], b["K"])
                self.assertLess(a["H"], b["H"])
            elif r["axis"] == "execution":
                self.assertEqual(a["H"], b["H"])
                self.assertLess(a["K"], b["K"])
            else:
                self.assertEqual(a["H"], a["K"])
                self.assertIn("cannot separate", r["effect_attribution"])

    def test_flow_step_counts_are_not_pooled(self):
        rows = [self.row(h, 1, n=n) for n in (4, 8) for h in (2, 4)]
        contrasts = [r for r in horizon_comparisons(rows, 20)
                     if r["metric"] == "position_error_mm"]
        self.assertEqual({r["flow_steps"]: r["mean_delta"] for r in contrasts}, {4: -2., 8: -200.})

    def test_unpaired_horizon_trials_are_excluded(self):
        rows = [self.row(4, 1, seed=0), self.row(2, 1, seed=1)]
        self.assertEqual(horizon_comparisons(rows, 20), [])

    def test_offline_does_not_infer_reactivity_from_prefix_MSE(self):
        rows = [self.row(h, k, suite="offline_horizon") for h in (2, 4) for k in (1, 2)]
        comparisons = horizon_comparisons(rows, 20)
        self.assertEqual({r["axis"] for r in comparisons}, {"generation"})
        self.assertTrue(any(r["metric"] == "action_mse" for r in comparisons))

    def test_horizon_bootstrap_clusters_correlated_episode_snapshots(self):
        rows = [dict(self.row(h, 1, seed=i), episode_id="same_episode")
                for h in (2, 4) for i in range(3)]
        comparisons = horizon_comparisons(rows, 20)
        self.assertTrue(all(r["independent_units"] == 1 and r["pairs"] == 3 for r in comparisons))

    def test_duplicate_horizon_pair_is_rejected(self):
        with self.assertRaises(ValueError):
            horizon_comparisons([self.row(4, 1), self.row(4, 1)], 20)


if __name__ == "__main__":
    unittest.main()
