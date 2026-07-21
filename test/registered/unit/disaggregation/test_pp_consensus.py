import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue
from sglang.srt.disaggregation.utils import (
    get_pp_consensus_polls,
    merge_pp_consensus,
)
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPPConsensusUtils(unittest.TestCase):
    def test_build_polls_is_authoritative_and_failure_wins(self):
        polls = get_pp_consensus_polls(
            ["good", "bad", "overlap", "pending"],
            KVPoll.Success,
            [["good", "overlap"], ["bad", "overlap"]],
        )

        self.assertEqual(
            polls,
            [KVPoll.Success, KVPoll.Failed, KVPoll.Failed, None],
        )

    def test_merge_intersects_success_and_unions_failure(self):
        payload = merge_pp_consensus(
            [["shared", "previous-only", "fails-later"], ["previous-failure"]],
            ["shared", "current-only"],
            ["current-failure", "fails-later"],
        )

        self.assertEqual(set(payload[0]), {"shared"})
        self.assertEqual(
            set(payload[1]),
            {"previous-failure", "current-failure", "fails-later"},
        )

    def test_first_stage_deduplicates_and_failure_wins(self):
        payload = merge_pp_consensus(
            None, ["good", "good", "overlap"], ["bad", "overlap", "bad"]
        )
        self.assertEqual(set(payload[0]), {"good"})
        self.assertEqual(set(payload[1]), {"bad", "overlap"})


class TestPrefillPPConsensus(unittest.TestCase):
    @patch("sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group")
    def test_failed_consensus_skips_second_local_poll(self, mock_poll):
        req = SimpleNamespace(rid="failed", disagg_kv_sender=MagicMock())
        queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
        queue.queue = [req]
        queue.scheduler = SimpleNamespace(handle_bootstrap_failure=MagicMock())

        good, failed = queue.pop_bootstrapped(
            return_failed_reqs=True,
            pp_consensus=[[], [req.rid]],
        )

        self.assertEqual(good, [])
        self.assertEqual(failed, [req])
        self.assertEqual(queue.queue, [])
        mock_poll.assert_not_called()
        queue.scheduler.handle_bootstrap_failure.assert_called_once_with(req)

    @patch("sglang.srt.disaggregation.prefill.should_force_retry", return_value=False)
    @patch("sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group")
    def test_success_consensus_skips_late_local_failure(
        self, mock_poll, _mock_should_force_retry
    ):
        req = SimpleNamespace(
            rid="successful",
            disagg_kv_sender=MagicMock(),
            time_stats=MagicMock(),
        )
        queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
        queue.queue = [req]
        queue.scheduler = SimpleNamespace(handle_bootstrap_failure=MagicMock())
        queue.finalize_bootstrap = MagicMock(return_value=True)

        good, failed = queue.pop_bootstrapped(
            return_failed_reqs=True,
            pp_consensus=[[req.rid], []],
        )

        self.assertEqual(good, [req])
        self.assertEqual(failed, [])
        self.assertEqual(queue.queue, [])
        mock_poll.assert_not_called()
        queue.finalize_bootstrap.assert_called_once_with(req)

    def test_scheduler_preserves_failure_and_filters_deferred_success(self):
        req = SimpleNamespace(rid="good")
        queue = MagicMock()
        queue.pop_bootstrapped.return_value = ([req], [SimpleNamespace(rid="bad")])
        scheduler = SimpleNamespace(
            disagg_prefill_bootstrap_queue=queue,
            waiting_queue=[],
        )
        payload = [["good"], ["bad"]]

        forwarded = SchedulerPPMixin.process_bootstrapped_queue(scheduler, payload)

        self.assertEqual(forwarded, payload)
        self.assertEqual(scheduler.waiting_queue, [req])
        queue.pop_bootstrapped.assert_called_once_with(
            return_failed_reqs=True,
            pp_consensus=payload,
        )

        queue.pop_bootstrapped.return_value = ([], [SimpleNamespace(rid="bad")])
        forwarded = SchedulerPPMixin.process_bootstrapped_queue(scheduler, payload)
        self.assertEqual(forwarded, [[], ["bad"]])


if __name__ == "__main__":
    unittest.main()
