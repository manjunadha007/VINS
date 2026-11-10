"""Exercise failed/late camera replies without needing a D455 or ROS."""
from concurrent.futures import Future
from types import SimpleNamespace as NS
import unittest

from projector_control import ALTERNATING, EMITTER, ProjectorControl, ProjectorState


class Parameters:
    def __init__(self):
        self.available = True
        self.requests = []

    def ready(self):
        return self.available

    def read(self):
        return self.request("read")

    def write(self, name, value):
        return self.request(name, value)

    def request(self, *args):
        future = Future()
        self.requests.append((args, future))
        return future

    def cancel(self, future):
        future.cancel()

    def reply(self, response):
        self.requests[-1][1].set_result(response)

    def state(self, emitter, alternating):
        self.reply(NS(values=[NS(type=2, integer_value=emitter),
                              NS(type=1, bool_value=alternating)]))

    def accepted(self, successful=True, reason=""):
        self.reply(NS(results=[NS(successful=successful, reason=reason)]))


class ProjectorTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.parameters = Parameters()
        self.control = ProjectorControl(self.parameters, clock=lambda: self.now)

    def connect(self, emitter=1, alternating=True):
        self.control.tick()
        self.parameters.state(emitter, alternating)
        self.control.tick()

    def test_initial_state_is_read_without_changing_camera(self):
        self.assertIsNone(self.control.state)
        self.assertFalse(self.control.set_enabled(True))
        self.connect()
        self.assertEqual(self.control.state, ProjectorState(1, True))
        self.assertEqual([r[0] for r in self.parameters.requests], [("read",)])

    def test_off_disables_alternation_then_laser_and_waits_for_readback(self):
        self.connect()
        self.assertTrue(self.control.set_enabled(False))
        self.assertEqual(self.parameters.requests[-1][0], (ALTERNATING, False))
        self.assertTrue(self.control.state.enabled)
        self.assertTrue(self.control.busy)
        self.assertFalse(self.control.set_enabled(True))
        self.parameters.accepted()
        self.control.tick()
        self.assertEqual(self.parameters.requests[-1][0], (EMITTER, 0))
        self.parameters.accepted()
        self.control.tick()
        self.assertEqual(self.parameters.requests[-1][0], ("read",))
        self.assertTrue(self.control.busy)
        self.parameters.state(0, False)
        self.control.tick()
        self.assertFalse(self.control.state.enabled)
        self.assertFalse(self.control.busy)

    def test_on_reapplies_alternation_after_enabling_laser(self):
        for alternating in (True, False):
            with self.subTest(alternating=alternating):
                self.setUp()
                self.connect(0, False)
                self.control.set_enabled(True, alternating=alternating)
                self.assertEqual(self.parameters.requests[-1][0],
                                 (EMITTER, 1) if alternating else (ALTERNATING, False))
                self.parameters.accepted()
                self.control.tick()
                self.assertEqual(self.parameters.requests[-1][0],
                                 (ALTERNATING, True) if alternating else (EMITTER, 1))
                self.parameters.accepted()
                self.control.tick()
                self.parameters.state(1, alternating)
                self.control.tick()
                self.assertEqual(self.control.state, ProjectorState(1, alternating))

    def test_partial_failure_reads_actual_state_and_preserves_error(self):
        self.connect()
        self.control.set_enabled(False)
        self.parameters.accepted()
        self.control.tick()
        self.parameters.accepted(False, "device busy")
        self.control.tick()
        self.assertFalse(self.control.busy)
        self.assertIsNone(self.control.state)
        self.assertIn("device busy", self.control.error)
        self.now = 2
        self.control.tick()
        self.parameters.state(1, False)
        self.control.tick()
        self.assertEqual(self.control.state, ProjectorState(1, False))
        self.assertIn("device busy", self.control.error)

    def test_readback_mismatch_is_not_reported_as_success(self):
        self.connect()
        self.control.set_enabled(False)
        for _ in range(2):
            self.parameters.accepted()
            self.control.tick()
        self.parameters.state(1, True)
        self.control.tick()
        self.assertTrue(self.control.state.enabled)
        self.assertIn("not confirmed", self.control.error)

    def test_timeout_cancels_pending_reply_then_recovers(self):
        self.connect()
        self.control.set_enabled(False)
        pending = self.parameters.requests[-1][1]
        self.now = 6
        self.control.tick()
        self.assertTrue(pending.cancelled())
        self.assertFalse(self.control.busy)
        self.assertIsNone(self.control.state)
        self.assertIn("did not respond", self.control.error)
        self.now = 8
        self.control.tick()
        self.parameters.state(0, False)
        self.control.tick()
        self.assertFalse(self.control.state.enabled)

    def test_disconnect_clears_stale_state_and_reconnect_reads_it(self):
        self.connect()
        self.parameters.available = False
        self.now = 2
        self.control.tick()
        self.assertIsNone(self.control.state)
        self.assertFalse(self.control.set_enabled(False))
        self.parameters.available = True
        self.now = 4
        self.control.tick()
        self.parameters.state(0, False)
        self.control.tick()
        self.assertFalse(self.control.state.enabled)
        self.assertEqual(self.control.connection_error, "")

    def test_missing_parameter_does_not_look_like_projector_off(self):
        self.control.tick()
        self.parameters.reply(NS(values=[NS(type=0), NS(type=0)]))
        self.control.tick()
        self.assertIsNone(self.control.state)
        self.assertIn("not supported", self.control.connection_error)

    def test_poll_observes_external_changes_and_close_cancels(self):
        self.connect()
        self.now = 2
        self.control.tick()
        self.parameters.state(0, False)
        self.control.tick()
        self.assertFalse(self.control.state.enabled)
        self.now = 4
        self.control.tick()
        pending = self.parameters.requests[-1][1]
        self.control.close()
        self.assertTrue(pending.cancelled())


if __name__ == "__main__":
    unittest.main()
