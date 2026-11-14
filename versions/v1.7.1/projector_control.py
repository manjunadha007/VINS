"""Non-blocking RealSense projector control, driven by the caller's timer.

Only parameter futures cross the ROS/UI thread boundary. A successful write is
followed by a readback; the displayed state never comes from a button click.
"""
from dataclasses import dataclass
import time


EMITTER = "depth_module.emitter_enabled"
ALTERNATING = "depth_module.emitter_on_off"


@dataclass(frozen=True)
class ProjectorState:
    emitter: int
    alternating: bool

    @property
    def enabled(self):
        return self.emitter != 0 or self.alternating


class RosProjectorParameters:
    def __init__(self, node, camera_node="/camera/camera"):
        from rcl_interfaces.srv import GetParameters, SetParameters
        self.get_type, self.set_type = GetParameters, SetParameters
        self.get = node.create_client(GetParameters, camera_node + "/get_parameters")
        self.set = node.create_client(SetParameters, camera_node + "/set_parameters")

    def ready(self):
        return self.get.service_is_ready() and self.set.service_is_ready()

    def read(self):
        return self.get.call_async(self.get_type.Request(names=[EMITTER, ALTERNATING]))

    def write(self, name, value):
        from rclpy.parameter import Parameter
        return self.set.call_async(self.set_type.Request(
            parameters=[Parameter(name, value=value).to_parameter_msg()]))

    def cancel(self, future):
        self.get.remove_pending_request(future)
        self.set.remove_pending_request(future)
        future.cancel()


class ProjectorControl:
    """Poll camera state and serialize writes without blocking an executor."""

    def __init__(self, parameters, clock=time.monotonic, timeout=5.0):
        self.parameters = parameters
        self.clock = clock
        self.timeout = timeout
        self.state = None
        self.error = ""
        self.connection_error = ""
        self._future = None
        self._operation = None
        self._deadline = 0.0
        self._next_read = 0.0
        self._writes = []
        self._target = None

    @property
    def busy(self):
        return self._target is not None

    def set_enabled(self, enabled, alternating=True):
        if self.busy or self.state is None or not self.parameters.ready():
            return False
        self._cancel_pending()
        self.error = ""
        self._target = ProjectorState(int(enabled), bool(enabled and alternating))
        # D455 options interact without updating each other's ROS values:
        # disabling alternation can leave the laser on; setting emitter_enabled
        # cancels hardware alternation. Set the final hardware mode LAST.
        # The splitter rejects projected tracking frames during the transition.
        self._writes = ([(EMITTER, 1), (ALTERNATING, True)]
                        if self._target.alternating else
                        [(ALTERNATING, False), (EMITTER, int(enabled))])
        self._advance()
        return True

    def _cancel_pending(self):
        if self._future is not None:
            self.parameters.cancel(self._future)
            self._future = None

    def close(self):
        self._cancel_pending()
        self._writes = []
        self._target = None

    def _advance(self):
        try:
            if self._writes:
                name, value = self._writes.pop(0)
                self._operation = name
                self._future = self.parameters.write(name, value)
            else:
                self._operation = "read"
                self._future = self.parameters.read()
            self._deadline = self.clock() + self.timeout
        except Exception as exc:
            self._failed(str(exc))

    def _failed(self, reason):
        self._cancel_pending()
        if self.busy:
            self.error = "Projector change failed: " + reason
        else:
            self.connection_error = "Camera control unavailable: " + reason
        self._target = None
        self._writes = []
        self.state = None
        self._next_read = self.clock() + 1.0

    def tick(self):
        now = self.clock()
        if self._future is not None:
            if not self._future.done():
                if now >= self._deadline:
                    self._failed("camera did not respond")
                return
            future, self._future = self._future, None
            try:
                response = future.result()
                if self._operation == "read":
                    values = response.values
                    if (len(values) != 2 or values[0].type != 2
                            or values[0].integer_value not in (0, 1, 2, 3)
                            or values[1].type != 1):
                        raise ValueError("projector parameters are not supported")
                    self.state = ProjectorState(values[0].integer_value,
                                                values[1].bool_value)
                    self.connection_error = ""
                    if self.busy and self.state != self._target:
                        self.error = "Projector change was not confirmed by the camera"
                    self._target = None
                    self._next_read = now + 1.0
                else:
                    if len(response.results) != 1 or not response.results[0].successful:
                        reason = (response.results[0].reason if response.results
                                  else "empty response")
                        raise ValueError(f"{self._operation}: {reason}")
                    self._advance()
            except Exception as exc:
                self._failed(str(exc))
            return
        if now >= self._next_read:
            if not self.parameters.ready():
                self.state = None
                self.connection_error = "Waiting for camera control"
                self._next_read = now + 1.0
            else:
                self._advance()
