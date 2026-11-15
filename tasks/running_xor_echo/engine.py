"""running_xor with the previous input digit echoed in brackets after every bar (tasks/common/echo.py).
Only marked spacings k in {2, 4, 8, 16} are run: the no-marker and plain marked runs are those of
running_xor, on the same inputs."""

from tasks.running_xor import engine as _base
from tasks.common.echo import make_echo

TASK_NAME = "running_xor_echo"
AXIS_NAME = _base.AXIS_NAME
AXIS_VALUES = _base.AXIS_VALUES
OUTPUT_HAS_BOUNDARIES = True
INTERVALS = [2, 4, 8, 16]
build_prompt = make_echo(_base)


def estimate_max_new_tokens(N):
    return 2 * N + 20          # room for a model that writes the echoes back
