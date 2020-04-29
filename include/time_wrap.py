import time
from functools import wraps

PROF_DATA = {}


def profile(fn):
    @wraps(fn)
    def with_profiling(*args, **kwargs):
        start_time = time.monotonic()

        ret = fn(*args, **kwargs)

        elapsed_time = time.monotonic() - start_time

        print(f'WRAP> Function {fn.__name__} with args {args} needed {elapsed_time:.2f} seconds to execute!')
        return ret

    return with_profiling

