# https://github.com/louisabraham/tqdm_joblib
import contextlib

import joblib

from .tqdm_wrapper import tqdm, _TQDM_AVAILABLE


@contextlib.contextmanager
def tqdm_joblib(*args, **kwargs):
    """Context manager to patch joblib to report into tqdm progress bar
    given as argument"""

    tqdm_object = tqdm(*args, **kwargs)

    if not _TQDM_AVAILABLE:
        yield tqdm_object
        return

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def ParallelPbar(desc=None, **tqdm_kwargs):
    class Parallel(joblib.Parallel):
        def __call__(self, it):
            it = list(it)
            with tqdm_joblib(total=len(it), desc=desc, **tqdm_kwargs):
                return super().__call__(it)

    return Parallel
