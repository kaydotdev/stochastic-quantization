from sys import version_info

import itertools


if version_info >= (3, 12) and hasattr(itertools, "batched"):
    # Built-in since 3.12 (returns tuples)
    batched = itertools.batched  # type: ignore[attr-defined]
else:

    def batched(iterable, n, *, strict=False):
        """Back-port of itertools.batched for Py < 3.12 (returns tuples)."""

        if n < 1:
            raise ValueError("n must be >= 1")

        it = iter(iterable)
        while chunk := tuple(itertools.islice(it, n)):
            if strict and len(chunk) != n:
                raise ValueError("last batch smaller than n")

            yield chunk
