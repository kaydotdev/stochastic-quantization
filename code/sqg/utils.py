from itertools import islice


def batched_iterable(iterable, batch_size):
    iterator = iter(iterable)
    while True:
        batch = list(islice(iterator, batch_size + 1))
        if not batch:
            break
        yield batch
