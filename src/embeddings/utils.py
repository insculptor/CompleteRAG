from time import perf_counter


def timer(func):
    """Decorator that measures the execution time of a method."""
    def wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        print(f"[INFO]: Time taken to execute {func.__name__}: {end_time - start_time:.5f} seconds.")
        return result
    return wrapper