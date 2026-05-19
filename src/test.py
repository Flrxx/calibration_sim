import time

def my_decorator(func):
    def measure_time(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        stop_time = time.time()
        print(f"Time {stop_time - start_time}")
    return measure_time

@my_decorator
def test(n):
    for _ in range(n):
        time.sleep(1)


test(3)