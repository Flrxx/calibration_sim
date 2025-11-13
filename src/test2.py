import multiprocessing
import ctypes

def worker(shared_array, index, value):
    shared_array[index] = value
    print(f"Process {multiprocessing.current_process().name}: set index {index} to {value:.2f}")

if __name__ == "__main__":
    # Create shared array of 8 floats, initialized to 0.0
    size = 8
    shared_array = multiprocessing.Array(ctypes.c_float, size)
    
    print(f"Initial array: {[shared_array[i] for i in range(size)]}")
    
    processes = []
    # Each process will update a different index
    for i in range(size):
        value = i * 1.5 + 0.7  # Some float calculation
        p = multiprocessing.Process(
            target=worker, 
            args=(shared_array, i, value),
            name=f"Process-{i}"
        )
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
    
    print(f"Final array: {[shared_array[i] for i in range(size)]}")