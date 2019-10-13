import multiprocessing

# Thanks to The Little Book of Semaphores by Allen B. Downey for pseudocode
class Barrier:
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count = multiprocessing.Value('i', 0)
        self.mutex = multiprocessing.Semaphore(1)
        self.barrier = multiprocessing.Semaphore(0)

    def wait(self):
        self.mutex.acquire()
        self.count += 1
        self.mutex.release()
        if self.count == self.num_threads:
            self.barrier.release()
        self.barrier.acquire()
        self.barrier.release()

    def number_threads_waiting(self):
        return self.count
