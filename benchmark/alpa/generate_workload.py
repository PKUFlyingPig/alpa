from copy import deepcopy
from typing import List
import numpy as np
from numpy.random import choice, exponential
import matplotlib.pyplot as plt
import pickle
import time
import threading

class PossoinWorkLoad:
    def __init__(self, model_num: int, tot_arrival_rate: float, proportions: List[float], duration: float, workload_name: str = "PossoinWorkLoad"):
        """
            @param model_num: the number of different models in this workload.
            @param tot_arrival_rate: The total arrival rate of the requests for 
                                     all the models, measured in Hz.
            @param proportions: the proportion of the requests for each model 
            @param duration: the duration of the workload, measured in Second.
        """
        assert len(proportions) == model_num and tot_arrival_rate > 0 and sum(proportions) == 1
        self.model_num = model_num
        self.tot_arrival_rate = tot_arrival_rate
        self.proportions = proportions
        self.duration = duration
        self.workload_name = workload_name

        self.model_ids = []
        self.arrive_times = []
        rela_time = 0.0
        while rela_time < duration:
            self.model_ids.append(choice(model_num, p=proportions))
            self.arrive_times.append(rela_time)
            rela_time += exponential(1/tot_arrival_rate)
        print(f"There are {len(self.arrive_times)} requests in total.")
     
    def plot(self, binwidth: int = 1, figname: str = None):
        plt.figure()
        for id in range(self.model_num):
            arrive_times = [self.arrive_times[i] for i, model_id in enumerate(self.model_ids) if model_id == id]
            arrive_hist, bin_edges = np.histogram(arrive_times, int((arrive_times[-1] - arrive_times[0])/binwidth))
            plt.plot(bin_edges[:-1], arrive_hist, label=f"model{id}")
        plt.title(self.workload_name)
        plt.xlabel("Time(s)")
        plt.xticks(np.arange(int(self.duration) + 1))
        plt.ylabel("Requests")
        plt.legend()
        if figname:
            plt.savefig(figname)
        else:
            plt.savefig(self.workload_name)
    
    def save(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename: str):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def run(self, callbacks, timers, tolerance: float = 0.005):
        """
        Run the workload with the given callbacks.
        @param callbacks: The list of callbacks, each corresponds to a model.
        @param tolerance: The tolerance error between workload request time and actual request time.
        """
        def wait_for_completion(loss_handle, request_id):
            _ = loss_handle._value
            timers(f"req{request_id}").stop()

        assert len(callbacks) == self.model_num
        now = start = time.time()
        model_ids, arrive_times = deepcopy(self.model_ids), deepcopy(self.arrive_times)
        next_id, next_arrive_time = model_ids.pop(0), arrive_times.pop(0)
        request_times = [] # for tolerance check
        waiting_threads = []
        request_id = 0
        # send requests
        while len(model_ids) > 0:
            if start + next_arrive_time <= now:
                request_times.append(now - start)
                timers(f"req{request_id}").start()
                loss = callbacks[next_id]()
                loss.get_remote_buffers_async()
                # spawn a thread to wait for completion
                wait_thread = threading.Thread(target=wait_for_completion, args=(loss, request_id))
                wait_thread.start()
                waiting_threads.append(wait_thread)
                next_id, next_arrive_time = model_ids.pop(0), arrive_times.pop(0)
                request_id += 1
            time.sleep(tolerance)
            now = time.time()
        
        # join the waiting threads
        for thd in waiting_threads:
            thd.join()

        # tolerance check
        for t_ref, t_req in zip(self.arrive_times, request_times):
            assert(t_ref - t_req <= tolerance)
        
        # save the result
        latencies = []
        for i in range(request_id):
            latencies.append(timers(f"req{i}").elapsed())
        with open(f"{self.workload_name}_latencies", 'wb') as f:
            pickle.dump(latencies, f)
        return latencies

def generate_workload(model_num: int, tot_arrival_rate: float, proportions: List[float], duration: float, workload_name: str = "PossoinWorkLoad"):
    workload = PossoinWorkLoad(model_num, tot_arrival_rate, proportions, duration, workload_name)
    workload.save(workload_name)
    return workload

def test_save_load():
    workload = generate_workload(2, 10, [0.5, 0.5], 20, "Even workload")
    workload.plot(0.5, "before save")
    del workload
    workload = PossoinWorkLoad.load("Even workload")
    workload.plot(0.5, "after load")

def test_run():
    workload = generate_workload(2, 10, [0.5, 0.5], 20, "Even workload")
    x = 1
    workload.run([lambda: x + 1]*2)

if __name__ == "__main__":
    even_workload = generate_workload(2, 10, [0.5, 0.5], 20, "Even_10Hz_20s")
    even_workload.plot(0.5)
    skew_workload = generate_workload(2, 10, [0.8, 0.2], 20, "Skewed8to2_10Hz_20s")
    skew_workload.plot(0.5)