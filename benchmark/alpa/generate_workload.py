from copy import deepcopy
import csv
from typing import List
import numpy as np
from numpy.random import choice, exponential
import matplotlib.pyplot as plt
import pickle
import time
import threading

class WorkLoad:
    """
    Abstract class for DL Workload.
    """
    def __init__(self, model_num: int, duration: float, SLOs: List[float], workload_name: str):
        """
            @param model_num: the number of different models in this workload.
            @param duration: the duration of the workload, measured in second.
            @param SLOs: the SLO for each model, measured in second.
        """
        assert duration > 0
        self.model_num = model_num
        self.duration = duration
        self.SLOs = SLOs
        self.workload_name = workload_name

        # attributes below are initialized by subclass
        # model id for each request
        self.model_ids = []   
        # arrival time for each request
        self.arrive_times = []

    def plot(self, binwidth: float = 1, figname: str = None):
        """
        Plot the workload.
            x-axis: arrival time
            y-axis: number of arrival tasks per binwidth (in seconds)
        """
        plt.figure()
        for id in range(self.model_num):
            arrive_times = [self.arrive_times[i] for i, model_id in enumerate(self.model_ids) if model_id == id]
            arrive_hist, bin_edges = np.histogram(arrive_times, int(self.duration/binwidth))
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
        model_ids, arrive_times = deepcopy(self.model_ids), deepcopy(self.arrive_times)
        next_id, next_arrive_time = model_ids.pop(0), arrive_times.pop(0)
        request_times = [] # for tolerance check
        waiting_threads = []
        request_id = 0
        now = start = time.time()
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
        # handle the last request
        # timers(f"req{request_id}").start()
        # loss = callbacks[next_id]()
        # loss.get_remote_buffers_async()
        wait_thread = threading.Thread(target=wait_for_completion, args=(loss, request_id))
        
        # join the waiting threads
        for thd in waiting_threads:
            thd.join()

        # tolerance check
        for t_ref, t_req in zip(self.arrive_times, request_times):
            assert(t_ref - t_req <= tolerance)
        
        # save the result
        latencies = []
        for i in range(request_id + 1):
            latencies.append(timers(f"req{i}").elapsed())
        with open(f"{self.workload_name}_latencies_baseline", 'wb') as f:
            pickle.dump(latencies, f)
        return latencies


class PossoinWorkLoad(WorkLoad):
    def __init__(self, model_num: int, tot_arrival_rate: float, proportions: List[float], duration: float, 
                       SLOs: List[float], workload_name: str = "PossoinWorkLoad", model_ids: List[int] = None, arrive_times: List[float] = None):
        """
            @param model_num: the number of different models in this workload.
            @param tot_arrival_rate: The total arrival rate of the requests for 
                                     all the models, measured in Hz.
            @param proportions: the proportion of the requests for each model 
            @param duration: the duration of the workload, measured in Second.
            @param SLOs: the SLO for each model, measured in second.
            @param model_ids: the model id for each request, if not given, generate during initialization.
            @param arrive_times: the arrival time for each request, if not given, generate during initialization.
        """
        super().__init__(model_num, duration, SLOs, workload_name)
        assert model_num == len(proportions) == len(SLOs) and sum(proportions) == 1 and tot_arrival_rate > 0
        self.tot_arrival_rate = tot_arrival_rate
        self.proportions = proportions

        if model_ids is None:
            assert arrive_times is None
            rela_time = 0.0
            while rela_time < duration:
                self.model_ids.append(choice(model_num, p=proportions))
                self.arrive_times.append(rela_time)
                rela_time += exponential(1/tot_arrival_rate)
        else:
            self.model_ids = model_ids
            self.arrive_times = arrive_times
        print(f"There are {len(self.arrive_times)} requests in total.")

    def save(self, filename: str):
        """
        Save the workload to a csv file.
            First row is metadata. (model_num, tot_arrival_rate, proportions, duration, SLOs, workload_name)
            From second row to the end are requests info. (model_id, arrive_timestamp)
        """
        with open(filename, 'w', newline='') as workload_file:
            writer = csv.writer(workload_file)
            # metadata
            writer.writerow([self.model_num, self.tot_arrival_rate, self.proportions, self.duration, self.SLOs, self.workload_name])
            # request model id, arrival timestamp
            for model_id, arrive_time in zip(self.model_ids, self.arrive_times):
                writer.writerow([model_id, arrive_time])

    @classmethod
    def load(cls, filename: str):
        """
        Load the workload from a csv file.
            First row is metadata. (model_num, tot_arrival_rate, proportions, duration, SLOs, workload_name)
            From second row to the end are requests info. (model_id, arrive_timestamp)
        """
        with open(filename, 'r', newline='') as workload_file:
            reader = csv.reader(workload_file)
            # first row is metadata
            model_num, tot_arrival_rate, proportions, duration, SLOs, workload_name = next(reader)
            # request model id, arrival timestamp
            model_ids, arrive_times = [], []
            for model_id, arrive_time in reader:
                model_ids.append(int(model_id))
                arrive_times.append(float(arrive_time))
            return cls(int(model_num), float(tot_arrival_rate), eval(proportions), float(duration), eval(SLOs), workload_name, model_ids, arrive_times)
    
def generate_workload(model_num: int, tot_arrival_rate: float, proportions: List[float], duration: float, SLOs: List[float], workload_name: str = "PossoinWorkLoad"):
    workload = PossoinWorkLoad(model_num, tot_arrival_rate, proportions, duration, SLOs, workload_name)
    workload.save(f"{workload_name}")
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
    # even_workload = generate_workload(2, 10, [0.5, 0.5], 20, "Even_10Hz_20s")
    # even_workload.plot(0.5)
    # skew_workload = generate_workload(2, 10, [0.8, 0.2], 20, "Skewed8to2_10Hz_20s")
    # skew_workload.plot(0.5)
    skew_workload = generate_workload(2, 10, [0.8, 0.2], 20, [0.25, 0.25], "test_workload_8to2_10Hz_20s")