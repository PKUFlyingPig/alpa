import pickle
import numpy as np
import matplotlib.pyplot as plt
from generate_workload import PossoinWorkLoad

#workload_filename = "Even_5Hz_20s"
#workload_filename = "Even_10Hz_20s"
#workload_filename = "Skewed8to2_5Hz_20s"
#workload_filename = "Skewed8to2_10Hz_20s"
workload_filename = "Skewed8to2_20Hz_20s"

def load_data():
    latency_filename = workload_filename + "_latencies_baseline"
    with open(latency_filename, "rb") as f:
        latencies = pickle.load(f)
    workload = PossoinWorkLoad.load(workload_filename)
    return workload, latencies

def plot_cdf(data1, data2, figname):
    plt.figure()
    # sort data
    x1, x2 = np.sort(data1), np.sort(data2)
    # calculate CDF values
    y1, y2 = 1. * np.arange(len(data1)) / (len(data1) - 1), 1. * np.arange(len(data2)) / (len(data2) - 1)
    # plot CDF
    plt.plot(x1, y1, label="model0")
    plt.plot(x2, y2, label="model1")
    plt.legend()
    plt.ylabel("CDF")
    plt.xlabel("latency(s)")
    plt.title("Latency CDF for " + workload_filename + "(Baseline)")
    # savefig
    plt.savefig(figname)

workload, latencies = load_data()
model0_latencies = [latencies[i] for i, model_id in enumerate(workload.model_ids) if model_id == 0]
model1_latencies = [latencies[i] for i, model_id in enumerate(workload.model_ids) if model_id == 1]
plot_cdf(model0_latencies, model1_latencies, workload_filename + "_cdf_baseline")
print(f"mean latency: {np.mean(latencies):.3f}s")
print(f"90% tail latency: {np.quantile(latencies, 0.9):.3f}s")
print(f"model0 mean latency: {np.mean(model0_latencies):.3f}s")
print(f"model0 90% tail latency: {np.quantile(model0_latencies, 0.9):.3f}s")
print(f"model1 mean latency: {np.mean(model1_latencies):.3f}s")
print(f"model1 90% tail latency: {np.quantile(model1_latencies, 0.9):.3f}s")