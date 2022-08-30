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
    parallel_latency_filename = workload_filename + "_latencies"
    baseline_latency_filename = workload_filename + "_latencies_baseline"
    with open(parallel_latency_filename, "rb") as f:
        parallel_latencies = pickle.load(f)
    with open(baseline_latency_filename, "rb") as f:
        baseline_latencies = pickle.load(f)
    workload = PossoinWorkLoad.load(workload_filename)
    return workload, parallel_latencies, baseline_latencies

def plot_cdf(latencies, is_baseline):
    model0_latencies = [latencies[i] for i, model_id in enumerate(workload.model_ids) if model_id == 0]
    model1_latencies = [latencies[i] for i, model_id in enumerate(workload.model_ids) if model_id == 1]
    # sort data
    x1, x2 = np.sort(model0_latencies), np.sort(model1_latencies)
    # calculate CDF values
    y1, y2 = 1. * np.arange(len(model0_latencies)) / (len(model0_latencies) - 1), 1. * np.arange(len(model1_latencies)) / (len(model1_latencies) - 1)
    # plot CDF
    if is_baseline:
        plt.plot(x1, y1, "--", color="c", label="model0 baseline")
        plt.plot(x2, y2, "--", color="orange", label="model1 baseline")
    else:
        plt.plot(x1, y1, "-", color="c", label="model0 parallel")
        plt.plot(x2, y2, "-", color="orange", label="model1 parallel")
    # print the statistics
    if is_baseline:
        print("---------------------------------------")
        print("Baseline latency statistics:")
    else:
        print("Parallel latency statistics:") 
    print(f"overall mean latency: {np.mean(latencies):.3f}s")
    print(f"overall 90% tail latency: {np.quantile(latencies, 0.9):.3f}s")
    print(f"model0 mean latency: {np.mean(model0_latencies):.3f}s")
    print(f"model0 90% tail latency: {np.quantile(model0_latencies, 0.9):.3f}s")
    print(f"model1 mean latency: {np.mean(model1_latencies):.3f}s")
    print(f"model1 90% tail latency: {np.quantile(model1_latencies, 0.9):.3f}s")

workload, parallel_latencies, baseline_latencies = load_data()
plt.figure()
plot_cdf(parallel_latencies, False)
plot_cdf(baseline_latencies, True)
plt.legend()
plt.ylabel("CDF")
plt.xlabel("Latency(s)")
plt.title("Latency CDF for " + workload_filename)

# savefig
plt.savefig(workload_filename + "_cdf_overall")


