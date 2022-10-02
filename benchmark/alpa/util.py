import json
import os
import time

import numpy as np

GB = 1 << 30


def write_tsv(heads, values, filename, print_line=True):
    """Write tsv data to a file."""
    assert len(heads) == len(values)

    values = [str(x) for x in values]

    with open(filename, "a") as fout:
        fout.write("\t".join(values) + "\n")

    if print_line:
        line = ""
        for i in range(len(heads)):
            line += heads[i] + ": " + values[i] + "  "
        print(line)


def benchmark_func(run_func, sync_func=None, warmup=1, repeat=3, number=5):
    """Benchmark the execution time of a function."""
    costs = []

    # Warmup
    for i in range(warmup):
        run_func()

    # Benchmark
    for i in range(repeat):
        if sync_func:
            sync_func()
        tic = time.time()

        for j in range(number):
            run_func()

        if sync_func:
            sync_func()
        costs.append(time.time() - tic)

    return np.array(costs) / number


def run_cmd(cmd):
    print(cmd)
    return os.system(cmd)


def get_torch_memory_usage(print_info=False):
    """Get accurate gpu memory usage by querying torch runtime"""
    import torch
    allocated = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)
    if print_info:
        print("allocated: %.2f GB" % (allocated / GB), flush=True)
        print("reserved:  %.2f GB" % (reserved / GB), flush=True)
    return allocated


def compute_gpt_tflops(batch_size,
                       seq_len,
                       num_layers,
                       hidden_size,
                       vocab_size,
                       num_gpus,
                       latency,
                       backward=True,
                       checkpoint_activations=False):
    factor = 24
    if backward:
        factor += 48
    if checkpoint_activations:
        factor += 24

    total_flop = factor * batch_size * seq_len * (hidden_size ** 2) * num_layers * \
          (1 + seq_len / (6 * hidden_size)) \
          + 6 * batch_size * seq_len * hidden_size * vocab_size
    # Note: The above formula does not count the first embedding table lookup
    # because it is a sparse operation.
    # If we use dense dot to compute the first embedding table lookup,
    # then the last term in total_flops should be
    # "+ 10 * batch_size * seq_len * hidden_size * vocab_size".
    tflops = total_flop / latency / num_gpus / 1e12
    return tflops


def compute_moe_tflops(batch_size,
                       seq_len,
                       num_layers,
                       hidden_size,
                       group_size,
                       vocab_size,
                       num_expert,
                       num_gpus,
                       latency,
                       mlp_factor=8,
                       checkpoint_activations=False):
    factor = 4 if checkpoint_activations else 3
    # num_layers / 2 attention block
    pure_transformer = batch_size * seq_len * (hidden_size ** 2) * (8 + 4 * mlp_factor) +\
        4 * batch_size * (seq_len ** 2) * hidden_size
    pure_transformer = pure_transformer * factor

    # num_layers / 2 attention-moe block
    # transformer
    moe_transformer = batch_size * seq_len * (hidden_size ** 2) * 8  +\
        4 * batch_size * (seq_len ** 2) * hidden_size
    # expert FFNs:
    # moe_transformer += 2 * batch_size * seq_len * (hidden_size ** 2) * mlp_factor * 2
    moe_transformer += 8 * batch_size * seq_len * (hidden_size**2) * mlp_factor

    # softmax
    moe_transformer += 2 * batch_size * seq_len * hidden_size * num_expert
    # top-2 gating
    moe_transformer += 2 * (batch_size * seq_len) * 2 * group_size
    # dispatch + combine
    moe_transformer += 2 * batch_size * seq_len * hidden_size * 2 * group_size * 2

    moe_transformer = moe_transformer * factor

    # vocab
    embedding = 6 * batch_size * seq_len * hidden_size * vocab_size

    total_flop = pure_transformer * num_layers / 2 + \
                 moe_transformer * num_layers / 2 + embedding
    tflops = total_flop / latency / num_gpus / 1e12
    return tflops


def compute_gpt_parameter_count(num_layers, hidden_size, vocab_size):
    return num_layers * (
        # self-attention
        hidden_size * (3 * hidden_size + 1) + hidden_size * (hidden_size + 1) +
        # mlp
        hidden_size * (4 * hidden_size + 1) + hidden_size * 4 *
        (hidden_size + 1) +
        # layer norm
        hidden_size * 4) + vocab_size * (hidden_size + 1)


def compute_moe_parameter_count(num_layers,
                                hidden_size,
                                vocab_size,
                                num_expert,
                                mlp_factor=8,
                                tie_embedding=True):
    pure_transformer = \
        hidden_size * (3 * hidden_size + 1) + hidden_size * (hidden_size + 1) + \
        hidden_size * (mlp_factor * hidden_size + 1) + hidden_size * mlp_factor * (hidden_size + 1) + \
        hidden_size * 4
    moe_transformer = \
        hidden_size * (3 * hidden_size + 1) + hidden_size * (hidden_size + 1) + \
        num_expert * (hidden_size * (mlp_factor * hidden_size + 1) + hidden_size * mlp_factor * (hidden_size + 1)) + \
        hidden_size * 4

    # embedding
    embedding_factor = 1 if tie_embedding else 2
    embedding = embedding_factor * vocab_size * (hidden_size + 1)

    if num_expert == 1:
        return pure_transformer * num_layers + embedding
    else:
        half = num_layers / 2
        return half * pure_transformer + half * moe_transformer + embedding


def dump_demo_cluster_trace(exec_info0, exec_info1, e2e_latency0, e2e_latency1, workload, trace_filename):
    # change absolute time to relative time
    start = min(exec_info0[0][0][0], exec_info1[0][0][0])
    exec_info0 = [[(s - start, e - start, node_ids, devices) for s, e, node_ids, devices in stage_info] for stage_info in exec_info0]
    exec_info1 = [[(s - start, e - start, node_ids, devices) for s, e, node_ids, devices in stage_info] for stage_info in exec_info1]
    rq_ids0 = [i for i, model_id in enumerate(workload.model_ids) if model_id == 0]
    rq_ids1 = [i for i, model_id in enumerate(workload.model_ids) if model_id == 1]
    arrive_0 = [t for i, t in zip(workload.model_ids, workload.arrive_times) if i == 0]
    arrive_1 = [t for i, t in zip(workload.model_ids, workload.arrive_times) if i == 1]
    latencies0 = [e - a for a, (_, e, _, _) in zip(arrive_0, exec_info0[-1])]
    latencies1 = [e - a for a, (_, e, _, _) in zip(arrive_1, exec_info1[-1])]
    # compare the e2e latency computed in driver with the one logged by the timers on meshhostworker
    # it should be around 10-20ms according to the communication && ray overhead 
    shift0 = [abs(x - y) for x, y in zip(latencies0, e2e_latency0)]
    shift1 = [abs(x - y) for x, y in zip(latencies1, e2e_latency1)]
    print(f"max shift: {max(max(shift0), max(shift1)):.5f}s")
    # dump cluster trace
    with open(trace_filename, 'w') as f:
        traces = {
                    0: {"rq_id": rq_ids0, 
                        "e2e_latency": e2e_latency0,
                        "arrive": arrive_0, 
                        "stage_exec_info": exec_info0},
                    1: {"rq_id": rq_ids1,
                        "e2e_latency": e2e_latency1,
                        "arrive": arrive_1, 
                        "stage_exec_info": exec_info1}
                }
        json.dump(traces, f)
    return traces


def dump_chrome_tracing(traces, dumpfile: str):
    color_list = [
        "thread_state_uninterruptible",
        "thread_state_iowait",
        "thread_state_running",
        "thread_state_runnable",
        "thread_state_unknown",
        "background_memory_dump",
        "light_memory_dump",
        "detailed_memory_dump",
        "vsync_highlight_color",
        "generic_work",
        "good",
        "bad",
        "terrible",
        "yellow",
        "olive",
        "rail_response",
        "rail_animation",
        "rail_idle",
        "rail_load",
        "startup",
        "heap_dump_stack_frame",
        "heap_dump_object_type",
        "heap_dump_child_node_arrow",
        "cq_build_running",
        "cq_build_passed",
        "cq_build_failed",
        "cq_build_attempt_runnig",
        "cq_build_attempt_passed",
        "cq_build_attempt_failed",
    ]
    def get_color(i):
        return color_list[i % len(color_list)]

    slot_list = []
    for model_id, trace in traces.items():
        for i, stage_exec_info in enumerate(trace["stage_exec_info"]):
            for rq_id, (s, e, node_ids, devices) in zip(trace["rq_id"], stage_exec_info):
                    for node_id, devices_per_node in zip(node_ids, devices):
                        for device in devices_per_node:
                            slot = {"name": f"r{rq_id}s{i}",
                                    "cat": f"model{model_id}",
                                    "ph": "X",
                                    "pid": node_id,
                                    "tid": device,
                                    "ts": float(s) * 1e6,
                                    "dur": float(e - s) * 1e6,
                                    "cname": get_color(rq_id)}
                            slot_list.append(slot)

    os.makedirs(os.path.dirname(dumpfile), exist_ok=True)
    with open(dumpfile, "w") as fout:
        fout.write(json.dumps({
            "traceEvents": slot_list,
            "displayTimeUnit": "ms",
        }))