"""Benchmark one case of inter-op + intra-op parallelism."""
import json

import jax
import jax.numpy as jnp
import numpy as np

from alpa import (parallelize, get_global_cluster,
                  set_global_virtual_physical_mesh, global_config)
from alpa.model.bert_model import BertConfig, FlaxBertLayerCollection
from alpa.model.gpt_model import FlaxGPTForLMModule
from alpa.timer import timers
from alpa.util import print_used_time

from util import compute_gpt_parameter_count, compute_gpt_tflops
from benchmark_parallel_utils import (
    get_pipeshard_parallel_method,
    compile_and_benchmark_pipeshard_inference_executable,
    compile_pipeshard_inference_executable)
from generate_workload import PossoinWorkLoad

def create_infer_params_aval(rngkey, model, batch, model_type):
    if model_type == "gpt_no_embedding_inference":
        params = jax.eval_shape(model.init, rngkey, batch["x"],
                                batch["attention_mask"])
    elif model_type == "gpt_inference":
        params = jax.eval_shape(model.init, rngkey, batch["input_ids"],
                                batch["attention_mask"],
                                batch["token_type_ids"], batch["position_ids"])
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    params = jax.eval_shape(
        lambda p: jax.tree_util.tree_map(
            lambda x: jnp.asarray(x, dtype=jnp.float16), p), params)
    return params


def get_infer_step(parallel_method, model, model_type):

    def infer_step_with_embedding(params, batch, rng_key):
        rngs = {"dropout": rng_key}
        logits = model.apply(params,
                             batch["input_ids"],
                             batch["attention_mask"],
                             batch["token_type_ids"],
                             batch["position_ids"],
                             deterministic=True,
                             rngs=rngs)[0]
        label_mask = jnp.where(batch["labels"] > 0, 1.0, 0.0)
        labels = jax.nn.one_hot(batch["labels"], logits.shape[-1])
        loss = -jnp.sum(labels * jax.nn.log_softmax(logits, axis=-1), axis=-1)
        loss = (label_mask * loss).sum() / label_mask.sum()
        return loss

    def infer_step_without_embedding(params, batch, rng_key):
        out = model.apply(params,
                          batch["x"],
                          batch["attention_mask"],
                          output_attentions=True,
                          output_hidden_states=True)
        loss = jnp.mean((out.last_hidden_state - batch["y"])**2)
        return loss

    if model_type == "gpt_no_embedding_inference":
        infer_step = infer_step_without_embedding
    elif model_type == "gpt_inference":
        infer_step = infer_step_with_embedding
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    return parallelize(infer_step, method=parallel_method, donate_argnums=())


def prepare_gpt_inference_input_and_model(model_type,
                                          benchmark_case,
                                          add_manual_layer_marker=None,
                                          num_manual_pipeline_stages=None,
                                          tie_word_embeddings=False):
    print_used_time(None)
    batch_size = benchmark_case.batch_size
    (seq_len, hidden_size, num_layers, num_heads,
     vocab_size) = benchmark_case.model_config
    dtype = jnp.float16

    bert_config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        intermediate_size=hidden_size * 4,
        num_hidden_layers=num_layers,
        type_vocab_size=0,
        tie_word_embeddings=tie_word_embeddings,
        add_manual_pipeline_markers=add_manual_layer_marker,
        pipeline_mp_size=num_manual_pipeline_stages,
    )

    # Init train state
    if model_type == "gpt_no_embedding_inference":
        batch = {
            "x": jnp.ones((batch_size, seq_len, hidden_size), dtype=dtype),
            "y": jnp.ones((batch_size, seq_len, hidden_size), dtype=dtype),
            "attention_mask": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        }
        model = FlaxBertLayerCollection(bert_config, dtype=dtype)
    elif model_type == "gpt_inference":
        batch = {
            "input_ids": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
            "attention_mask": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
            "token_type_ids": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
            "position_ids": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
            "labels": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        }

        model = FlaxGPTForLMModule(bert_config, dtype=dtype)
    else:
        raise ValueError(f"Invalid model {model_type}")

    rngkey = jax.random.PRNGKey(0)
    params = create_infer_params_aval(rngkey, model, batch, model_type)
    print_used_time("Create infer state")
    return model, params, batch, rngkey


def compute_gpt_inference_statistics(benchmark_case, latencies, num_devices):
    batch_size = benchmark_case.batch_size
    (seq_len, hidden_size, num_layers, num_heads,
     vocab_size) = benchmark_case.model_config
    use_remat = benchmark_case.parallel_args.use_remat

    tflops = compute_gpt_tflops(batch_size,
                                seq_len,
                                num_layers,
                                hidden_size,
                                vocab_size,
                                num_devices,
                                np.mean(latencies),
                                backward=False)
    parameter_count = compute_gpt_parameter_count(num_layers, hidden_size,
                                                  vocab_size)
    return tflops, parameter_count

BASELINE = True
def benchmark_gpt_inference_internal(model_type,
                                     benchmark_case,
                                     niter,
                                     num_hosts,
                                     num_devices_per_host,
                                     profile_driver_time=False):
    if BASELINE:
        benchmark_baseline_demo(model_type, benchmark_case, num_hosts, num_devices_per_host)
    else:
        pass

def benchmark_baseline_demo(model_type,
                            benchmark_case,
                            num_hosts,
                            num_devices_per_host):
    (method, _, add_manual_layer_marker,
     num_manual_pipeline_stages) = get_pipeshard_parallel_method(
         benchmark_case,
         num_devices_per_host,
         pipeline_schedule="inference")

    # Slice virtual_mesh1 for model1
    virtual_mesh1 = get_global_cluster().get_virtual_physical_mesh(
        host_ids=list(range(num_hosts)),
        num_devices_per_host=num_devices_per_host)
    set_global_virtual_physical_mesh(virtual_mesh1)
    # compile
    model1, params1, batch1, rngkey1 = prepare_gpt_inference_input_and_model(
        model_type, benchmark_case, add_manual_layer_marker,
        num_manual_pipeline_stages)
    infer_step1 = get_infer_step(method, model1, model_type)
    exec1, params1 = compile_pipeshard_inference_executable(
                        benchmark_case.parallel_mode,
                        infer_step1,
                        params1, (batch1, rngkey1))
    # warmup for model1
    _ = infer_step1(params1, batch1, rngkey1)
    exec1.sync()

    # Slice virtual_mesh2 for model2
    virtual_mesh2 = get_global_cluster().get_virtual_physical_mesh(
        host_ids=list(range(num_hosts)),
        num_devices_per_host=num_devices_per_host)
    set_global_virtual_physical_mesh(virtual_mesh2)
    # compile
    model2, params2, batch2, rngkey2 = prepare_gpt_inference_input_and_model(
        model_type, benchmark_case, add_manual_layer_marker,
        num_manual_pipeline_stages)
    infer_step2 = get_infer_step(method, model2, model_type)
    exec2, params2 = compile_pipeshard_inference_executable(
                        benchmark_case.parallel_mode,
                        infer_step2,
                        params2, (batch2, rngkey2))
    # warmup for model2
    _ = infer_step2(params1, batch1, rngkey1)
    exec2.sync()

    # no synchronization
    global_config.pipeline_check_alive = False

    # Load workload and Benchmark
    # workload_name = "test_workload_8to2_10Hz_20s"
    workload_name = "test_workload_8to2_6.667Hz_20s"
    workload = PossoinWorkLoad.load(workload_name)
    l0, l1 = workload.run([lambda: infer_step1(params1, batch1, rngkey1), 
                              lambda: infer_step2(params2, batch2, rngkey2)], 
                              timers)
    start_times1, stop_times1 = exec1.get_execution_timestamps()
    start_times2, stop_times2 = exec2.get_execution_timestamps()
    # Baseline config has only one pipeline stage
    assert len(start_times1) == len(stop_times1) == 1
    s0, e0 = start_times1[0], stop_times1[0]
    s1, e1 = start_times2[0], stop_times2[0]
    # drop timestamps for warmup
    s0.pop(0)
    e0.pop(0)
    s1.pop(0)
    e1.pop(0)
    print(f"trace_len:{len(l0)}, real_len:{len(s0)}")
    print(f"trace_len:{len(l1)}, real_len:{len(s1)}")
    start_timestamp = min(s0[0], s1[0])
    s0 = [t - start_timestamp for t in s0]
    e0 = [t - start_timestamp for t in e0]
    s1 = [t - start_timestamp for t in s1]
    e1 = [t - start_timestamp for t in e1]
    rq_ids0 = [i for i, model_id in enumerate(workload.model_ids) if model_id == 0]
    rq_ids1 = [i for i, model_id in enumerate(workload.model_ids) if model_id == 1]
    arrive_0 = [t for i, t in zip(workload.model_ids, workload.arrive_times) if i == 0]
    arrive_1 = [t for i, t in zip(workload.model_ids, workload.arrive_times) if i == 1]
    latencies0 = [e - a for a, e in zip(arrive_0, e0)]
    latencies1 = [e - a for a, e in zip(arrive_1, e1)]
    # compare the e2e latency experienced in driver with the one logged by the timers on meshhostwork
    shift0 = [abs(x - y) for x, y in zip(latencies0, l0)]
    shift1 = [abs(x - y) for x, y in zip(latencies1, l1)]
    print(max(shift0))
    print(max(shift1))
    # dump for comparison with simulator
    with open(f"{workload.workload_name}_baseline_trace.json", 'w') as f:
            json.dump({0: {"rq_id": rq_ids0, "arrive": arrive_0, "start": s0, "end": e0, "latency": l0}, 
                       1: {"rq_id": rq_ids1, "arrive": arrive_1, "start": s1, "end": e1, "latency": l1}}, f)
    print_used_time("Benchmark")
   
    # Compute statistics
    tflops, parameter_count = compute_gpt_inference_statistics(
        benchmark_case, latencies0, num_devices_per_host)
    metadata = {
        "latencies": latencies0,
        "compilation_times": 0,
    }
    return parameter_count, 0, latencies0, tflops, metadata

