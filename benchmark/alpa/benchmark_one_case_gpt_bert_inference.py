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

from util import compute_gpt_parameter_count, compute_gpt_tflops, dump_chrome_tracing, dump_demo_cluster_trace
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

BASELINE = False
def benchmark_gpt_inference_internal(model_type,
                                     benchmark_case,
                                     niter,
                                     num_hosts,
                                     num_devices_per_host,
                                     profile_driver_time=False):
    if BASELINE:
        return benchmark_baseline_demo(model_type, benchmark_case, num_hosts, num_devices_per_host)
    else:
        return benchmark_parallel_demo(model_type, benchmark_case, num_hosts, num_devices_per_host)

def benchmark_baseline_demo(model_type,
                            benchmark_case,
                            num_hosts,
                            num_devices_per_host):
    # get parallel method
    (method, _, add_manual_layer_marker,
     num_manual_pipeline_stages) = get_pipeshard_parallel_method(
         benchmark_case,
         num_devices_per_host,
         pipeline_schedule="inference")

    # Slice virtual_mesh1 for model1
    virtual_mesh1 = get_global_cluster().get_virtual_physical_mesh(
        host_ids=[0],
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
    exec1.mesh_group[0].devices=[[0]]
    # warmup for model1
    _ = infer_step1(params1, batch1, rngkey1)
    exec1.sync()

    # Slice virtual_mesh2 for model2
    virtual_mesh2 = get_global_cluster().get_virtual_physical_mesh(
        host_ids=[0],
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
    exec2.mesh_group[0].devices=[[1]]
    # warmup for model2
    _ = infer_step2(params1, batch1, rngkey1)
    exec2.sync()

    # no synchronization
    global_config.pipeline_check_alive = False

    # Load workload and Benchmark
    workload_name = "test_workload_8to2_6.667Hz_20s"
    workload = PossoinWorkLoad.load(workload_name)
    l0, l1 = workload.run([lambda: infer_step1(params1, batch1, rngkey1), 
                              lambda: infer_step2(params2, batch2, rngkey2)], 
                              timers)
    exec_info0 = exec1.get_execution_info()
    exec_info1 = exec2.get_execution_info()

    # sanity check
    assert len(exec_info0) == len(exec_info1) == num_manual_pipeline_stages

    # drop warmup's execution info
    for stage0_info, stage1_info in zip(exec_info0, exec_info1):
        stage0_info.pop(0), stage1_info.pop(0)
    
    print(f"trace_len:{len(l0)}, real_len:{len(exec_info0[0])}")
    print(f"trace_len:{len(l1)}, real_len:{len(exec_info1[0])}")

    # dump cluster traces
    trace_filename = f"{workload.workload_name}_baseline_trace.json"
    traces = dump_demo_cluster_trace(exec_info0, exec_info1, l0, l1, workload, trace_filename)

    # dump chrome trace for visualization
    chrome_trace_filename = f"./chrome_trace/{workload.workload_name}_baseline_chrometrace.json"
    dump_chrome_tracing(traces, chrome_trace_filename)
    print_used_time("Benchmark")
 
    # Compute statistics
    tflops, parameter_count = compute_gpt_inference_statistics(
        benchmark_case, l0, num_devices_per_host)
    metadata = {
        "latencies": l0,
        "compilation_times": 0,
    }
    return parameter_count, 0, l0, tflops, metadata


def benchmark_parallel_demo(model_type,
                            benchmark_case,
                            num_hosts,
                            num_devices_per_host):
    # Get parallel method
    (method, _, add_manual_layer_marker,
     num_manual_pipeline_stages) = get_pipeshard_parallel_method(
         benchmark_case,
         num_devices_per_host,
         pipeline_schedule="inference")

    # model1 and model2 share the same mesh
    virtual_mesh = get_global_cluster().get_virtual_physical_mesh(
        host_ids=list(range(num_hosts)),
        num_devices_per_host=num_devices_per_host)
    virtual_mesh.devices=[[1,0]]
    set_global_virtual_physical_mesh(virtual_mesh)

    # compile
    model1, params1, batch1, rngkey1 = prepare_gpt_inference_input_and_model(
        model_type, benchmark_case, add_manual_layer_marker,
        num_manual_pipeline_stages)
    model2, params2, batch2, rngkey2 = prepare_gpt_inference_input_and_model(
        model_type, benchmark_case, add_manual_layer_marker,
        num_manual_pipeline_stages)

    infer_step1 = get_infer_step(method, model1, model_type)
    infer_step2 = get_infer_step(method, model2, model_type)

    exec1, params1 = compile_pipeshard_inference_executable(
                        benchmark_case.parallel_mode,
                        infer_step1,
                        params1, (batch1, rngkey1))
    exec2, params2 = compile_pipeshard_inference_executable(
                        benchmark_case.parallel_mode,
                        infer_step2,
                        params2, (batch2, rngkey2))

    # warmup
    _ = infer_step1(params1, batch1, rngkey1)
    _ = infer_step1(params1, batch1, rngkey1)
    exec1.sync()
    exec2.sync()

    # no synchronization
    global_config.pipeline_check_alive = False

    # Load workload and Benchmark
    workload_name = "test_workload_8to2_6.667Hz_20s"
    workload = PossoinWorkLoad.load(workload_name)
    l0, l1 = workload.run([lambda: infer_step1(params1, batch1, rngkey1), 
                              lambda: infer_step2(params2, batch2, rngkey2)], 
                              timers)
    # model1 and model2 share the same meshgroup, so the timers logged all the requests
    exec_info = exec1.get_execution_info()

    # drop warmup's execution info
    for stage_info in exec_info:
        stage_info.pop(0), stage_info.pop(0)
 
    rq_ids0 = [i for i, model_id in enumerate(workload.model_ids) if model_id == 0]
    rq_ids1 = [i for i, model_id in enumerate(workload.model_ids) if model_id == 1]
    exec_info0 = [[stage_info[i] for i in rq_ids0] for stage_info in exec_info]
    exec_info1 = [[stage_info[i] for i in rq_ids1] for stage_info in exec_info]

   # sanity check
    assert len(exec_info0) == len(exec_info1) == num_manual_pipeline_stages

   
    # sanity check
    print(f"trace_len:{len(l0)}, real_len:{len(exec_info0[0])}")
    print(f"trace_len:{len(l1)}, real_len:{len(exec_info1[0])}")
    assert len(l0) == len(exec_info0[0])
    assert len(l1) == len(exec_info1[0])

    # dump cluster traces
    if num_manual_pipeline_stages > 1:
        trace_filename = f"{workload.workload_name}_interop_trace.json"
    else:
        trace_filename = f"{workload.workload_name}_intraop_trace.json"
    traces = dump_demo_cluster_trace(exec_info0, exec_info1, l0, l1, workload, trace_filename)

    # dump chrome trace for visualization
    if num_manual_pipeline_stages > 1:
        chrome_trace_filename = f"./chrome_trace/{workload.workload_name}_interop_chrometrace.json"
    else:
        chrome_trace_filename = f"./chrome_trace/{workload.workload_name}_intraop_chrometrace.json"
    dump_chrome_tracing(traces, chrome_trace_filename)
    print_used_time("Benchmark")
 
    # Compute statistics
    tflops, parameter_count = compute_gpt_inference_statistics(
        benchmark_case, l0, virtual_mesh.num_devices_per_host)
    metadata = {
        "latencies": l0,
        "compilation_times": 0,
    }
    return parameter_count, 0, l0, tflops, metadata


