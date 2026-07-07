# Gate-1 micro benchmark wrapper: run DeepEP's own tests/elastic/test_ep.py
# cross-node via torchrun (NOT mp.spawn — spawn overflows NCCL's 256-node XML
# topo cap on p5en). This is the exact wrapper that produced
# the 443.6us D+C PASS (2026-07-06); only the master address is
# parameterized (EP_BENCH_MASTER, required).
#
# Launch (node 0 and node 1):
#   python -m torch.distributed.run --nproc-per-node=8 --nnodes=2 --node-rank=<0|1> \
#     --rdzv-backend=c10d --rdzv-endpoint=<node0-ip>:29650 \
#     bench/micro/test_ep_torchrun_wrapper.py
# with EP_BENCH_MASTER=<node0-ip> and the common serving env (REPRODUCE.md §2)
# plus EP_EFA_MAX_QPS=6 (bench contract config, not the serving value 2).
#
# Two traps this file encodes (each cost a debugging session):
# 1. test_ep's init_dist env contract is WORLD_SIZE=<num NODES>, RANK=<node rank>
#    (mp.spawn convention). torchrun sets WORLD_SIZE=<total procs>, RANK=<global
#    rank>. Passing torchrun's values through makes init_process_group wait for
#    8*8=64 ranks — an infinite stall. Normalize BEFORE calling test_loop.
# 2. MASTER_PORT must EQUAL the torchrun rdzv port: TORCHELASTIC_USE_AGENT_STORE
#    =True makes every rank a TCPStore CLIENT of the agent's store. A different
#    port means nobody binds it -> infinite connect-retry (observed: 16 procs
#    parked in hrtimer_nanosleep, no listener).
import os, sys, argparse

lws = int(os.environ.get("LOCAL_WORLD_SIZE", "8"))
node_rank = int(os.environ.get("GROUP_RANK", str(int(os.environ["RANK"]) // lws)))
os.environ["WORLD_SIZE"] = os.environ.get("GROUP_WORLD_SIZE", "2")   # num NODES
os.environ["RANK"] = str(node_rank)                                   # node rank
os.environ["MASTER_ADDR"] = os.environ["EP_BENCH_MASTER"]
os.environ["MASTER_PORT"] = os.environ.get("EP_BENCH_RDZV_PORT", "29650")

sys.path.insert(0, os.environ.get("DEEPEP_TESTS_DIR", "/opt/DeepEP/tests/elastic"))
import test_ep

# Mirrors bench/bench_v2_only.sh (pr521gin ~740us contract config), first case
# only (full enumerate_ep_modes x JIT >30min; case 1 gives the D+C perf lines).
ns = argparse.Namespace(
    num_processes=lws, num_sms=4, num_qps=6, num_allocated_qps=6,
    num_gpu_timeout_secs=100, num_cpu_timeout_secs=100, sl_idx=0,
    num_tokens=128, hidden=7168, num_topk=8, num_experts=256,
    do_cpu_sync=1, allow_hybrid_mode=1, allow_multiple_reduction=1,
    prefer_overlap_with_compute=0, deterministic=False, seed=0,
    skip_check=True, skip_perf_test=False, do_pressure_test=False,
    reuse_elastic_buffer=False, test_first_only=True,
    unbalanced_ratio=1.0, precise_unbalanced_ratio=False, masked_ratio=0.0,
    dump_profile_traces="", ignore_local_traffic=True,
)
lr = int(os.environ["LOCAL_RANK"])
print(f"[micro] node{node_rank} lr{lr} nodes={os.environ['WORLD_SIZE']} x {lws} "
      f"store={os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}", flush=True)
test_ep.test_loop(lr, lws, ns)
print(f"[micro] node{node_rank} lr{lr} DONE", flush=True)
