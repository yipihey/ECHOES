"""NUMA-local process parallelism for embarrassingly-parallel ensembles.

On a many-NUMA-domain machine (this node: 2 sockets, **16 NUMA nodes × 8 cores / 128 GB**),
a single process threaded across all 128 cores spans 16 NUMA domains and pays a heavy
remote-memory penalty — measured to **saturate around ~32 threads** (a 64-thread GraphGP
generate is no faster than 32). The right shape for ensemble work (K field-gen realizations,
K clustering measurements, K completed-catalog draws) is therefore **one process per NUMA
node, pinned, each using only that node's cores** — every worker's data is first-touched
local, and the whole box is filled without cross-domain traffic.

:func:`numa_map` runs ``func`` over ``items`` across such NUMA-pinned workers. Each worker:
  * is CPU-pinned to one NUMA node's cores (``os.sched_setaffinity``) — first-touch keeps the
    worker's numpy arrays / Julia heap local to that node;
  * has ``OMP_NUM_THREADS`` / ``OPENBLAS_NUM_THREADS`` / ``JULIA_NUM_THREADS`` set to the
    per-node core count, so any nested BLAS / Corrfunc / Julia-subprocess it spawns stays on
    that node (this also sets ``OPENBLAS_NUM_THREADS`` low, avoiding the dense-Cholesky
    oversubscription segfault).

Falls back to a plain serial map if NUMA topology is unavailable or ``workers==1``.
"""
from __future__ import annotations

import glob
import os
import re


def numa_topology():
    """``{node_id: [cpu, ...]}`` from sysfs, or ``{}`` if unavailable."""
    nodes = {}
    for d in sorted(glob.glob("/sys/devices/system/node/node[0-9]*")):
        m = re.search(r"node(\d+)$", d)
        if not m:
            continue
        try:
            with open(os.path.join(d, "cpulist")) as fh:
                cpus = _parse_cpulist(fh.read().strip())
        except OSError:
            continue
        if cpus:
            nodes[int(m.group(1))] = cpus
    return nodes


def _parse_cpulist(s):
    """'0-7,16,20-21' -> [0,1,...,7,16,20,21]."""
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


# --- worker globals (set by the pool initializer, then used by _run) ---
_WORKER_FUNC = None


def _init_worker(func, node_cpus, threads):
    global _WORKER_FUNC
    _WORKER_FUNC = func
    try:
        os.sched_setaffinity(0, set(node_cpus))            # pin this worker to its NUMA node's cores
    except (AttributeError, OSError):
        pass
    t = str(int(threads))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "JULIA_NUM_THREADS"):
        os.environ[var] = t
    os.environ.setdefault("ECHOES_JULIA_THREADS", t)       # picked up by the Julia bridge (if wired)


def _run(arg):
    return _WORKER_FUNC(arg)


def numa_map(func, items, *, workers_per_node=1, threads_per_worker=None, nodes=None):
    """Apply ``func`` to each of ``items`` across NUMA-pinned worker processes, returning a
    list of results (input order). One worker per NUMA node by default; ``workers_per_node>1``
    oversubscribes each node (e.g. for latency-bound tasks). ``threads_per_worker`` defaults to
    ``floor(cores_per_node / workers_per_node)``. ``nodes`` restricts to a subset of node ids.

    Serial fallback when topology is unavailable, there is one item, or one worker is requested.
    """
    items = list(items)
    topo = numa_topology()
    if nodes is not None:
        topo = {n: topo[n] for n in nodes if n in topo}
    n_nodes = len(topo)

    if n_nodes <= 1 or len(items) <= 1:
        return [func(x) for x in items]

    import multiprocessing as mp

    node_ids = sorted(topo)
    # one logical worker slot per (node, replica); each slot pinned to its node's cpus
    slots = []  # (cpus, threads)
    for nid in node_ids:
        cpus = topo[nid]
        thr = threads_per_worker or max(1, len(cpus) // workers_per_node)
        for _ in range(workers_per_node):
            slots.append((cpus, thr))
    n_workers = min(len(slots), len(items))

    # We need each worker bound to a SPECIFIC slot. multiprocessing's initializer can't easily
    # give each worker a distinct arg, so we launch one single-process Pool per slot and feed it
    # a round-robin shard of items — that guarantees a fixed (worker ↔ node) pinning.
    shards = [items[i::n_workers] for i in range(n_workers)]
    procs_out = [None] * n_workers
    pools = []
    for w in range(n_workers):
        cpus, thr = slots[w]
        p = mp.get_context("fork").Pool(
            1, initializer=_init_worker, initargs=(func, cpus, thr))
        pools.append((p, p.map_async(_run, shards[w])))
    for w, (p, res) in enumerate(pools):
        procs_out[w] = res.get()
        p.close()
    for p, _ in pools:
        p.join()

    # un-shard back to original order
    out = [None] * len(items)
    for w in range(n_workers):
        for k, idx in enumerate(range(w, len(items), n_workers)):
            out[idx] = procs_out[w][k]
    return out
