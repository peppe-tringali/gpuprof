"""User-facing profiler.

Key concepts:

- **Step context** wraps one training step; sub-phases are declared
  with `s.phase("name")` and can nest.
- **Inter-step gap** is measured automatically — the time from the
  previous step's exit to this step's entry. That captures the *real*
  dataloader stall in a `for batch in loader:` loop (where the wait
  happens between step contexts, not inside one). The explicit
  `dataloader_wait` phase is still supported for loops that put the
  batch fetch inside the step context.
- **Comm phase** for distributed training. Either wrap explicitly
  (`s.phase("comm")`) or call `prof.instrument_ddp(model)` to have
  DDP's allreduce time attributed automatically.
- **Warmup steps** (0, 1, 5) are always traced when tracing is on,
  even between the periodic captures, so compilation and first-step
  outliers get a kernel view.
- **FLOPs** are auto-computed from a declared architecture when the
  user records `tokens` — see `gpuprof.flops.TransformerArch`.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from typing import Optional

from .flops import TransformerArch, arch_from_meta, transformer_flops_per_step
from .host_sampler import HostSample, HostSampler
from .sampler import Sample, Sampler
from .store import Store


class GpuProfiler:
    def __init__(
        self,
        run_name: str,
        db_path: Optional[str] = "gpuprof.db",
        gpu_indices: Optional[list[int]] = None,
        sample_hz: float = 10.0,
        meta: Optional[dict] = None,
        cuda: bool = False,
        trace_every_n_steps: int = 0,
        trace_warmup_steps: tuple[int, ...] = (0, 1, 5),
        trace_range: Optional[tuple[int, int]] = None,
        continuous_traces_hz: float = 0.0,
        cuda_event_comm: bool = True,
        # OFF by default: `dist.recv` inside the ping-pong is blocking
        # with no timeout, so a single crashed peer would freeze the
        # `profile()` context before training even starts. Opt in
        # when you need sub-ms cross-rank precision and know the
        # cluster is stable.
        estimate_clock_offset: bool = False,
        clock_offset_pings: int = 64,
        host_sampling: bool = True,
        host_sample_hz: float = 1.0,
        server_url: Optional[str] = None,
        api_key: Optional[str] = None,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        group_id: Optional[str] = None,
    ):
        self.run_name = run_name
        self._meta = dict(meta or {})
        self._cuda = cuda
        # NVML polling is generally reliable to ~50 Hz; we allow up to
        # 100 for short bursts, capped to protect the training thread.
        self._sample_hz = min(max(1.0, float(sample_hz)), 100.0)
        self._trace_every = int(trace_every_n_steps)
        self._trace_warmup = set(trace_warmup_steps or ())
        self._trace_range = trace_range  # (start, end) — trace every step in [start, end)
        self._continuous_hz = float(continuous_traces_hz)
        self._continuous: Optional[object] = None  # ContinuousProfiler
        self._cuda_event_comm = bool(cuda_event_comm)
        self._estimate_offset = bool(estimate_clock_offset)
        self._clock_offset_pings = int(clock_offset_pings)

        # Distributed identity — either explicit args or picked up from
        # torch.distributed if the caller already initialized it.
        self._rank = rank
        self._world_size = world_size
        self._group_id = group_id
        self._maybe_read_dist_env()

        # Architecture for auto-FLOP calc (safe if meta has no arch).
        self._arch: Optional[TransformerArch] = arch_from_meta(self._meta)

        self._local: Optional[Store] = Store(db_path) if db_path else None
        server_url = server_url or os.environ.get("GPUPROF_SERVER")
        api_key = api_key or os.environ.get("GPUPROF_API_KEY")
        self._remote = None
        if server_url:
            from .remote import Remote
            self._remote = Remote(server_url, api_key=api_key)

        self._sampler = Sampler(
            on_sample=self._on_sample,
            gpu_indices=gpu_indices,
            hz=self._sample_hz,
        )
        # Host sampler is a no-op if psutil isn't installed — it just
        # never emits, and the CPU-bound / I/O-bound rules quietly
        # don't fire. Enabled by default because the cost is tiny.
        self._host_sampler = (
            HostSampler(on_sample=self._on_host_sample, hz=host_sample_hz)
            if host_sampling else None
        )
        self._run_id: Optional[int] = None
        self._prev_step_end: Optional[float] = None
        self._current_step: Optional[_StepRecorder] = None
        # Stashed by `wrap_dataloader` — the last time `next(loader)`
        # blocked. Applied to the next step's `dataloader_wait_s`
        # phase so we can distinguish "prefetch queue was empty" from
        # inter-step gaps that come from other causes.
        self._pending_prefetch_wait_s: float = 0.0
        # Optional per-step + end-of-run callbacks. Used by the W&B
        # adapter, webhook alerts, etc. — anything that wants to see
        # data as it flows without wiring another sink into the store.
        self._on_step_cbs: list = []
        self._on_end_cbs: list = []
        # perf_counter() at the moment of a `dist.barrier()` sync, or
        # profiler.start() otherwise. Cross-rank comm events are
        # reported as (t_local - _epoch) so ranks are comparable.
        self._epoch: Optional[float] = None
        # CUDA event recorded at the same moment as `_epoch`. Used as
        # the anchor for converting per-bucket CUDA-event timestamps
        # (device-clock, sub-μs) into t_rel form.
        self._epoch_cuda_event = None

    # -- distributed helpers ------------------------------------------

    def _maybe_read_dist_env(self) -> None:
        """Pick up rank/world_size from torch.distributed if not passed."""
        if self._rank is not None and self._world_size is not None:
            return
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                if self._rank is None: self._rank = dist.get_rank()
                if self._world_size is None: self._world_size = dist.get_world_size()
        except Exception:
            pass
        # Fall back to torchrun env vars.
        if self._rank is None and "RANK" in os.environ:
            try: self._rank = int(os.environ["RANK"])
            except ValueError: pass
        if self._world_size is None and "WORLD_SIZE" in os.environ:
            try: self._world_size = int(os.environ["WORLD_SIZE"])
            except ValueError: pass
        # torchrun sets TORCHELASTIC_RUN_ID — unique per launch, same
        # across every rank of that launch. Perfect group id.
        if self._group_id is None:
            self._group_id = os.environ.get("TORCHELASTIC_RUN_ID") or None

    def instrument_ddp(self, ddp_model) -> None:
        """Register a comm hook so DDP allreduce time is attributed to
        each step's `comm` phase and to per-bucket `comm_events` for
        cross-rank correlation.

        When `cuda_event_comm=True` (the default) and CUDA is
        available, `torch.cuda.Event` timestamps replace
        `time.perf_counter()` — sub-microsecond precision per rank
        vs. millisecond-precision wall-clock. Combined with the
        NCCL-ping clock-offset estimation done in `start()`, cross-
        rank correlation drops from ~1 ms floor to ~10-100 μs.
        """
        from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as h
        prof = self
        use_cuda_events = prof._cuda_event_comm
        try:
            import torch
            if not torch.cuda.is_available():
                use_cuda_events = False
            elif use_cuda_events and prof._epoch_cuda_event is None:
                # Record an anchor event at the sync-epoch so we can
                # convert each CUDA event's timestamp to a t_rel via
                # start_ev.elapsed_time — that call gives ms between
                # the two events on the GPU global timer.
                anchor = torch.cuda.Event(enable_timing=True)
                anchor.record()
                prof._epoch_cuda_event = anchor
        except Exception:
            use_cuda_events = False

        def wrapping_hook(state, bucket):
            try: bidx = int(bucket.index())
            except Exception: bidx = -1

            # Capture the current step recorder *now*, in the hook body,
            # before the future's callback runs. `_on_done` fires on the
            # NCCL callback thread; reading `prof._current_step` from
            # there races with the training thread's `finally` block
            # (which sets `_current_step = None` at step exit). A late
            # callback would silently drop the comm event. Closing over
            # `rec_snapshot` ties this event to the step that produced it.
            rec_snapshot = prof._current_step

            if use_cuda_events:
                import torch
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                start_ev.record()
            else:
                start = time.perf_counter()
                start_ev = end_ev = None

            fut = h.allreduce_hook(state, bucket)

            def _on_done(f):
                rec = rec_snapshot
                if rec is None:
                    return f.value()
                if use_cuda_events and start_ev is not None:
                    # Defer resolution: syncing here would stall the
                    # comm callback. Hand the pair to the recorder
                    # so it resolves alongside phase events at step exit.
                    end_ev.record()
                    rec.add_pending_cuda_comm_event(
                        bidx, start_ev, end_ev,
                        prof._epoch_cuda_event,
                    )
                else:
                    end = time.perf_counter()
                    rec.add_comm_time(end - start)
                    if bidx >= 0:
                        rec.add_comm_event(
                            bucket_id=bidx,
                            t_start_rel=start - prof._epoch,
                            t_end_rel=end - prof._epoch,
                        )
                return f.value()

            return fut.then(_on_done)

        ddp_model.register_comm_hook(state=None, hook=wrapping_hook)

    @contextmanager
    def nsys_capture(self):
        """Bracket a code region for nsys capture. Use with
            nsys profile --capture-range=cudaProfilerApi \\
                --capture-range-end=stop -o out.nsys-rep python train.py
        The generated `.nsys-rep` covers exactly the wrapped region.
        Then:
            nsys export --type sqlite --output out.sqlite out.nsys-rep
            python -m gpuprof.nsys_import out.sqlite --gpuprof-db ... --run-id ...
        adds the CUPTI kernel timeline into gpuprof's `trace_windows`
        table for this run.
        """
        from .nsys import nsys_capture as _nc
        with _nc() as active:
            yield active

    # -- lifecycle ----------------------------------------------------

    @property
    def gpu_name(self) -> str: return self._sampler.gpu_name

    @property
    def gpu_indices(self) -> list[int]: return self._sampler.gpu_indices

    def start(self) -> int:
        # Advertise multi-GPU + distributed layout so the dashboard
        # knows what to plot and how to group.
        self._meta.setdefault("gpu_indices", self._sampler.gpu_indices)
        if self._rank is not None:
            self._meta.setdefault("rank", self._rank)
        if self._world_size is not None:
            self._meta.setdefault("world_size", self._world_size)

        # Sync-epoch: for a distributed run, barrier so every rank's
        # perf_counter starts in lockstep (~ms precision). Cross-rank
        # comm-event correlation works off this shared t=0.
        self._epoch = time.perf_counter()
        rank_offset_s: Optional[float] = None
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
                self._epoch = time.perf_counter()
                if self._estimate_offset and self._world_size and self._world_size > 1:
                    # NCCL ping-pong estimates each rank's clock offset
                    # from rank 0 to ~10-100 μs precision — an order of
                    # magnitude better than the 1 ms barrier alone.
                    rank_offset_s = self._estimate_clock_offset_via_pings()
        except Exception:
            pass

        gpu = self._sampler.gpu_name
        if self._local:
            self._run_id = self._local.start_run(
                name=self.run_name, gpu_name=gpu,
                meta_json=json.dumps(self._meta),
                group_id=self._group_id,
                rank=self._rank, world_size=self._world_size,
            )
        if self._remote:
            remote_id = self._remote.start_run(
                name=self.run_name, gpu_name=gpu, meta=self._meta,
                group_id=self._group_id, rank=self._rank,
                world_size=self._world_size,
            )
            if self._run_id is None:
                self._run_id = remote_id
        self._sampler.start()
        if self._host_sampler is not None:
            self._host_sampler.start()

        # Persist the clock offset on the run record (server-side or
        # local SQLite) so the analyzer can subtract it later.
        if rank_offset_s is not None:
            if self._local:
                try: self._local.set_rank_offset(rank_offset_s)
                except Exception: pass
            if self._remote:
                try: self._remote.set_rank_offset(rank_offset_s)
                except Exception: pass

        # Kick off the continuous kernel-aggregate profiler if enabled.
        if self._continuous_hz > 0:
            from .continuous import ContinuousProfiler
            self._continuous = ContinuousProfiler(
                on_window=self._on_trace_window,
                window_s=1.0 / self._continuous_hz,
                cuda=self._cuda,
                epoch=self._epoch,
            )
            self._continuous.start()

        return self._run_id or 0

    def stop(self) -> None:
        self._sampler.stop()
        if self._host_sampler is not None:
            self._host_sampler.stop()
        if self._continuous is not None:
            self._continuous.stop()
        if self._local:
            self._local.end_run()
        if self._remote:
            self._remote.end_run()
        for cb in self._on_end_cbs:
            try: cb()
            except Exception: pass

    def on_step(self, fn) -> None:
        """Register a per-step callback. `fn(step_dict)` fires after
        each step commits, with the step's fields as a dict."""
        self._on_step_cbs.append(fn)

    def on_end(self, fn) -> None:
        """Register an end-of-run callback. `fn()` fires after the
        run is fully closed (samplers stopped, DB flushed)."""
        self._on_end_cbs.append(fn)

    # -- fan-out helper: every event goes to both local SQLite and the
    # -- remote pusher when either is configured. Kept as a method so
    # -- adding a new sink is one place, not five.
    def _dispatch(self, method_name: str, payload) -> None:
        if self._local is not None:
            getattr(self._local, method_name)(payload)
        if self._remote is not None:
            getattr(self._remote, method_name)(payload)

    def _on_trace_window(self, window: dict) -> None:
        self._dispatch("push_trace_window", window)

    # ---- cross-rank clock alignment ---------------------------------

    def _estimate_clock_offset_via_pings(self) -> float:
        """Ping-pong NCCL between this rank and rank 0 to estimate a
        clock offset (this rank - rank 0) with median-of-RTT/2.

        Precision: typically 10-100 μs on a healthy fabric — an order
        of magnitude tighter than the 1 ms `dist.barrier()` alone but
        still bounded by scheduler jitter + NIC queueing. Real
        sub-microsecond sync needs PTP hardware timestamps.

        Rank 0 returns 0.0 (it defines the reference clock). Other
        ranks return their offset in seconds.
        """
        import statistics
        import torch
        import torch.distributed as dist
        if self._rank is None or self._world_size is None or self._world_size < 2:
            return 0.0
        device = "cuda" if torch.cuda.is_available() else "cpu"
        N = self._clock_offset_pings

        if self._rank == 0:
            # For every peer, exchange N pings; broadcast their offset back.
            for peer in range(1, self._world_size):
                samples: list[float] = []
                for _ in range(N):
                    t_send = time.perf_counter()
                    send = torch.tensor([t_send], dtype=torch.float64,
                                        device=device)
                    dist.send(send, peer)
                    recv = torch.zeros(1, dtype=torch.float64, device=device)
                    dist.recv(recv, peer)
                    t_ack = time.perf_counter()
                    t_peer_recv = float(recv.item())
                    rtt = t_ack - t_send
                    # peer clock at (t_send + rtt/2 on our clock) was t_peer_recv
                    samples.append(t_peer_recv - (t_send + rtt / 2))
                med = statistics.median(samples)
                offset_t = torch.tensor([med], dtype=torch.float64, device=device)
                # Broadcast this peer's offset to that specific rank; use
                # a group of {0, peer} implicitly via a targeted send/recv
                # rather than broadcast so it doesn't touch other peers.
                dist.send(offset_t, peer)
            return 0.0
        else:
            for _ in range(N):
                recv = torch.zeros(1, dtype=torch.float64, device=device)
                dist.recv(recv, 0)
                t_here = time.perf_counter()
                back = torch.tensor([t_here], dtype=torch.float64, device=device)
                dist.send(back, 0)
            offset_t = torch.zeros(1, dtype=torch.float64, device=device)
            dist.recv(offset_t, 0)
            return float(offset_t.item())

    def wrap_dataloader(self, loader):
        """Wrap a DataLoader (or any iterable) so time spent blocked on
        `next(iter)` — the actual prefetch-queue wait — is measured
        and attributed to each step's `dataloader_wait_s`.

        Use in place of the raw loader:

            for i, batch in enumerate(prof.wrap_dataloader(loader)):
                with prof.step(i) as s:
                    ...

        A large `dataloader_wait_s` means the worker pool couldn't
        keep the queue full — the `rule_prefetch_queue_starved`
        insight fires on that pattern.
        """
        return _WrappedLoader(loader, self)

    def _record_prefetch_wait(self, seconds: float) -> None:
        """Stashed by the wrapped loader; consumed on step entry."""
        self._pending_prefetch_wait_s = max(0.0, float(seconds))

    @contextmanager
    def step(self, step_index: int):
        rec = _StepRecorder(step_index, cuda=self._cuda)
        self._current_step = rec

        # Attribute any prefetch-queue wait captured by wrap_dataloader
        # into this step's dataloader_wait phase.
        if self._pending_prefetch_wait_s > 0:
            rec.phases["dataloader_wait"] = self._pending_prefetch_wait_s
            self._pending_prefetch_wait_s = 0.0

        now = time.perf_counter()
        if self._prev_step_end is not None:
            # This is the "real" dataloader stall in a
            # `for batch in loader:` loop — the wait between step
            # contexts, not inside one.
            rec.inter_step_gap_s = now - self._prev_step_end
        rec.t_start = now

        trace = None
        if self._trace_every > 0 and self._should_trace(step_index):
            try:
                from .traces import ProfilerTrace
                trace = ProfilerTrace(cuda=self._cuda).__enter__()
            except Exception:
                trace = None

        try:
            yield rec
        finally:
            if rec.cuda:
                rec.resolve_cuda_events()
            # CUDA-event comm timing (when the DDP hook is CUDA-based)
            # is resolved once per step, same idea — one sync flushes
            # the whole queue.
            rec.resolve_cuda_comm_events()
            rec.t_end = time.perf_counter()
            self._prev_step_end = rec.t_end
            self._current_step = None

            # Auto-compute FLOPs from architecture if user provided
            # tokens but didn't override flops.
            if rec.flops is None and rec.tokens and self._arch:
                fp = transformer_flops_per_step(self._arch, rec.tokens)
                rec.flops = fp["flops"]

            d = rec.to_dict()
            self._dispatch("push_step", d)
            for cb in self._on_step_cbs:
                try: cb(d)
                except Exception: pass

            # Flush per-bucket comm events collected by the DDP hook.
            for ev in rec._comm_events:
                self._dispatch("push_comm_event", ev)

            if trace is not None:
                try:
                    trace.__exit__(None, None, None)
                    kernels = trace.top_kernels(k=25)
                except Exception:
                    kernels = []
                if kernels:
                    self._dispatch("push_trace", {
                        "step": step_index,
                        "captured_at": time.time(),
                        "kernels": kernels,
                    })

    def _should_trace(self, step_index: int) -> bool:
        if step_index in self._trace_warmup:
            return True
        if self._trace_range and \
                self._trace_range[0] <= step_index < self._trace_range[1]:
            # Burst-trace window: every step in [start, end) gets a
            # kernel-level trace. This is the closest we get to
            # CUPTI-continuous visibility without a full C-level rewrite
            # — torch.profiler itself uses CUPTI under the hood.
            return True
        return step_index > 0 and step_index % self._trace_every == 0

    def _on_sample(self, s: Sample) -> None:
        self._dispatch("push_sample", s)

    def _on_host_sample(self, h: HostSample) -> None:
        self._dispatch("push_host_sample", h)


class _WrappedLoader:
    """Iterable wrapper that times how long `next()` blocks.

    A short wait means the prefetch queue had a batch ready. A long
    wait means the queue was empty and we had to sit through worker
    fetch + decode + augment. The rate of long-wait steps is what
    `rule_prefetch_queue_starved` counts.
    """

    def __init__(self, loader, prof: "GpuProfiler"):
        self._loader = loader
        self._prof = prof

    def __iter__(self):
        it = iter(self._loader)
        while True:
            t0 = time.perf_counter()
            try:
                batch = next(it)
            except StopIteration:
                return
            self._prof._record_prefetch_wait(time.perf_counter() - t0)
            yield batch

    def __len__(self):
        return len(self._loader)

    def __getattr__(self, name):
        # Delegate everything else (dataset, batch_size, sampler…) to
        # the wrapped loader so `wrap_dataloader(loader)` is a true
        # drop-in replacement.
        return getattr(self._loader, name)


class _StepRecorder:
    def __init__(self, step_index: int, cuda: bool = False):
        self.step = step_index
        self.cuda = cuda
        self.t_start: float = 0.0
        self.t_end: float = 0.0
        self.inter_step_gap_s: Optional[float] = None
        self.phases: dict[str, float] = {}
        self.loss: Optional[float] = None
        self.tokens: Optional[int] = None
        self.flops: Optional[float] = None
        # (name, start_event, end_event) — resolved at step exit
        self._pending: list[tuple[str, "object", "object"]] = []
        # Per-bucket comm events collected during the step (populated by
        # the DDP comm hook via `add_comm_event`).
        self._comm_events: list[dict] = []
        # Pending CUDA-event pairs for per-bucket comm timing, resolved
        # alongside phase events at step exit for sub-μs precision.
        # Each entry: (bucket_id, start_ev, end_ev, epoch_anchor_ev)
        self._pending_cuda_comm: list[tuple] = []

    @contextmanager
    def phase(self, name: str):
        if self.cuda:
            import torch
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try: yield
            finally:
                end.record()
                self._pending.append((name, start, end))
        else:
            t0 = time.perf_counter()
            try: yield
            finally:
                self.phases[name] = self.phases.get(name, 0.0) + (
                    time.perf_counter() - t0
                )

    def resolve_cuda_events(self) -> None:
        if not self._pending:
            return
        _, _, last_end = self._pending[-1]
        last_end.synchronize()
        for name, start, end in self._pending:
            ms = start.elapsed_time(end)
            self.phases[name] = self.phases.get(name, 0.0) + ms / 1000.0
        self._pending.clear()

    def add_comm_time(self, seconds: float) -> None:
        """Called by DDP comm-hook — accumulate into the `comm` phase."""
        self.phases["comm"] = self.phases.get("comm", 0.0) + max(0.0, seconds)

    def add_comm_event(self, bucket_id: int,
                       t_start_rel: float, t_end_rel: float) -> None:
        """Per-bucket comm timing for cross-rank correlation. Times are
        relative to the run's shared sync-epoch."""
        self._comm_events.append({
            "step": self.step, "bucket_id": bucket_id,
            "t_start_rel": t_start_rel, "t_end_rel": t_end_rel,
        })

    def add_pending_cuda_comm_event(self, bucket_id: int,
                                    start_ev, end_ev, epoch_ev) -> None:
        """Stash a CUDA-event pair; resolved via `resolve_cuda_comm_events`
        at step exit so we sync only once per step."""
        self._pending_cuda_comm.append(
            (bucket_id, start_ev, end_ev, epoch_ev)
        )

    def resolve_cuda_comm_events(self) -> None:
        """Sync the last pending CUDA event, then read all timings as
        ms-since-epoch → seconds. Populates both the aggregate `comm`
        phase and per-bucket comm_events."""
        if not self._pending_cuda_comm:
            return
        # Sync on the last end event; that flushes the queue.
        _, _, last_end, _ = self._pending_cuda_comm[-1]
        try:
            last_end.synchronize()
        except Exception:
            self._pending_cuda_comm.clear()
            return
        total = 0.0
        for bid, start_ev, end_ev, epoch_ev in self._pending_cuda_comm:
            try:
                # `elapsed_time(other)` on CUDA events returns ms between
                # the two events on the GPU global timer — sub-μs
                # precision. We convert to seconds and offset from the
                # run's epoch anchor event.
                t_start_ms = epoch_ev.elapsed_time(start_ev)
                t_end_ms = epoch_ev.elapsed_time(end_ev)
            except Exception:
                continue
            dt = (t_end_ms - t_start_ms) / 1000.0
            total += max(0.0, dt)
            self._comm_events.append({
                "step": self.step, "bucket_id": bid,
                "t_start_rel": t_start_ms / 1000.0,
                "t_end_rel": t_end_ms / 1000.0,
            })
        self.phases["comm"] = self.phases.get("comm", 0.0) + total
        self._pending_cuda_comm.clear()

    def record(self, *, loss=None, tokens=None, flops=None) -> None:
        if loss is not None: self.loss = float(loss)
        if tokens is not None: self.tokens = int(tokens)
        if flops is not None: self.flops = float(flops)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "inter_step_gap_s": self.inter_step_gap_s,
            "dataloader_wait_s": self.phases.get("dataloader_wait"),
            "forward_s": self.phases.get("forward"),
            "backward_s": self.phases.get("backward"),
            "optimizer_s": self.phases.get("optimizer"),
            "comm_s": self.phases.get("comm"),
            "loss": self.loss,
            "tokens": self.tokens,
            "flops": self.flops,
        }
