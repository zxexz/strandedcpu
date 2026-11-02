# Measuring performance

A CUDA ISA kernel runs integer instructions. A transformer step runs attention and feed-forward networks. Name the backend when reporting a timing; the two numbers measure different computations.

## Time the call you intend to compare

For scoring latency, include translation, compilation, execution, option parsing, and matching. If timing only a CUDA kernel, say so. Synchronize device work before stopping the timer; measuring an asynchronous launch omits queued work.

Keep the corpus and inputs fixed. Save the source programs, option texts, word width, number mode, memory configuration, step limit, batch size, device, dependency versions, and commit. Save raw timings from several runs rather than just the best result. Separate successful matches from abstentions when comparing latency.

Inspect the returned `backend` and `fallback_reason`. `auto` can use the interpreter, and binary-mode CUDA scoring above 32 bits falls back even when the source contains only integers. The requested backend is not enough to identify the run.

## Account for caches and workers

Compilation and transformer construction are cached. Use a fresh process for startup latency. For repeated calls, record which programs and shapes were warmed.

`compile_workers` uses forkserver processes. Include their startup and serialization costs when timing a fresh process, and call multiprocessing code behind an `if __name__ == "__main__":` guard.

CUDA batches independent programs, one per thread. The transformer's work also depends on tensor dimensions and instruction count: more columns increase attention work, while wider words increase the threshold networks. Record the constructed shape and retired steps.

Scoring with `transformer` or `cuda-transformer` runs a second execution path for comparison. To time the transformer alone, use `run_function(..., backend="transformer", device=...)`. State whether construction, transfers, and CUDA graph capture are included.

## iCEstick

UART upload and state download can cost more than execution on the small ISA core. Measure them separately if reporting core throughput. ISA steps are retired instructions, not FPGA clocks.

The transformer MAC path keeps weights, state, and much of the arithmetic on the host. Report its offloaded dot and term counts, fallback counts, and serial bytes with the timing. The recorded [iCEstick validation](../../hardware/icestick/validation.json) tests correctness; it is not a throughput benchmark.
