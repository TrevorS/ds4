CC ?= cc
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
NATIVE_CPU_FLAG ?= -mcpu=native
else
NATIVE_CPU_FLAG ?= -march=native
endif

DEBUG_FLAGS ?= -g
CFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99
OBJCFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -fobjc-arc

LDLIBS ?= -lm -pthread
METAL_SRCS := $(wildcard metal/*.metal)

ifeq ($(UNAME_S),Darwin)
METAL_LDLIBS := $(LDLIBS) -framework Foundation -framework Metal
CORE_OBJS = ds4.o ds4_distributed.o ds4_metal.o
CPU_CORE_OBJS = ds4_cpu.o ds4_distributed.o
else
CFLAGS += -D_GNU_SOURCE -fno-finite-math-only
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?=
ifneq ($(strip $(CUDA_ARCH)),)
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif
NVCCFLAGS ?= -O3 -g -lineinfo --use_fast_math $(NVCC_ARCH_FLAGS) -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread
CORE_OBJS = ds4.o ds4_distributed.o ds4_cuda.o
CPU_CORE_OBJS = ds4_cpu.o ds4_distributed.o
# Experimental CUDA-graph decode (non-strict, opt-in): build with DS4_GRAPH_DECODE=1.
# --default-stream per-thread moves the engine's default-stream launches onto a
# capturable per-thread stream (cudaStreamPerThread), the prerequisite for stream
# capture. -DDS4_GRAPH_DECODE_BUILD gates the (off-by-default) capture code path.
ifdef DS4_GRAPH_DECODE
NVCCFLAGS += --default-stream per-thread -DDS4_GRAPH_DECODE_BUILD
CFLAGS += -DDS4_GRAPH_DECODE_BUILD
endif
CUDA_LDLIBS ?= -lm -Xcompiler -pthread -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas
METAL_LDLIBS := $(LDLIBS)
endif

.PHONY: all help clean test cpu cuda cuda-spark cuda-generic cuda-regression token-diff cuda-tap-capture cuda-tap-regression cuda-ppl cuda-ppl-baseline

ifeq ($(UNAME_S),Darwin)
all: ds4 ds4-server ds4-bench ds4-eval ds4-agent

help:
	@echo "DS4 build targets:"
	@echo "  make              Build Metal ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make cpu          Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make test         Build and run tests"
	@echo "  make clean        Remove build outputs"

ds4: ds4_cli.o ds4_help.o linenoise.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli.o ds4_help.o linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-server: ds4_server.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_server.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-bench: ds4_bench.o ds4_help.o ds4_kvstore.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_bench.o ds4_help.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-eval: ds4_eval.o ds4_help.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_eval.o ds4_help.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-agent: ds4_agent.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_agent.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o ds4_eval_cpu.o ds4_agent_cpu.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o rax.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 ds4_cli_cpu.o ds4_help.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server ds4_server_cpu.o ds4_help.o ds4_kvstore.o rax.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench ds4_bench_cpu.o ds4_help.o ds4_kvstore.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval ds4_eval_cpu.o ds4_help.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-agent ds4_agent_cpu.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)

cuda-regression:
	@echo "cuda-regression requires a CUDA build"
else
all: help

help:
	@echo "DS4 build targets:"
	@echo "  make cuda-spark          Build CUDA for DGX Spark / GB10"
	@echo "  make cuda-generic        Build CUDA for a generic local CUDA GPU"
	@echo "  make cuda CUDA_ARCH=sm_N Build CUDA with an explicit nvcc -arch value"
	@echo "  make cpu                 Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make test                Build and run tests"
	@echo "  make clean               Remove build outputs"

cuda-spark:
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH=

cuda-generic:
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH=native

cuda:
	@if [ -z "$(strip $(CUDA_ARCH))" ]; then \
		echo "error: specify CUDA_ARCH, for example: make cuda CUDA_ARCH=sm_120"; \
		echo "       or use make cuda-spark / make cuda-generic"; \
		exit 2; \
	fi
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH="$(CUDA_ARCH)"

ds4: ds4_cli.o ds4_help.o linenoise.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-server: ds4_server.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-bench: ds4_bench.o ds4_help.o ds4_kvstore.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-eval: ds4_eval.o ds4_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-agent: ds4_agent.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o ds4_eval_cpu.o ds4_agent_cpu.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o rax.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 ds4_cli_cpu.o ds4_help.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server ds4_server_cpu.o ds4_help.o ds4_kvstore.o rax.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench ds4_bench_cpu.o ds4_help.o ds4_kvstore.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval ds4_eval_cpu.o ds4_help.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-agent ds4_agent_cpu.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)

cuda-regression: tests/cuda_long_context_smoke
	./tests/cuda_long_context_smoke

# Two-shot greedy token-diff determinism gate. ds4-bench auto-sets
# DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 whenever DS4_BENCH_TOKEN_DUMP is present, so a
# re-run of the same command must produce a byte-identical token stream.
# Override TOKEN_DIFF_ARGS to point at your model / MTP / KV anchor.
TOKEN_DIFF_ARGS ?= --cuda --warm-weights -m ds4flash.gguf --gen-tokens 32 --temp 0
token-diff: ds4-bench
	DS4_BENCH_TOKEN_DUMP=/tmp/ds4-tokdiff-a.txt ./ds4-bench $(TOKEN_DIFF_ARGS)
	DS4_BENCH_TOKEN_DUMP=/tmp/ds4-tokdiff-b.txt ./ds4-bench $(TOKEN_DIFF_ARGS)
	@diff -u /tmp/ds4-tokdiff-a.txt /tmp/ds4-tokdiff-b.txt \
		&& echo "token-diff: OK (deterministic)" \
		|| { echo "token-diff: MISMATCH — nondeterministic decode"; exit 1; }

# CUDA fingerprint-tap regression gate (ds4_gpu_fingerprint_tap_f32). Capture
# per-layer tap fingerprints for a fixed prompt into a committed golden dir;
# cuda-tap-regression re-captures and diffs (bit-exact byte_xor + soft RMS).
# byte_xor is per-binary/per-arch (GB10 sm_121a) — golden lives under the
# arch-tagged dir. Override TAP_ARGS for your model.
TAP_GOLDEN_DIR ?= tests/golden/cuda-taps-gb10
TAP_ARGS       ?= --cuda -m ds4flash.gguf -p "The quick brown fox jumps over the lazy dog." -n 4 --temp 0
cuda-tap-capture: ds4
	@mkdir -p $(TAP_GOLDEN_DIR)
	@rm -f $(TAP_GOLDEN_DIR)/*.fp
	DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 DS4_CUDA_TAP_PREFIX=$(TAP_GOLDEN_DIR)/tap ./ds4 $(TAP_ARGS) >/dev/null
	@# Consolidate the per-(tag,layer,pos) files into one git-friendly golden,
	@# dropping ffn_moe_gate — a debug-materialize-only scratch tap that is stale
	@# (nondeterministic) in a normal decode forward. The differ ignores it too
	@# (tools/cuda_tap_diff.py DEFAULT_IGNORE_TAGS is the policy source of truth);
	@# we just keep the committed baseline to gateable signals only.
	@grep -hv 'tag=ffn_moe_gate ' $(TAP_GOLDEN_DIR)/tap_*.fp > $(TAP_GOLDEN_DIR)/golden.fp && rm -f $(TAP_GOLDEN_DIR)/tap_*.fp
	@echo "cuda-tap-capture: golden -> $(TAP_GOLDEN_DIR)/golden.fp ($$(wc -l < $(TAP_GOLDEN_DIR)/golden.fp) taps, ffn_moe_gate excluded)"
cuda-tap-regression: ds4
	@rm -rf /tmp/ds4-tap-cand && mkdir -p /tmp/ds4-tap-cand
	DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 DS4_CUDA_TAP_PREFIX=/tmp/ds4-tap-cand/tap ./ds4 $(TAP_ARGS) >/dev/null
	python3 tools/cuda_tap_diff.py $(TAP_GOLDEN_DIR) /tmp/ds4-tap-cand

# Perplexity regression gate: teacher-forced avg-NLL on a committed corpus vs a
# committed baseline (tests/test-vectors/ppl-baseline.txt). One scale-free scalar
# that integrates every layer's numeric drift. Determinism is forced in the test.
cuda-ppl: ds4_test
	./ds4_test --cuda-ppl
cuda-ppl-baseline: ds4_test
	DS4_TEST_PPL_WRITE_BASELINE=1 ./ds4_test --cuda-ppl

# Shared library for in-process embedding via ctypes/cffi (Linux + CUDA).
# Core engine API lives in ds4.c (+ ds4_cuda.cu); no server/cli objects needed.
ds4_pic.o: ds4.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -fPIC -c -o $@ ds4.c

ds4_cuda_pic.o: ds4_cuda.cu ds4_gpu.h ds4_iq2_tables_cuda.inc
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fPIC -c -o $@ ds4_cuda.cu

libds4.so: ds4_pic.o ds4_cuda_pic.o
	$(NVCC) $(NVCCFLAGS) -shared -Xcompiler -fPIC -o $@ $^ $(CUDA_LDLIBS)
endif

ds4.o: ds4.c ds4.h ds4_distributed.h ds4_gpu.h
	$(CC) $(CFLAGS) -c -o $@ ds4.c

ds4_cli.o: ds4_cli.c ds4.h ds4_distributed.h ds4_help.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_cli.c

ds4_distributed.o: ds4_distributed.c ds4_distributed.h ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_distributed.c

ds4_help.o: ds4_help.c ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_help.c

ds4_server.o: ds4_server.c ds4.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -c -o $@ ds4_server.c

ds4_bench.o: ds4_bench.c ds4.h ds4_distributed.h ds4_help.h ds4_kvstore.h
	$(CC) $(CFLAGS) -c -o $@ ds4_bench.c

ds4_eval.o: ds4_eval.c ds4.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_eval.c

ds4_agent.o: ds4_agent.c ds4.h ds4_distributed.h ds4_help.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_agent.c

ds4_web.o: ds4_web.c ds4_web.h
	$(CC) $(CFLAGS) -c -o $@ ds4_web.c

ds4_kvstore.o: ds4_kvstore.c ds4_kvstore.h ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_kvstore.c

ds4_test.o: tests/ds4_test.c ds4_server.c ds4.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -Wno-unused-function -c -o $@ tests/ds4_test.c

tests/cuda_long_context_smoke.o: tests/cuda_long_context_smoke.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/cuda_long_context_smoke.c

rax.o: rax.c rax.h rax_malloc.h
	$(CC) $(CFLAGS) -c -o $@ rax.c

linenoise.o: linenoise.c linenoise.h
	$(CC) $(CFLAGS) -c -o $@ linenoise.c

ds4_cpu.o: ds4.c ds4.h ds4_distributed.h ds4_gpu.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4.c

ds4_cli_cpu.o: ds4_cli.c ds4.h ds4_distributed.h ds4_help.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_cli.c

ds4_server_cpu.o: ds4_server.c ds4.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_server.c

ds4_bench_cpu.o: ds4_bench.c ds4.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_bench.c

ds4_eval_cpu.o: ds4_eval.c ds4.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_eval.c

ds4_agent_cpu.o: ds4_agent.c ds4.h ds4_distributed.h ds4_help.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_agent.c

ds4_metal.o: ds4_metal.m ds4_gpu.h $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -c -o $@ ds4_metal.m

ds4_cuda.o: ds4_cuda.cu ds4_gpu.h ds4_iq2_tables_cuda.inc
	$(NVCC) $(NVCCFLAGS) -c -o $@ ds4_cuda.cu

tests/cuda_long_context_smoke: tests/cuda_long_context_smoke.o ds4_cuda.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4_test: ds4_test.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ ds4_test.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)
else
	$(NVCC) $(NVCCFLAGS) -o $@ ds4_test.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS) $(CUDA_LDLIBS)
endif

test: ds4_test ds4-eval q4k-dot-test
	./ds4-eval --self-test-extractors
	./ds4_test

q4k-dot-test: tests/test_q4k_dot.c
	$(CC) -O2 -Wall -Wextra -std=c99 -o tests/test_q4k_dot tests/test_q4k_dot.c -lm -pthread
	./tests/test_q4k_dot

clean:
	rm -f ds4 ds4-server ds4-bench ds4-eval ds4-agent ds4_cpu ds4_native ds4_server_test ds4_test tests/test_q4k_dot libds4.so *.o tests/cuda_long_context_smoke tests/cuda_long_context_smoke.o
