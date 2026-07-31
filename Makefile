# choose your compiler, e.g. gcc/clang
# example override to clang: make run CC=clang
CC = gcc

# the most basic way of building that is most likely to work on most systems
.PHONY: run
run: src/run.c
	$(CC) -O3 -o run src/run.c -lm
	$(CC) -O3 -o runq src/runq.c -lm

# GELU-gated build for ReaLLM-Forge GeGLU models (activation_variant=gelu).
# See doc/reallmforge_to_llama2c.md. Produces run_gelu / runq_gelu.
.PHONY: rungelu
rungelu: src/run.c
	$(CC) -O3 -DACT_GELU -o run_gelu  src/run.c  -lm
	$(CC) -O3 -DACT_GELU -o runq_gelu src/runq.c -lm

# Tier-2 build: GELU + peri-LN (extra rmsnorm on each sub-layer output). For ReaLLM-Forge
# models with use_peri_ln=True (export_reallmforge.py auto-includes the peri norms).
# Produces run_periln / runq_periln.
.PHONY: runperiln
runperiln: src/run.c
	$(CC) -O3 -DACT_GELU -DPERI_LN -o run_periln  src/run.c  -lm
	$(CC) -O3 -DACT_GELU -DPERI_LN -o runq_periln src/runq.c -lm

# Heterogeneous / infinite-head-attention build: runs NSGA-searched ReaLLM-Forge models with
# per-layer dims, infinite attention (concat path), identity layers, GQA, peri-LN, GeGLU + erf
# GELU, RoPE. Reads the .rlm format (reallmforge/export_reallm_hetero.py). fp32 only.
# See doc/reallmforge_hetero_infinite.md. Produces run_reallm (single-threaded; the OpenMP
# pragmas are ignored without -fopenmp, so this builds everywhere incl. stock macOS clang).
.PHONY: runreallm
runreallm: src/run_reallm.c src/bpe.h
	$(CC) -O3 -o run_reallm src/run_reallm.c -lm

# Q8_0 (int8 group-quantized) variant of run_reallm — ~4x smaller, for on-device/Android.
# Reads version-2 .rlm (export_reallm_hetero.py --version 2). Produces runq_reallm.
.PHONY: runqreallm
runqreallm: src/runq_reallm.c src/bpe.h
	$(CC) -O3 -o runq_reallm src/runq_reallm.c -lm

# Multithreaded (OpenMP) builds of the .rlm runners. Big speedup on the classifier/MLP matmuls;
# needs libomp on macOS (brew install libomp). Run with e.g. OMP_NUM_THREADS=4 ./run_reallm ...
.PHONY: runreallmomp
runreallmomp: src/run_reallm.c src/bpe.h
	$(CC) -Ofast -fopenmp -march=native -o run_reallm  src/run_reallm.c  -lm
	$(CC) -Ofast -fopenmp -march=native -o runq_reallm src/runq_reallm.c -lm

# Android NDK cross-compile (aarch64) of the quantized GELU engine for on-device runs.
# Requires the Android NDK. Set ANDROID_NDK to your install, e.g.:
#   make runq_android ANDROID_NDK=$HOME/android-ndk-r27c
# Produces a dynamic PIE aarch64 binary `runq_android` (the standard Android artifact; links
# the on-device Bionic libc/linker64). Push via `adb push` alongside the .bin model +
# tokenizer_gpt2.bin, then run from `adb shell`. Verified: cross-compiles to a valid aarch64
# ELF, and the identical src/runq.c runs correctly on aarch64 (fp32 path is bit-exact vs x86 under
# qemu). `runq_android_static` builds a fully-static variant (uses Bionic static linking, which
# Google discourages; handy if the device shell has linker issues).
ANDROID_NDK ?= $(HOME)/android-ndk-r27c
ANDROID_API ?= 24
ANDROID_CLANG = $(ANDROID_NDK)/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android$(ANDROID_API)-clang
ANDROID_CLANG_X86 = $(ANDROID_NDK)/toolchains/llvm/prebuilt/linux-x86_64/bin/x86_64-linux-android$(ANDROID_API)-clang
.PHONY: runq_android runq_android_static runq_android_x86_64
runq_android:
	$(ANDROID_CLANG) -O3 -DACT_GELU -o runq_android src/runq.c -lm
runq_android_static:
	$(ANDROID_CLANG) -O3 -DACT_GELU -static -o runq_android src/runq.c -lm
# x86_64 Android build (for x86-host emulators / AVD x86_64 system images)
runq_android_x86_64:
	$(ANDROID_CLANG_X86) -O3 -DACT_GELU -o runq_android_x86_64 src/runq.c -lm

# useful for a debug build, can then e.g. analyze with valgrind, example:
# $ valgrind --leak-check=full ./run out/model.bin -n 3
rundebug: src/run.c
	$(CC) -g -o run src/run.c -lm
	$(CC) -g -o runq src/runq.c -lm

# https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
# https://simonbyrne.github.io/notes/fastmath/
# -Ofast enables all -O3 optimizations.
# Disregards strict standards compliance.
# It also enables optimizations that are not valid for all standard-compliant programs.
# It turns on -ffast-math, -fallow-store-data-races and the Fortran-specific
# -fstack-arrays, unless -fmax-stack-var-size is specified, and -fno-protect-parens.
# It turns off -fsemantic-interposition.
# In our specific application this is *probably* okay to use
.PHONY: runfast
runfast: src/run.c
	$(CC) -Ofast -o run src/run.c -lm
	$(CC) -Ofast -o runq src/runq.c -lm

# additionally compiles with OpenMP, allowing multithreaded runs
# make sure to also enable multiple threads when running, e.g.:
# OMP_NUM_THREADS=4 ./run out/model.bin
.PHONY: runomp
runomp: src/run.c
	$(CC) -Ofast -fopenmp -march=native src/run.c  -lm  -o run
	$(CC) -Ofast -fopenmp -march=native src/runq.c  -lm  -o runq

.PHONY: win64
win64:
	x86_64-w64-mingw32-gcc -Ofast -D_WIN32 -o run.exe -I. src/run.c src/win.c
	x86_64-w64-mingw32-gcc -Ofast -D_WIN32 -o runq.exe -I. src/runq.c src/win.c

# compiles with gnu99 standard flags for amazon linux, coreos, etc. compatibility
.PHONY: rungnu
rungnu:
	$(CC) -Ofast -std=gnu11 -o run src/run.c -lm
	$(CC) -Ofast -std=gnu11 -o runq src/runq.c -lm

.PHONY: runompgnu
runompgnu:
	$(CC) -Ofast -fopenmp -std=gnu11 src/run.c  -lm  -o run
	$(CC) -Ofast -fopenmp -std=gnu11 src/runq.c  -lm  -o runq

# run all tests
.PHONY: test
test:
	pytest

# run only tests for src/run.c C implementation (is a bit faster if only C code changed)
.PHONY: testc
testc:
	pytest -k runc

# run the C tests, without touching pytest / python
# to increase verbosity level run e.g. as `make testcc VERBOSITY=1`
VERBOSITY ?= 0
.PHONY: testcc
testcc:
	$(CC) -DVERBOSITY=$(VERBOSITY) -O3 -o testc src/test.c -lm
	./testc

# build the GPT-2 BPE tokenizer unit test (src/bpe_test.c). Run it as:
#   ./bpe_test models/smollm2_135M/tokenizer_gpt2.bin < some_lines.txt
.PHONY: bpetest
bpetest:
	$(CC) -O3 -o bpe_test src/bpe_test.c -lm

.PHONY: clean
clean:
	rm -f run runq run_gelu runq_gelu run_periln runq_periln
	rm -f run_reallm runq_reallm
	rm -f runq_android runq_android_x86_64 bpe_test testc
