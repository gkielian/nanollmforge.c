#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/../../src/runq.c" ]; then
  ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
elif [ -f "$SCRIPT_DIR/../src/runq.c" ]; then
  ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
else
  ROOT="$(pwd)"
fi
cd "$ROOT"

NDK="${NDK:-$HOME/Library/Android/sdk/ndk/30.0.14904198}"
ADB="${ADB:-$HOME/Library/Android/sdk/platform-tools/adb}"
ANDROID_SERIAL="46131JEAYW0243"
SERIAL="${1:-${ANDROID_SERIAL:-}}"
PROMPT="${2:-Once upon a time, a little girl named Lily}"

ADB_CMD=("$ADB")
if [ -n "$SERIAL" ]; then
  ADB_CMD+=("-s" "$SERIAL")
fi

MODEL_DIR="models/tinystories_15M"
MODEL_PT="$MODEL_DIR/stories15M.pt"
MODEL_Q8="$MODEL_DIR/stories15M.q8.bin"
TOKENIZER="tokenizer.bin"
if [ ! -f "$TOKENIZER" ] && [ -f "$MODEL_DIR/tokenizer.bin" ]; then
  TOKENIZER="$MODEL_DIR/tokenizer.bin"
fi
MODEL_URL="https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.pt"

mkdir -p "$MODEL_DIR"

# 1. Download PyTorch checkpoint if not present
if [ ! -f "$MODEL_PT" ]; then
  echo "Downloading pre-trained TinyStories-15M PyTorch checkpoint from Hugging Face..."
  curl -L -o "$MODEL_PT" "$MODEL_URL"
fi

# 2. Export and Quantize to INT8 Q8_0 format (.bin)
EXPORT_SCRIPT="pytorch/export.py"
if [ ! -f "$EXPORT_SCRIPT" ]; then
  EXPORT_SCRIPT="export.py"
fi

if [ ! -f "$MODEL_Q8" ]; then
  echo "Quantizing TinyStories-15M to INT8 Q8_0 format (version 2)..."
  python3 "$EXPORT_SCRIPT" "$MODEL_Q8" --checkpoint "$MODEL_PT" --version 2
fi

# 3. Detect device target architecture (32-bit armv7a / armv8l vs 64-bit aarch64)
ARCH=$("${ADB_CMD[@]}" shell "uname -m" 2>/dev/null | tr -d '\r\n' || echo "aarch64")

if [[ "$ARCH" == "armv7"* ]] || [[ "$ARCH" == "armv8l" ]]; then
  echo "Target device architecture: 32-bit ARM ($ARCH) [Pixel Watch]"
  COMPILER="$NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin/armv7a-linux-androideabi24-clang"
else
  echo "Target device architecture: 64-bit ARM64 ($ARCH) [Phone/Emulator]"
  COMPILER="$NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android24-clang"
fi

# 4. Compile runq engine for target architecture
RUNQ_SRC="src/runq.c"
if [ ! -f "$RUNQ_SRC" ]; then
  RUNQ_SRC="src/runq.c"
fi

echo "Compiling runq for $ARCH from $RUNQ_SRC..."
"$COMPILER" -O3 -Isrc -o runq_tinystories_android "$RUNQ_SRC" -lm

# 5. Push binaries and model files to target device
echo "Pushing binaries and model files to target device..."
"${ADB_CMD[@]}" push runq_tinystories_android /data/local/tmp/runq_tinystories_android
"${ADB_CMD[@]}" push "$TOKENIZER" /data/local/tmp/
"${ADB_CMD[@]}" push "$MODEL_Q8" /data/local/tmp/

# 6. Set permissions and run inference on device
"${ADB_CMD[@]}" shell "chmod +x /data/local/tmp/runq_tinystories_android"
echo "Running INT8 Quantized TinyStories 15M on Android target device..."
"${ADB_CMD[@]}" shell "cd /data/local/tmp && ./runq_tinystories_android $(basename "$MODEL_Q8") -z $(basename "$TOKENIZER") -i '$PROMPT' -t 0.8 -p 0.9 -n 128"
