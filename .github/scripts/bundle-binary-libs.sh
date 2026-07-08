#!/usr/bin/env bash
set -euo pipefail

BIN="${1:-target/release/candle-vllm}"
LIB_DIR="${2:-target/release/lib}"

mkdir -p "$LIB_DIR"

resolve_ldd_lib() {
  local prefix="$1"
  ldd "$BIN" 2>/dev/null | awk -v prefix="$prefix" '
    index($1, prefix) == 1 && $2 == "=>" && $3 != "not" { print $3; exit }
    index($1, prefix) == 1 && $2 ~ /^\// { print $2; exit }
  '
}

resolve_ldconfig_lib() {
  local prefix="$1"
  ldconfig -p 2>/dev/null | awk -v prefix="$prefix" '
    index($1, prefix) == 1 { print $NF; exit }
  '
}

copy_with_soname_links() {
  local lib_path="$1"
  local base soname linker_name

  if [[ -z "$lib_path" || ! -e "$lib_path" ]]; then
    return 1
  fi

  base="$(basename "$lib_path")"
  cp -L "$lib_path" "$LIB_DIR/$base"

  soname="$(objdump -p "$LIB_DIR/$base" 2>/dev/null | awk '/SONAME/{ print $2; exit }')"
  if [[ -n "$soname" ]]; then
    if [[ "$soname" != "$base" ]]; then
      ln -sf "$base" "$LIB_DIR/$soname"
    fi
    if [[ "$soname" == *.so.* ]]; then
      linker_name="${soname%%.so.*}.so"
      ln -sf "$soname" "$LIB_DIR/$linker_name"
    fi
  fi
}

bundle_dependency() {
  local prefix="$1"
  local lib_path=""

  lib_path="$(resolve_ldd_lib "$prefix")"

  if [[ -z "$lib_path" && "$prefix" == "libcudart.so" ]]; then
    lib_path="$(find /usr/local/cuda -follow -name 'libcudart.so.*.*' -not -name '*.a' 2>/dev/null | head -1 || true)"
  fi

  if [[ -z "$lib_path" ]]; then
    lib_path="$(resolve_ldconfig_lib "$prefix")"
  fi

  if [[ -z "$lib_path" ]]; then
    echo "Warning: could not find $prefix to bundle."
    return 0
  fi

  echo "Bundling $prefix from $lib_path"
  copy_with_soname_links "$lib_path"
}

bundle_dependency "libcudart.so"
bundle_dependency "libssl.so"
bundle_dependency "libcrypto.so"

patchelf --set-rpath '$ORIGIN/lib:/usr/local/lib/candle-vllm' "$BIN"
ls -la "$LIB_DIR"/
