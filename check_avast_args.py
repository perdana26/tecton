"""
Cek ketersediaan argumen API di data Avast-CTU Anda
===================================================

Tujuan: menentukan apakah MAMBAFull (butuh argumen) bisa dijalankan, atau Anda
terpaksa ke MAMBALite. Skrip ini TIDAK mengubah apa pun -- hanya membaca dan
melaporkan.

Cara pakai:
    python check_avast_args.py /path/ke/folder/unduhan-avast-ctu

Lalu tempelkan seluruh output ke chat.

Yang dicari, berurutan dari yang paling menentukan:
  1. Format apa yang Anda punya (laporan CAPE mentah? file agregat? parquet/csv?)
  2. Apakah tiap API call punya field 'arguments' yang terisi?
  3. Jenis argumen apa yang ada (file path, registry, command, domain, IP)?
  4. Seberapa sering argumen terisi (coverage) -- karena MAMBA butuh resource,
     kalau 90% call tidak punya argumen maka MAMBAFull pun lemah.
"""

from __future__ import annotations

import gzip
import json
import os
import sys
from collections import Counter
from pathlib import Path


# API yang argumennya paling relevan untuk MAMBA (menyentuh resource)
RESOURCE_APIS = {
    "NtCreateFile", "NtWriteFile", "NtOpenFile", "CreateFileW", "CreateFileA",
    "RegSetValueExW", "RegCreateKeyExW", "RegOpenKeyExW", "NtSetValueKey",
    "CreateProcessInternalW", "ShellExecuteExW", "WriteProcessMemory",
    "InternetOpenUrlW", "HttpOpenRequestW", "DnsQuery_W", "connect", "send",
    "getaddrinfo", "URLDownloadToFileW", "LdrLoadDll",
}

ARG_KEYS_HINT = ("arg", "param", "value", "buffer", "path", "filepath",
                 "filename", "regkey", "key", "command", "cmdline", "url",
                 "host", "ip", "domain")


def open_maybe_gz(p: Path):
    if str(p).endswith(".gz"):
        return gzip.open(p, "rt", encoding="utf-8", errors="replace")
    return open(p, "r", encoding="utf-8", errors="replace")


def scan_directory(root: Path) -> dict:
    exts = Counter()
    sample_files = []
    total = 0
    for dirpath, _, files in os.walk(root):
        for f in files:
            total += 1
            ext = "".join(Path(f).suffixes).lower() or "<none>"
            exts[ext] += 1
            if len(sample_files) < 5 and _looks_relevant(f):
                sample_files.append(Path(dirpath) / f)
    return {"total_files": total, "extensions": dict(exts.most_common(15)),
            "sample_files": [str(p) for p in sample_files]}


def _looks_relevant(name: str) -> bool:
    n = name.lower()
    return any(n.endswith(e) for e in
               (".json", ".json.gz", ".parquet", ".csv", ".csv.gz", ".jsonl",
                ".jsonl.gz", ".pkl", ".pickle"))


def inspect_json_report(path: Path) -> dict:
    """Coba baca satu laporan sebagai JSON CAPE dan cari struktur argumen."""
    try:
        with open_maybe_gz(path) as fh:
            data = json.load(fh)
    except Exception as e:
        return {"file": str(path), "readable_json": False, "error": str(e)[:200]}

    out = {"file": str(path), "readable_json": True, "top_level_keys": list(data)[:20]}

    # Jalur umum CAPE: data['behavior']['apistats'] / ['processes'][i]['calls']
    calls = _find_calls(data)
    if calls is None:
        out["found_api_calls"] = False
        return out

    out["found_api_calls"] = True
    out["n_calls_in_this_report"] = len(calls)

    with_args, arg_key_counter, examples = 0, Counter(), []
    for c in calls[:5000]:
        if not isinstance(c, dict):
            continue
        arg_field = _extract_args(c)
        if arg_field:
            with_args += 1
            for k in arg_field:
                arg_key_counter[k] += 1
            if len(examples) < 8 and (c.get("api") in RESOURCE_APIS or
                                      not RESOURCE_APIS):
                examples.append({"api": c.get("api"),
                                 "args_preview": _preview(arg_field)})

    n = min(len(calls), 5000)
    out["pct_calls_with_arguments"] = round(with_args / max(n, 1), 3)
    out["common_argument_keys"] = dict(arg_key_counter.most_common(12))
    out["argument_examples"] = examples
    out["verdict_for_this_file"] = _verdict(with_args / max(n, 1), arg_key_counter)
    return out


def _find_calls(data):
    # Struktur bervariasi antar rilis CAPE; coba beberapa jalur.
    b = data.get("behavior") if isinstance(data, dict) else None
    if isinstance(b, dict):
        procs = b.get("processes")
        if isinstance(procs, list) and procs:
            calls = []
            for p in procs:
                calls.extend(p.get("calls", []) or [])
            if calls:
                return calls
        if isinstance(b.get("apistats"), (dict, list)):
            return []  # apistats = ringkasan frekuensi -> TANPA argumen
    # Sebagian rilis agregat menaruh urutan API langsung
    for k in ("api_calls", "apis", "calls", "sequence"):
        v = data.get(k) if isinstance(data, dict) else None
        if isinstance(v, list) and v:
            return v
    return None


def _extract_args(call: dict):
    for k in ("arguments", "args", "parameters"):
        v = call.get(k)
        if isinstance(v, dict) and v:
            return v
        if isinstance(v, list) and v:
            return {str(i): x for i, x in enumerate(v)}
    # Kadang argumen tersebar sebagai key langsung
    hits = {k: call[k] for k in call
            if any(h in k.lower() for h in ARG_KEYS_HINT) and k not in ("api",)}
    return hits or None


def _preview(argdict) -> dict:
    out = {}
    for k, v in list(argdict.items())[:4]:
        s = str(v)
        out[str(k)] = s[:80] + ("..." if len(s) > 80 else "")
    return out


def _verdict(pct, key_counter) -> str:
    if pct == 0:
        return "TANPA ARGUMEN -> hanya MAMBALite yang mungkin."
    resourceish = sum(v for k, v in key_counter.items()
                      if any(h in k.lower() for h in
                             ("path", "file", "key", "reg", "cmd", "command",
                              "url", "host", "ip", "domain")))
    if pct > 0.3 and resourceish > 0:
        return ("ARGUMEN RESOURCE TERSEDIA -> MAMBAFull layak. "
                "Inilah yang kita harapkan.")
    if pct > 0.3:
        return ("Ada argumen tapi belum jelas mengandung resource "
                "(path/registry/command). Perlu dilihat contohnya.")
    return "Argumen jarang (<30%) -> MAMBAFull mungkin lemah; laporkan coverage."


def inspect_tabular(path: Path) -> dict:
    """Kalau data Anda parquet/csv agregat, cek kolomnya."""
    out = {"file": str(path)}
    try:
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq
            schema = pq.read_schema(path)
            out["columns"] = [f.name for f in schema]
        else:
            with open_maybe_gz(path) as fh:
                header = fh.readline().strip()
            out["columns"] = header.split(",")[:40]
    except Exception as e:
        out["error"] = str(e)[:200]
        return out
    cols = [c.lower() for c in out.get("columns", [])]
    out["has_argument_column"] = any(
        any(h in c for h in ("arg", "param", "path", "regkey", "cmd", "url",
                             "host", "domain", "resource"))
        for c in cols)
    out["verdict_for_this_file"] = (
        "Ada kolom argumen -> mungkin MAMBAFull." if out["has_argument_column"]
        else "Hanya kolom nama/frekuensi API -> kemungkinan MAMBALite. "
             "Cek apakah laporan CAPE mentah masih ada terpisah.")
    return out


def main():
    if len(sys.argv) < 2:
        print("Pemakaian: python check_avast_args.py /path/ke/folder-avast-ctu")
        sys.exit(1)
    root = Path(sys.argv[1]).expanduser()
    if not root.exists():
        print(f"Path tidak ada: {root}")
        sys.exit(1)

    print("=" * 70)
    print("DIAGNOSTIK AVAST-CTU — ARGUMEN API")
    print("=" * 70)

    overview = scan_directory(root)
    print("\n[1] Ikhtisar folder")
    print(json.dumps(overview, indent=2, ensure_ascii=False))

    print("\n[2] Inspeksi file contoh")
    for f in overview["sample_files"]:
        p = Path(f)
        if p.suffix == ".parquet" or "csv" in "".join(p.suffixes):
            res = inspect_tabular(p)
        else:
            res = inspect_json_report(p)
        print("-" * 70)
        print(json.dumps(res, indent=2, ensure_ascii=False))

    print("\n" + "=" * 70)
    print("Tempelkan SELURUH output di atas ke chat.")
    print("=" * 70)


if __name__ == "__main__":
    main()