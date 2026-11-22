"""Summarise a jetson/bench.sh results folder into summary.md and summary.json.

    python3 jetson/summarize.py bench_results/<timestamp>

Reads, for each precision folder: run/run_metrics.json (the pipeline's own timings), run/tegrastats.log
(power, GPU/CPU use, memory, temperatures), run/time.txt (GNU time), cold/ (engine build), profile/
(ONNX Runtime per-operator profiles), trtexec/ (each model alone) and compare_*.txt (accuracy
against the CPU FP32 reference). Standard library only, so it runs on the Jetson or a Mac.
"""

import glob
import json
import math
import os
import re
import sys
from collections import defaultdict


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def read(path):
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def stats(values):
    v = sorted(x for x in values if x is not None and math.isfinite(x))
    if not v:
        return None
    q = lambda p: v[round((len(v) - 1) * p)]
    mean = sum(v) / len(v)
    return {"n": len(v), "mean": mean, "min": v[0], "p50": q(0.5), "p95": q(0.95), "p99": q(0.99), "max": v[-1]}


def f(x, digits=1, unit=""):
    if x is None:
        return "–"
    if isinstance(x, (int, float)):
        return f"{x:,.{digits}f}{unit}"
    return str(x)


# ---- tegrastats

RE_RAM = re.compile(r"RAM (\d+)/(\d+)MB")
RE_SWAP = re.compile(r"SWAP (\d+)/(\d+)MB")
RE_CPU = re.compile(r"CPU \[([^\]]*)\]")
RE_EMC = re.compile(r"EMC_FREQ (\d+)%")
RE_GPU = re.compile(r"GR3D_FREQ (\d+)%(?:@\[?(\d+)\]?)?")
RE_TEMP = re.compile(r"\b([A-Za-z0-9_]+)@(-?[\d.]+)C\b")
RE_RAIL = re.compile(r"\b([A-Z][A-Z0-9_]+) (\d+)mW/(\d+)mW")
RE_ENGINE = re.compile(r"\b(NVDEC|NVENC|NVJPG|VIC|OFA)\s+(\d+)%")


def tegrastats(path, interval_s=0.1):
    lines = [l for l in read(path).splitlines() if "RAM" in l]
    if not lines:
        return None
    ram, swap, cpu_avg, cpu_max_core, emc, gpu, gpu_freq = [], [], [], [], [], [], []
    temps, rails, engines = defaultdict(list), defaultdict(list), defaultdict(list)
    for l in lines:
        if m := RE_RAM.search(l):
            ram.append(int(m[1]))
        if m := RE_SWAP.search(l):
            swap.append(int(m[1]))
        if m := RE_CPU.search(l):
            cores = [int(c.split("%")[0]) for c in m[1].split(",") if "%" in c]
            if cores:
                cpu_avg.append(sum(cores) / len(cores))
                cpu_max_core.append(max(cores))
        if m := RE_EMC.search(l):
            emc.append(int(m[1]))
        if m := RE_GPU.search(l):
            gpu.append(int(m[1]))
            if m[2]:
                gpu_freq.append(int(m[2]))
        for name, t in RE_TEMP.findall(l):
            temps[name].append(float(t))
        for name, cur, _avg in RE_RAIL.findall(l):
            rails[name].append(int(cur) / 1000.0)
        for name, pct in RE_ENGINE.findall(l):
            engines[name].append(int(pct))
    total_m = RE_RAM.search(lines[0])
    out = {
        "samples": len(lines),
        "seconds": len(lines) * interval_s,
        "ram_total_mb": int(total_m[2]) if total_m else None,
        "ram_used_mb": stats(ram),
        "swap_used_mb": stats(swap),
        "cpu_util_pct": stats(cpu_avg),
        "cpu_busiest_core_pct": stats(cpu_max_core),
        "emc_util_pct": stats(emc),
        "gpu_util_pct": stats(gpu),
        "gpu_freq_mhz": stats(gpu_freq),
        "temps_c": {k: stats(v) for k, v in temps.items()},
        "power_w": {k: stats(v) for k, v in rails.items()},
        "engines_pct": {k: stats(v) for k, v in engines.items()},
    }
    # Total board power: VDD_IN on the Orin Nano / NX; otherwise the sum of the rails.
    if "VDD_IN" in rails:
        out["board_power_w"] = stats(rails["VDD_IN"])
    elif rails:
        n = min(len(v) for v in rails.values())
        out["board_power_w"] = stats([sum(v[i] for v in rails.values()) for i in range(n)])
    return out


# ---- GNU time -v

def gnu_time(path):
    text = read(path)
    if not text:
        return None
    kv = {}
    for line in text.splitlines():
        if ":" in line:
            k, _, v = line.strip().rpartition(": ")
            kv[k.strip()] = v.strip()
    def num(k):
        try:
            return float(kv[k].rstrip("%"))
        except (KeyError, ValueError):
            return None
    return {
        "max_rss_mb": (num("Maximum resident set size (kbytes)") or 0) / 1024 or None,
        "user_s": num("User time (seconds)"),
        "system_s": num("System time (seconds)"),
        "cpu_pct": num("Percent of CPU this job got"),
        "wall": kv.get("Elapsed (wall clock) time (h:mm:ss or m:ss)"),
        "major_page_faults": num("Major (requiring I/O) page faults"),
        "minor_page_faults": num("Minor (reclaiming a frame) page faults"),
        "voluntary_switches": num("Voluntary context switches"),
        "involuntary_switches": num("Involuntary context switches"),
        "fs_inputs": num("File system inputs"),
        "fs_outputs": num("File system outputs"),
        "exit_status": num("Exit status"),
    }


# ---- trtexec

RE_TRT_STAT = re.compile(r"\[I\] (Latency|GPU Compute Time|H2D Latency|D2H Latency|Enqueue Time|Host Latency): (.*)")
RE_TRT_KV = re.compile(r"(min|max|mean|median|percentile\(\d+%\)) = ([\d.]+) ms")


def trtexec(folder):
    out = {}
    for log in sorted(glob.glob(os.path.join(folder, "*.log"))):
        name = os.path.basename(log)[:-4]
        text = read(log)
        r = {"ok": "PASSED" in text}
        if m := re.search(r"Throughput: ([\d.]+) qps", text):
            r["throughput_qps"] = float(m[1])
        if m := re.search(r"Engine built in ([\d.]+) sec", text):
            r["engine_build_s"] = float(m[1])
        for kind, rest in RE_TRT_STAT.findall(text):
            r[kind.lower().replace(" ", "_") + "_ms"] = {k: float(v) for k, v in RE_TRT_KV.findall(rest)}
        layers = load_json(os.path.join(folder, name + ".layers.json"))
        if isinstance(layers, list):
            rows = [l for l in layers if isinstance(l, dict) and "name" in l]
            rows.sort(key=lambda l: -l.get("averageMs", 0))
            r["layers"] = len(rows)
            r["top_layers"] = [{"name": l["name"][:80], "avg_ms": l.get("averageMs"), "pct": l.get("percentage")} for l in rows[:5]]
        out[name] = r
    return out


# ---- ONNX Runtime profiles

def ort_profiles(folder):
    files = glob.glob(os.path.join(folder, "**", "*.json"), recursive=True)
    if not files:
        return None
    by_provider = defaultdict(lambda: {"nodes": 0, "us": 0})
    by_op = defaultdict(int)
    runs = defaultdict(list)
    for p in files:
        events = load_json(p) or []
        model = re.sub(r"_\d{4}-\d\d-\d\d_.*$", "", os.path.basename(p))
        for e in events:
            if not isinstance(e, dict):
                continue
            if e.get("cat") == "Node" and e.get("name", "").endswith("_kernel_time"):
                args = e.get("args", {})
                prov = args.get("provider", "?")
                by_provider[prov]["nodes"] += 1
                by_provider[prov]["us"] += e.get("dur", 0)
                by_op[(prov, args.get("op_name", "?"))] += e.get("dur", 0)
            elif e.get("cat") == "Session" and e.get("name") == "model_run":
                runs[model].append(e.get("dur", 0) / 1000.0)
    total = sum(v["us"] for v in by_provider.values()) or 1
    return {
        "files": len(files),
        "providers": {k: {"node_runs": v["nodes"], "time_pct": 100.0 * v["us"] / total} for k, v in by_provider.items()},
        "top_ops": [{"provider": k[0], "op": k[1], "time_pct": 100.0 * us / total} for k, us in sorted(by_op.items(), key=lambda x: -x[1])[:8]],
        "session_run_ms": {k: stats(v[1:]) for k, v in runs.items() if len(v) > 1},
    }


# ---- compare_*.txt

def compare_text(folder):
    out = {}
    for p in sorted(glob.glob(os.path.join(folder, "compare_*.txt"))):
        out[os.path.basename(p)[8:-4]] = read(p).strip()
    return out


# ---- one precision

def precision(folder):
    clip_metrics = glob.glob(os.path.join(folder, "run", "*", "run_metrics.json"))
    run = load_json(clip_metrics[0]) if clip_metrics else None
    cold = glob.glob(os.path.join(folder, "cold", "*", "run_metrics.json"))
    cold = load_json(cold[0]) if cold else None
    quality = glob.glob(os.path.join(folder, "quality", "*", "run_metrics.json"))
    quality = load_json(quality[0]) if quality else None
    tegra = tegrastats(os.path.join(folder, "run", "tegrastats.log"))
    s = {
        "run": run,
        "cold_load": cold and cold.get("load"),
        "cold_first_frame_s": cold and cold["detect"].get("first_frame_s"),
        "quality_run": quality and {"fps": quality["detect"]["fps"], "rink_fit": quality.get("rink_fit")},
        "tegrastats": tegra,
        "gnu_time": gnu_time(os.path.join(folder, "run", "time.txt")),
        "trtexec": trtexec(os.path.join(folder, "trtexec")),
        "ort_profile": ort_profiles(os.path.join(folder, "profile", "ort")),
        "compare": compare_text(folder),
    }
    if run and tegra and tegra.get("board_power_w"):
        w = tegra["board_power_w"]["mean"]
        frames = run["detect"]["frames"]
        wall = run["end_to_end"]["s"]
        s["energy"] = {
            "board_w_mean": w,
            "board_w_max": tegra["board_power_w"]["max"],
            "joules": w * wall,
            "joules_per_frame": w * wall / max(frames, 1),
            "fps_per_watt": run["end_to_end"]["fps"] / w if w else None,
        }
    return s


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("bench_results/*"))[-1]
    precs = [p for p in ("fp32", "tf32", "fp16") if os.path.isdir(os.path.join(root, p))]
    summary = {
        "folder": root,
        "system": {os.path.basename(p)[:-4]: read(p).strip() for p in glob.glob(os.path.join(root, "system", "*.txt"))},
        "idle": tegrastats(os.path.join(root, "idle_tegrastats.log")),
        "precisions": {p: precision(os.path.join(root, p)) for p in precs},
    }
    ref = glob.glob(os.path.join(root, "reference_cpu", "*", "run_metrics.json"))
    summary["cpu_reference"] = load_json(ref[0]) if ref else None
    idle_w = summary["idle"] and summary["idle"].get("board_power_w") and summary["idle"]["board_power_w"]["mean"]
    for p in summary["precisions"].values():
        if idle_w and p.get("energy"):
            p["energy"]["idle_w"] = idle_w
            p["energy"]["above_idle_w"] = p["energy"]["board_w_mean"] - idle_w

    with open(os.path.join(root, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    md = render(summary)
    with open(os.path.join(root, "summary.md"), "w") as fh:
        fh.write(md)
    print(md)


def render(s):
    P = s["precisions"]
    sysinfo = s["system"]
    L = [f"# Hockey-Vision Jetson benchmark", "", f"Folder: `{s['folder']}`", ""]
    L += ["## System", ""]
    for k in ("device", "nvpmodel", "segment", "git_commit", "onnxruntime"):
        if sysinfo.get(k):
            L += [f"**{k}**", "```", sysinfo[k][:1500], "```"]
    any_run = next((p["run"] for p in P.values() if p.get("run")), None)
    if any_run:
        L += [f"ONNX Runtime: `{(any_run.get('onnxruntime') or '')[:160]}`", ""]
        L += [f"Settings: `{json.dumps(any_run['settings'])[:600]}`", ""]

    # Headline
    L += ["## Headline", "", "| | " + " | ".join(P) + " |", "|---|" + "---|" * len(P)]
    def row(label, fn):
        vals = []
        for p in P.values():
            try:
                vals.append(fn(p))
            except (KeyError, TypeError, IndexError):
                vals.append("–")
        L.append(f"| {label} | " + " | ".join(vals) + " |")
    row("Frames", lambda p: f(p["run"]["detect"]["frames"], 0))
    row("End-to-end FPS (detect + rink fit + export)", lambda p: f(p["run"]["end_to_end"]["fps"], 2))
    row("Detect FPS", lambda p: f(p["run"]["detect"]["fps"], 2))
    row("Steady-state FPS (after warm-up)", lambda p: f(p["run"]["detect"]["steady_fps"], 2))
    row("Frame step (every Nth frame read)", lambda p: f(p["run"]["video"].get("frame_step", 1), 0))
    row("Frames needed per second for real time", lambda p: f(p["run"]["video"]["fps"] / p["run"]["video"].get("frame_step", 1), 2))
    row("Real-time factor (FPS / frames needed per second; 1.0 = real time)", lambda p: f(p["run"]["detect"]["real_time_factor"], 3, "x"))
    row("Frame interval p50 / p95 / p99 ms", lambda p: "{} / {} / {}".format(*(f(p["run"]["detect"]["frame_interval_ms"][k], 2) for k in ("p50", "p95", "p99"))))
    row("First frame (engines cached) s", lambda p: f(p["run"]["detect"]["first_frame_s"], 2))
    row("Cold run: engine build (or cache load if already built) s", lambda p: f(sum(x["s"] for m in p["cold_load"]["models"] for x in m["fixed_shape_sessions"]), 1))
    row("Engines from cache, all models s", lambda p: f(sum(x["s"] for m in p["run"]["load"]["models"] for x in m["fixed_shape_sessions"]), 2))
    row("Peak RSS MB (process)", lambda p: f(p["gnu_time"]["max_rss_mb"], 0))
    row("System RAM used, max MB", lambda p: f(p["tegrastats"]["ram_used_mb"]["max"], 0))
    row("GPU util mean / p95 %", lambda p: f"{f(p['tegrastats']['gpu_util_pct']['mean'], 0)} / {f(p['tegrastats']['gpu_util_pct']['p95'], 0)}")
    row("GPU clock mean MHz", lambda p: f(p["tegrastats"]["gpu_freq_mhz"]["mean"], 0))
    row("CPU util mean % (busiest core p95)", lambda p: f"{f(p['tegrastats']['cpu_util_pct']['mean'], 0)} ({f(p['tegrastats']['cpu_busiest_core_pct']['p95'], 0)})")
    row("Memory controller (EMC) util mean %", lambda p: f(p["tegrastats"]["emc_util_pct"]["mean"], 0))
    row("Board power mean / max W", lambda p: f"{f(p['energy']['board_w_mean'], 2)} / {f(p['energy']['board_w_max'], 2)}")
    row("Power above idle W", lambda p: f(p["energy"].get("above_idle_w"), 2))
    row("Energy per frame J", lambda p: f(p["energy"]["joules_per_frame"], 3))
    row("FPS per watt", lambda p: f(p["energy"]["fps_per_watt"], 2))
    row("Max temperature C (tj)", lambda p: f(max(v["max"] for k, v in p["tegrastats"]["temps_c"].items() if v and k.lower() in ("tj", "gpu", "cpu")), 1))
    row("User + system CPU s", lambda p: f(p["gnu_time"]["user_s"] + p["gnu_time"]["system_s"], 1))
    row("Rink fit rate", lambda p: f(100 * p["run"]["rink_fit"]["fit_rate"], 1, "%"))
    if s.get("cpu_reference"):
        cpu_fps = s["cpu_reference"]["detect"]["fps"]
        L.append(f"| CPU FP32 reference FPS ({s['cpu_reference']['detect']['frames']} frames) | " + " | ".join(f(cpu_fps, 2) for _ in P) + " |")
        row("Speed-up vs CPU", lambda p: f(p["run"]["detect"]["fps"] / cpu_fps, 1, "x"))
    if s.get("idle") and s["idle"].get("board_power_w"):
        L += ["", f"Idle board power: {f(s['idle']['board_power_w']['mean'], 2)} W"]

    for name, p in P.items():
        r = p.get("run")
        if not r:
            L += ["", f"## {name}: no run_metrics.json (see {name}/run/stderr.txt)"]
            continue
        d = r["detect"]
        L += ["", f"## {name}", "", "### Per model (pipeline, after warm-up) vs trtexec (model alone)", "",
              "| model | pipeline p50 | p95 | p99 | first run | trtexec GPU compute mean | trtexec latency p99 | trtexec qps | boxes/frame |",
              "|---|---|---|---|---|---|---|---|---|"]
        trt = p.get("trtexec") or {}
        files = {"player": "new_player_model", "puck": "new_nano_puck", "rink": "new_rink_model", "dots": "new_dots", "number": "new_nums"}
        for m, v in d["models"].items():
            t = trt.get(files.get(m, m), {})
            ms = v.get("ms") or {}
            gpu = (t.get("gpu_compute_time_ms") or {}).get("mean")
            lat = (t.get("latency_ms") or {}).get("percentile(99%)")
            if m == "number":
                L.append(f"| number ({v['crops']} crops, {f(v['ms_per_crop'], 2)} ms/crop) | {f((v.get('ms_per_frame') or {}).get('p50'), 2)}/frame | {f((v.get('ms_per_frame') or {}).get('p95'), 2)} | {f((v.get('ms_per_frame') or {}).get('p99'), 2)} | – | {f(gpu, 2)} | {f(lat, 2)} | {f(t.get('throughput_qps'), 0)} | {f((v.get('crops_per_frame') or {}).get('mean'), 1)} crops |")
            else:
                L.append(f"| {m} | {f(ms.get('p50'), 2)} | {f(ms.get('p95'), 2)} | {f(ms.get('p99'), 2)} | {f(v.get('first_ms'), 0)} | {f(gpu, 2)} | {f(lat, 2)} | {f(t.get('throughput_qps'), 0)} | {f((v.get('boxes_per_frame') or {}).get('mean'), 1)} |")
        L += ["", "### Stages per frame (ms, after warm-up; parallel stages overlap)", "", "| stage | mean | p50 | p95 | p99 | max |", "|---|---|---|---|---|---|"]
        for k, v in d["stages_ms"].items():
            if v:
                L.append(f"| {k} | {f(v['mean'], 2)} | {f(v['p50'], 2)} | {f(v['p95'], 2)} | {f(v['p99'], 2)} | {f(v['max'], 2)} |")
        for st in r.get("later_stages", []):
            L.append(f"| {st['stage']} (whole clip {f(st['s'], 2)} s) | {f(st['ms_per_frame'], 3)} | | | | |")
        prof = p.get("ort_profile")
        if prof:
            L += ["", "### Where operators ran (ONNX Runtime profile)", ""]
            for k, v in prof["providers"].items():
                L.append(f"- {k}: {v['node_runs']} operator runs, {f(v['time_pct'], 1)}% of operator time")
            L += ["", "Top operators: " + ", ".join(f"{o['op']} on {o['provider'].replace('ExecutionProvider', '')} {f(o['time_pct'], 1)}%" for o in prof["top_ops"])]
        tops = [(m, t["top_layers"]) for m, t in trt.items() if t.get("top_layers")]
        if tops:
            L += ["", "### Slowest TensorRT layers per model (trtexec)", ""]
            for m, layers in tops:
                L.append(f"- {m}: " + "; ".join(f"{l['name']} {f(l['avg_ms'], 3)} ms" for l in layers[:3]))
        tg = p.get("tegrastats")
        if tg:
            L += ["", "### Temperatures and power rails (measured run)", ""]
            L.append("Temperatures C (mean / max): " + ", ".join(f"{k} {f(v['mean'], 1)}/{f(v['max'], 1)}" for k, v in tg["temps_c"].items() if v))
            L.append("")
            L.append("Rails W (mean / max): " + ", ".join(f"{k} {f(v['mean'], 2)}/{f(v['max'], 2)}" for k, v in tg["power_w"].items() if v))
        if p.get("compare"):
            L += ["", "### Accuracy against the CPU FP32 reference", ""]
            for k, text in p["compare"].items():
                L += [f"**{k}**", "```", text[:3000], "```"]

    best = max(((n, p) for n, p in P.items() if p.get("run")), key=lambda x: x[1]["run"]["end_to_end"]["fps"], default=None)
    if best:
        n, p = best
        e = p.get("energy") or {}
        L += ["", "## Resume line", "",
              f"... via TensorRT ({n.upper()}) on an NVIDIA Jetson Orin Nano at {p['run']['end_to_end']['fps']:.0f} FPS end to end"
              + (f" ({e['board_w_mean']:.1f} W, {e['fps_per_watt']:.1f} FPS/W)" if e.get("board_w_mean") else "")
              + f" on 1080p{round(p['run']['video']['fps'])} video"
              + (f" (every {p['run']['video']['frame_step']}th frame, {p['run']['detect']['real_time_factor']:.2f}x real time)"
                 if p['run']['video'].get('frame_step', 1) > 1 else "")]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
