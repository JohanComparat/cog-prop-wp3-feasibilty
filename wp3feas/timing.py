"""Run a command under /usr/bin/time -v and parse its report."""

import subprocess
from pathlib import Path

KEYS = {
    "Elapsed (wall clock) time (h:mm:ss or m:ss)": "wall_time_s",
    "User time (seconds)": "user_time_s",
    "System time (seconds)": "system_time_s",
    "Percent of CPU this job got": "cpu_percent",
    "Maximum resident set size (kbytes)": "max_rss_kb",
    "Exit status": "exit_status",
}


def _seconds(hms):
    parts = [float(x) for x in hms.split(":")]
    s = 0.0
    for p in parts:
        s = 60 * s + p
    return s


def parse_time_v(text):
    out = {}
    for line in text.splitlines():
        key, _, val = line.strip().rpartition(": ")
        if key in KEYS:
            name = KEYS[key]
            val = val.strip().rstrip("%")
            out[name] = _seconds(val) if name == "wall_time_s" else float(val)
    if "user_time_s" in out:
        out["cpu_time_s"] = out["user_time_s"] + out.get("system_time_s", 0.0)
    return out


def timed(cmd, cwd, log, cpus=None, env=None):
    """Run `cmd` (a list) in `cwd`, stdout and stderr to `log`; return the time -v report.

    `cpus` ("0-3,8-15") restricts the command and its children with taskset.
    """
    log = Path(log)
    tfile = log.with_suffix(".time")
    full = ["/usr/bin/time", "-v", "-o", str(tfile)]
    if cpus:
        full += ["taskset", "-c", cpus]
    full += [str(c) for c in cmd]
    with open(log, "w") as fh:
        rc = subprocess.run(full, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
    rep = parse_time_v(tfile.read_text())
    rep["returncode"] = rc
    rep["command"] = " ".join(full)
    rep["log"] = str(log)
    return rep


def host_info():
    """The machine a step ran on: host, CPU model, cores available to the process, memory, OAR job."""
    import os
    import socket
    cpu = mem = None
    try:
        cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
        mem = int(next(l.split()[1] for l in open("/proc/meminfo") if l.startswith("MemTotal"))) / 1024**2
    except (OSError, StopIteration):
        pass
    return {"hostname": socket.gethostname(), "cpu_model": cpu, "cpus_available": len(os.sched_getaffinity(0)),
            "cpus_total": os.cpu_count(), "mem_total_gib": mem, "oar_job_id": os.environ.get("OAR_JOB_ID")}


def du_bytes(paths):
    return int(sum(Path(p).stat().st_size for p in paths))


class GPUSampler:
    """Sample the GPU with nvidia-smi while a run goes on (Part C5).

        with GPUSampler(rundir / "gpu.csv", period_ms=2000) as gpu:
            rep = timed(...)
        gpu.summary()   # mean utilisation, peak memory, mean power, energy

    Each line of the CSV is: time stamp, utilisation (%), memory used (MiB), power (W).
    """

    QUERY = "timestamp,utilization.gpu,memory.used,power.draw"

    def __init__(self, path, period_ms=2000):
        self.path, self.period_ms = Path(path), period_ms

    def __enter__(self):
        self.fh = open(self.path, "w")
        self.proc = subprocess.Popen(["nvidia-smi", f"--query-gpu={self.QUERY}", "--format=csv,noheader,nounits",
                                      f"-lms={self.period_ms}"], stdout=self.fh, stderr=subprocess.DEVNULL)
        return self

    def __exit__(self, *exc):
        self.proc.terminate()
        self.proc.wait()
        self.fh.close()
        return False

    def summary(self):
        rows = [line.split(", ") for line in self.path.read_text().splitlines() if line.count(",") == 3]
        util, mem, power = ([float(r[i]) for r in rows if r[i].strip() not in ("[N/A]", "")] for i in (1, 2, 3))
        dt = self.period_ms / 1000.0
        return {"n_samples": len(rows), "period_s": dt,
                "mean_util_percent": sum(util) / len(util) if util else None,
                "max_memory_mib": max(mem) if mem else None,
                "mean_power_w": sum(power) / len(power) if power else None,
                "energy_wh": sum(power) * dt / 3600.0 if power else None,
                "csv": str(self.path)}
