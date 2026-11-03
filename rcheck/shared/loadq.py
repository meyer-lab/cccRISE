"""Load-aware job queue for the shared machine.

Runs independent, checkpointed, single-threaded jobs, adjusting how many run
at once to the machine's load and free memory instead of a fixed worker
count, so a long run neither saturates the machine nor risks an OOM:

- every TICK seconds it reads the 1-min load average and MemAvailable;
  other users' load = load - jobs we are running;
- it launches new jobs only while (others + ours) stays under
  CPU_FRAC x cores, free memory stays above RESERVE + MEM_PER_JOB per job,
  and ours < MAX_WORKERS; at most RAMP new jobs per tick;
- it never kills a running job when load rises (each job is a resumable
  unit); it just stops launching until load falls;
- each job runs in its own systemd scope (MemoryMax = 1.5 x MEM_PER_JOB,
  no swap, nice 10, 1 BLAS thread), so one job's OOM stays its own;
- a job is done when its done-file exists (the job's own atomic
  checkpoint); finished jobs are skipped on restart; a failed job is retried
  once, then reported.

JOBS file: one job per line, tab-separated: id, done-file, shell command.
Usage: python rcheck/shared/loadq.py JOBS --name s8 [--max-workers 12]
         [--cpu-frac 0.5] [--mem-per-job 4] [--reserve 32] [--ramp 2] [--tick 30]
Logs: <JOBS dir>/loadq_<name>/<id>.log and status.txt.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def mem_available_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1e6
    return 0.0


def read_jobs(path: Path) -> list[tuple[str, Path, str]]:
    jobs = []
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            jid, done, cmd = line.split("\t", 2)
            jobs.append((jid, Path(done), cmd))
    return jobs


class Queue:
    def __init__(self, args):
        self.a = args
        self.jobs = read_jobs(args.jobs)
        self.logdir = args.jobs.parent / f"loadq_{args.name}"
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.running: dict[str, subprocess.Popen] = {}
        self.attempts: dict[str, int] = {}
        self.failed: list[str] = []
        self.stopping = False
        self.ncpu = os.cpu_count() or 1

    def scope(self, jid: str) -> str:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in jid)
        return f"loadq-{self.a.name}-{safe}.scope"

    def launch(self, jid: str, cmd: str) -> None:
        env = dict(
            os.environ,
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            PYTHONUNBUFFERED="1",
        )
        cap = f"{int(self.a.mem_per_job * 1.5 * 1024)}M"
        argv = [
            "systemd-run",
            "--user",
            "--scope",
            "-q",
            "--unit",
            self.scope(jid),
            "-p",
            f"MemoryMax={cap}",
            "-p",
            "MemorySwapMax=0",
            "nice",
            "-n",
            "10",
            "bash",
            "-c",
            cmd,
        ]
        # the child keeps its own copy of the log descriptor
        with open(self.logdir / f"{jid}.log", "a") as log:
            log.write(f"\n== {time.strftime('%F %T')} start: {cmd}\n")
            log.flush()
            self.running[jid] = subprocess.Popen(
                argv, stdout=log, stderr=subprocess.STDOUT, env=env
            )
        self.attempts[jid] = self.attempts.get(jid, 0) + 1

    def reap(self, done_files: dict[str, Path]) -> None:
        for jid, p in list(self.running.items()):
            if p.poll() is None:
                continue
            del self.running[jid]
            if done_files[jid].exists():
                continue
            if self.attempts[jid] >= 2:
                self.failed.append(jid)
                self.say(f"FAILED {jid} (exit {p.returncode}, 2 attempts)")

    def allowed(self) -> int:
        load = os.getloadavg()[0]
        others = max(0.0, load - len(self.running))
        by_cpu = int(self.ncpu * self.a.cpu_frac - others)
        by_mem = len(self.running) + int(
            (mem_available_gb() - self.a.reserve) / self.a.mem_per_job
        )
        target = max(0, min(self.a.max_workers, by_cpu, by_mem))
        return min(target, len(self.running) + self.a.ramp)

    def say(self, msg: str) -> None:
        line = f"{time.strftime('%F %T')} {msg}"
        print(line, flush=True)
        with open(self.logdir / "status.txt", "a") as f:
            f.write(line + "\n")

    def stop(self, *_):
        self.stopping = True
        self.say("TERM: no new launches; stopping running scopes (checkpointed)")
        for jid in self.running:
            subprocess.run(
                ["systemctl", "--user", "stop", self.scope(jid)], check=False
            )
        sys.exit(143)

    def run(self) -> int:
        signal.signal(signal.SIGTERM, self.stop)
        done_files = {jid: done for jid, done, _ in self.jobs}
        cmds = {jid: cmd for jid, _, cmd in self.jobs}
        pending = [jid for jid, done, _ in self.jobs if not done.exists()]
        self.say(
            f"{len(pending)} of {len(self.jobs)} jobs to run on {self.ncpu} cores "
            f"(cpu-frac {self.a.cpu_frac}, max {self.a.max_workers}, "
            f"{self.a.mem_per_job} GB/job, reserve {self.a.reserve} GB)"
        )
        last_report = 0.0
        while pending or self.running:
            self.reap(done_files)
            pending = [
                j
                for j in pending
                if not done_files[j].exists()
                and j not in self.running
                and j not in self.failed
            ]
            n = self.allowed() - len(self.running)
            while n > 0 and pending:
                jid = pending.pop(0)
                self.launch(jid, cmds[jid])
                n -= 1
            if time.time() - last_report > 300:
                done = sum(f.exists() for f in done_files.values())
                self.say(
                    f"{done}/{len(self.jobs)} done, {len(self.running)} running, "
                    f"load {os.getloadavg()[0]:.1f}, free {mem_available_gb():.0f} GB"
                )
                last_report = time.time()
            if pending or self.running:
                time.sleep(self.a.tick)
        done = sum(f.exists() for f in done_files.values())
        self.say(
            f"finished: {done}/{len(self.jobs)} done, {len(self.failed)} failed {self.failed}"
        )
        return 0 if not self.failed else 1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("jobs", type=Path)
    p.add_argument("--name", required=True)
    p.add_argument("--max-workers", type=int, default=12)
    p.add_argument(
        "--cpu-frac",
        type=float,
        default=0.5,
        help="keep total machine load (others + ours) under this x cores",
    )
    p.add_argument(
        "--mem-per-job", type=float, default=4.0, help="GB per job (scope cap 1.5x)"
    )
    p.add_argument(
        "--reserve", type=float, default=32.0, help="GB of free memory always left"
    )
    p.add_argument("--ramp", type=int, default=2, help="max new jobs per tick")
    p.add_argument("--tick", type=float, default=30.0, help="seconds between checks")
    sys.exit(Queue(p.parse_args()).run())


if __name__ == "__main__":
    main()
