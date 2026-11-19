"""Sandboxing primitives: a filesystem jail and a resource-limited Python runner.

FsJail
    Resolves client-supplied paths against a root directory. The *real* path
    (all symlinks resolved) must stay inside the root, so ``../`` traversal,
    absolute paths elsewhere and symlinks pointing outside are all rejected.
    Final opens use ``O_NOFOLLOW`` to narrow the check-then-open race.

PythonSandbox
    Runs untrusted code in a fresh interpreter with layered defenses:

    1. process isolation: ``python -I -S`` (no env vars, no user site, no
       site-packages, script dir not on ``sys.path``), *empty* environment,
       new session/process group, private temp working directory;
    2. kernel resource limits (``setrlimit``) applied by a bootstrap that then
       ``exec``s the interpreter: CPU seconds, file size (caps output and disk
       writes), open files, processes, and address space/data where the OS
       enforces them;
    3. a wall-clock timeout that SIGKILLs the whole process group;
    4. an in-interpreter PEP 578 audit hook that denies process creation,
       sockets, ``ctypes``, and file access outside the work dir (reads are
       also allowed from the stdlib so imports work).

    This is defense in depth for a portfolio server, not a substitute for a
    container/VM/seccomp boundary against a determined attacker.
"""
from __future__ import annotations

import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path


class SandboxViolation(PermissionError):
    """A request tried to leave its sandbox."""


# ---------------------------------------------------------------------------
# Filesystem jail
# ---------------------------------------------------------------------------
@dataclass
class FsJail:
    """Confines file operations to ``root``."""

    root: Path
    max_read_bytes: int = 1_000_000
    max_write_bytes: int = 1_000_000

    def __post_init__(self) -> None:
        self.root = Path(os.path.realpath(self.root))
        if not self.root.is_dir():
            raise ValueError(f"jail root {self.root} is not a directory")

    def _inside(self, real: str) -> bool:
        root = str(self.root)
        return real == root or real.startswith(root + os.sep)

    def resolve(self, user_path: str) -> Path:
        """Map a client path to a real path inside the jail or raise.

        Absolute paths are interpreted relative to the jail root (``/a`` ==
        ``a``), which is what clients usually mean and never escapes.
        """
        if not isinstance(user_path, str) or "\x00" in user_path:
            raise SandboxViolation("invalid path")
        rel = user_path.lstrip("/\\") or "."
        candidate = os.path.join(str(self.root), rel)
        real = os.path.realpath(candidate)
        if not self._inside(real):
            raise SandboxViolation(f"path escapes sandbox root: {user_path!r}")
        return Path(real)

    def relative(self, real: Path) -> str:
        rel = os.path.relpath(real, self.root)
        return "." if rel == "." else rel.replace(os.sep, "/")

    def read_text(self, user_path: str, max_bytes: int | None = None) -> tuple[str, int, bool]:
        """Read a file; returns ``(text, size, truncated)``."""
        path = self.resolve(user_path)
        limit = min(max_bytes or self.max_read_bytes, self.max_read_bytes)
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                raise IsADirectoryError(f"not a regular file: {user_path!r}")
            with os.fdopen(fd, "rb", closefd=False) as fh:
                data = fh.read(limit + 1)
        finally:
            os.close(fd)
        truncated = len(data) > limit
        return data[:limit].decode("utf-8", errors="replace"), st.st_size, truncated

    def write_text(self, user_path: str, content: str, overwrite: bool = False) -> int:
        """Write a file (creating parent dirs inside the jail). Returns bytes written."""
        data = content.encode("utf-8")
        if len(data) > self.max_write_bytes:
            raise ValueError(f"content exceeds {self.max_write_bytes} bytes")
        path = self.resolve(user_path)
        if path == self.root:
            raise SandboxViolation("cannot write to the jail root")
        parent = self.resolve(self.relative(path.parent))
        parent.mkdir(parents=True, exist_ok=True)
        # Re-check after mkdir: a racing symlink swap must not redirect us.
        if not self._inside(os.path.realpath(parent)):
            raise SandboxViolation("parent directory escapes sandbox root")
        flags = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | (os.O_TRUNC if overwrite else os.O_EXCL)
        try:
            fd = os.open(path, flags, 0o644)
        except FileExistsError:
            raise FileExistsError(f"{user_path!r} exists (pass overwrite=true)") from None
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        return len(data)

    def list_dir(self, user_path: str = ".", limit: int = 1000) -> list[dict]:
        """List a directory; symlinks are reported but never followed."""
        path = self.resolve(user_path)
        if not path.is_dir():
            raise NotADirectoryError(f"not a directory: {user_path!r}")
        entries = []
        for entry in sorted(os.scandir(path), key=lambda e: e.name)[:limit]:
            if entry.is_symlink():
                kind, size = "symlink", 0
            elif entry.is_dir(follow_symlinks=False):
                kind, size = "dir", 0
            else:
                kind, size = "file", entry.stat(follow_symlinks=False).st_size
            entries.append({"name": entry.name, "type": kind, "size": size})
        return entries

    def walk_files(self, limit: int = 200) -> list[str]:
        """Relative paths of regular files under the root (no symlink following)."""
        out: list[str] = []
        for dirpath, dirnames, filenames in os.walk(self.root, followlinks=False):
            dirnames.sort()
            for name in sorted(filenames):
                full = os.path.join(dirpath, name)
                if os.path.islink(full):
                    continue
                out.append(self.relative(Path(full)))
                if len(out) >= limit:
                    return out
        return out


# ---------------------------------------------------------------------------
# Python sandbox
# ---------------------------------------------------------------------------
_BOOTSTRAP = r"""
import json, os, resource, sys
cfg = json.loads(sys.argv[1])
applied, failed = {}, {}
for name, value in cfg["limits"].items():
    rl = getattr(resource, name, None)
    if rl is None:
        failed[name] = "unsupported"
        continue
    try:
        soft, hard = resource.getrlimit(rl)
        new_hard = value if hard == resource.RLIM_INFINITY or value <= hard else hard
        resource.setrlimit(rl, (min(value, new_hard), new_hard))
        applied[name] = value
    except (ValueError, OSError) as e:
        failed[name] = str(e)
with open(cfg["limits_report"], "w") as fh:
    json.dump({"applied": applied, "failed": failed}, fh)
os.execv(cfg["python"], [cfg["python"], "-I", "-S", "-B", cfg["runner"]])
"""

_RUNNER = r"""
import os, sys

with open("main.py", encoding="utf-8") as _fh:
    _SRC = _fh.read()

def _install():
    work = os.path.realpath(os.getcwd())
    read_roots = tuple(sorted({os.path.realpath(p) for p in sys.path if p and os.path.exists(p)}))
    dev_ok = ("/dev/null", "/dev/urandom", "/dev/random")
    deny_events = frozenset({
        "subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn",
        "os.fork", "os.forkpty", "os.kill", "os.killpg", "os.chroot", "os.setuid",
        "socket.__new__", "socket.connect", "socket.bind", "socket.getaddrinfo",
        "socket.gethostbyname", "socket.gethostbyaddr", "socket.sendto",
        "ctypes.dlopen", "ctypes.dlsym", "ctypes.cdata", "sys.addaudithook",
        "sys.setprofile", "sys.settrace", "pty.spawn", "webbrowser.open",
    })
    deny_imports = frozenset({
        "ctypes", "_ctypes", "gc", "_posixsubprocess", "subprocess", "_interpreters",
        "_xxsubinterpreters", "_interpqueues", "concurrent.interpreters", "socket", "_socket",
        "multiprocessing", "_multiprocessing", "pty", "fcntl", "resource",
    })
    path_events = frozenset({
        "os.remove", "os.rmdir", "os.mkdir", "os.chmod", "os.chown", "os.utime",
        "os.truncate", "os.listdir", "os.scandir", "os.chdir", "os.chflags", "os.setxattr",
        "os.removexattr", "shutil.rmtree", "os.walk", "glob.glob",
    })
    two_path_events = frozenset({"os.rename", "os.link", "os.symlink", "shutil.copyfile", "shutil.move"})
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC

    def under(p, roots):
        return any(p == r or p.startswith(r + os.sep) for r in roots)

    def real(p):
        if isinstance(p, bytes):
            p = os.fsdecode(p)
        return os.path.realpath(os.fspath(p))

    def check_path(event, p, write):
        if p is None or isinstance(p, int):
            return
        rp = real(p)
        if under(rp, (work,)):
            return
        if not write and (under(rp, read_roots) or rp in dev_ok):
            return
        raise PermissionError(f"sandbox: {event} denied for {os.fspath(p)!r}")

    def hook(event, args):
        if event in deny_events or event.startswith("os.spawn") or event.startswith("os.exec"):
            raise PermissionError(f"sandbox: {event} is not permitted")
        if event == "import":
            name = args[0] or ""
            if name in deny_imports or name.split(".")[0] in deny_imports:
                raise PermissionError(f"sandbox: import of {name!r} is not permitted")
            return
        if event == "open":
            path, mode, flags = args
            write = bool(mode and any(c in mode for c in "wax+")) or bool(
                isinstance(flags, int) and flags & write_flags)
            check_path(event, path, write)
        elif event in path_events and args:
            check_path(event, args[0], event not in ("os.listdir", "os.scandir", "os.chdir", "os.walk", "glob.glob"))
        elif event in two_path_events and len(args) >= 2:
            check_path(event, args[0], event != "shutil.copyfile")
            check_path(event, args[1], True)

    sys.addaudithook(hook)

_install()
del _install
_g = {"__name__": "__main__", "__builtins__": __builtins__}
exec(compile(_SRC, "<sandbox>", "exec"), _g)
"""


@dataclass
class ExecResult:
    """Outcome of a sandboxed run."""

    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    output_truncated: bool
    duration_ms: float
    signal: str | None = None
    memory_exceeded: bool = False
    peak_rss_bytes: int = 0
    limits_applied: dict = field(default_factory=dict)
    limits_failed: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and not self.memory_exceeded


def _rss_bytes(pid: int) -> int:
    """Resident set size of ``pid`` via ``ps`` (portable across macOS/Linux)."""
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=2, check=False
        ).stdout.strip()
        return int(out) * 1024 if out else 0
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return 0


@dataclass
class PythonSandbox:
    """Runs Python source in a constrained child interpreter.

    Memory: on Linux ``RLIMIT_AS`` is enforced by the kernel. macOS rejects
    lowering AS/DATA limits, so there a parent-side watchdog samples the
    child's RSS every ``rss_poll_seconds`` and SIGKILLs it past the budget.
    """

    cpu_seconds: int = 5
    wall_seconds: float = 10.0
    memory_bytes: int = 256 * 1024 * 1024
    max_output_bytes: int = 64 * 1024
    max_open_files: int = 64
    rss_poll_seconds: float = 0.05
    python: str = sys.executable

    def limits(self) -> dict[str, int]:
        """The rlimits the bootstrap will try to apply."""
        lim = {
            "RLIMIT_CPU": int(self.cpu_seconds),
            # Output files and any file the code writes are capped by this.
            "RLIMIT_FSIZE": int(self.max_output_bytes) + 1,
            "RLIMIT_NOFILE": int(self.max_open_files),
            "RLIMIT_NPROC": 0,  # the child may not create further processes
            "RLIMIT_CORE": 0,
        }
        if sys.platform.startswith("linux"):
            lim["RLIMIT_AS"] = int(self.memory_bytes)
        return lim

    def run(self, code: str, wall_seconds: float | None = None) -> ExecResult:
        """Execute ``code``; never raises for misbehaving user code."""
        wall = min(float(wall_seconds or self.wall_seconds), self.wall_seconds)
        with tempfile.TemporaryDirectory(prefix="mcp-sbx-") as tmp:
            work = os.path.join(tmp, "work")
            ctl = os.path.join(tmp, "ctl")
            os.mkdir(work, 0o700)
            os.mkdir(ctl, 0o700)
            Path(work, "main.py").write_text(code, encoding="utf-8")
            runner = os.path.join(ctl, "runner.py")
            Path(runner).write_text(_RUNNER, encoding="utf-8")
            report = os.path.join(ctl, "limits.json")
            cfg = json.dumps({"limits": self.limits(), "limits_report": report, "python": self.python, "runner": runner})
            out_path, err_path = os.path.join(ctl, "stdout"), os.path.join(ctl, "stderr")
            start = time.perf_counter()
            timed_out = memory_exceeded = False
            peak_rss = 0
            with open(out_path, "wb") as out_fh, open(err_path, "wb") as err_fh:
                proc = subprocess.Popen(
                    [self.python, "-I", "-S", "-B", "-c", _BOOTSTRAP, cfg],
                    cwd=work,
                    env={},
                    stdin=subprocess.DEVNULL,
                    stdout=out_fh,
                    stderr=err_fh,
                    start_new_session=True,
                    close_fds=True,
                )
                deadline = start + wall
                watch_rss = "RLIMIT_AS" not in self.limits()
                while True:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        timed_out = True
                        break
                    try:
                        proc.wait(timeout=min(self.rss_poll_seconds, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        pass
                    if watch_rss:
                        rss = _rss_bytes(proc.pid)
                        peak_rss = max(peak_rss, rss)
                        if rss > self.memory_bytes:
                            memory_exceeded = True
                            break
                if proc.returncode is None:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait()
            duration = (time.perf_counter() - start) * 1000
            stdout, t1 = self._read_capped(out_path)
            stderr, t2 = self._read_capped(err_path)
            stderr = stderr.replace(runner, "<sandbox-runtime>")
            sig = None
            rc = proc.returncode
            if rc is not None and rc < 0:
                try:
                    sig = signal.Signals(-rc).name
                except ValueError:
                    sig = str(-rc)
            truncated = t1 or t2 or sig == "SIGXFSZ" or "[Errno 27] File too large" in stderr
            try:
                limits = json.loads(Path(report).read_text())
            except (OSError, ValueError):
                limits = {"applied": {}, "failed": {}}
            if watch_rss:
                limits.setdefault("applied", {})["RSS_WATCHDOG"] = int(self.memory_bytes)
            if timed_out:
                stderr += f"\n[sandbox] wall-clock timeout after {wall:.1f}s"
            elif memory_exceeded:
                stderr += f"\n[sandbox] memory limit exceeded ({self.memory_bytes} bytes)"
            elif sig == "SIGXCPU":
                stderr += "\n[sandbox] CPU time limit exceeded"
            elif truncated:
                stderr += "\n[sandbox] output/file size limit exceeded"
            return ExecResult(
                exit_code=rc,
                stdout=stdout,
                stderr=stderr,
                timed_out=timed_out,
                output_truncated=truncated,
                duration_ms=round(duration, 2),
                signal=sig,
                memory_exceeded=memory_exceeded,
                peak_rss_bytes=peak_rss,
                limits_applied=limits.get("applied", {}),
                limits_failed=limits.get("failed", {}),
            )

    def _read_capped(self, path: str) -> tuple[str, bool]:
        with open(path, "rb") as fh:
            data = fh.read(self.max_output_bytes + 1)
        truncated = len(data) > self.max_output_bytes
        return data[: self.max_output_bytes].decode("utf-8", errors="replace"), truncated
