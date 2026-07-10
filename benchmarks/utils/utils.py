import subprocess
import os
import logging
import signal
import sys
import atexit
import argparse


# Global flag to track E-core state
cpu_modified = False
# Remember the *exact* Bazel command we executed, to add to CSV
last_build_cmd = ""


def get_base_parser():
    """Create base argument parser with common arguments."""
    parser = argparse.ArgumentParser(add_help=False)  # add_help=False to avoid duplicate help
    
    # Common arguments
    parser.add_argument("--input_mode", choices=["uniform", "trunk", "csv"], default="trunk")
    parser.add_argument("--train_csv")
    parser.add_argument("--label_col")
    parser.add_argument("--experiment_name", default="")
    parser.add_argument("--feature_split_type", default="Oblique",
                       choices=["Axis Aligned", "Oblique"])
    parser.add_argument("--numerical_split_type", default="Dynamic Random Histogram",
                       choices=["Exact", "Random", "Equal Width",
                                "Dynamic Random Histogram", "Dynamic Equal Width Histogram"])
    parser.add_argument("--vectorized", choices=["None", "avx2", "avx512"], default="avx2")
    parser.add_argument("--tree_depth", type=int, default=-1)
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--num_trees", type=int, default=1)
    parser.add_argument("--projection_density_factor", type=int)
    parser.add_argument("--max_num_projections", type=int)
    parser.add_argument("--sample_projection_mode", choices=["Fast", "Slow"], default="Fast") # TODO deprecate
    parser.add_argument("--fixed_1000_projections", action="store_true")
    parser.add_argument("--depthwise_1_pass", action="store_true",
                        help="Build with -DDEPTHWISE_1_PASS=1: Depthwise fused "
                             "per-level CPU ApplyProjection (single-pass "
                             "kernel across all (row, projection) tasks at "
                             "the level; thread-parallel, contention-free).")
    parser.add_argument("--symmetric_depthwise_ap", action="store_true",
                        help="Build with -DSYMMETRIC_DEPTHWISE_AP=1: "
                             "CatBoost-style symmetric-trees bag-wide "
                             "ApplyProjection (K stride-1 sweeps over the "
                             "sorted bag, shared projections per depth).")
    parser.add_argument("--dataset_layout",
                        choices=["column", "row", "dynamic_row_col_major",
                                 "dynamic_row_col_major_bf16", "dual_bf16", "dual_fp32"],
                        default="column",
                        help="Synthetic trunk dataset feature storage layout.")
    # parser.add_argument("--enable_fast_equal_width_binning", action="store_true") # This is on by default now
    parser.add_argument("--use_gpu", type=lambda x: x.lower() in ("true", "1", "yes"),
                       default=False, help="Use GPU for oblique projections (default: false)")
    parser.add_argument("--bazel_config", action="append", default=[], metavar="NAME",
                       help="Extra --config=NAME to pass to the bazel build. Repeatable: "
                            "--bazel_config=enable_applyprojection_isnan. "
                            "Applied after the flag-driven configs (avx2, "
                            "depthwise_1_pass, ...).")

    return parser


# CPUs that set_cpu_e_features.sh --disable takes offline on the 185H.
_HT_SIBLING_CPUS = [2, 4, 5, 7, 9, 11]
_E_CORE_CPUS = list(range(12, 22))
_NO_TURBO_PATH = "/sys/devices/system/cpu/intel_pstate/no_turbo"


def _read_sys_file(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _is_pcore_only_state():
    """Whether the CPU is already in the state set_cpu_e_features.sh --disable
    produces. Missing /sys files are ignored, matching the shell script."""
    for cpu in _HT_SIBLING_CPUS + _E_CORE_CPUS:
        state = _read_sys_file(f"/sys/devices/system/cpu/cpu{cpu}/online")
        if state is not None and state != "0":
            return False

    turbo = _read_sys_file(_NO_TURBO_PATH)
    if turbo is not None and turbo != "1":
        return False

    return True


def configure_cpu_for_benchmarks(enable_pcore_only=True):
    """
    Configure CPU for benchmarking.

    Args:
        enable_pcore_only: If True, disable HT/E-cores/turbo. If False, restore all.

    If --disable is requested and the machine is already in that state, skip
    the sudo call entirely and leave cpu_modified=False so cleanup won't
    re-enable what the user pre-staged.
    """
    global cpu_modified

    if get_cpu_model_proc() != "Intel(R) Core(TM) Ultra 9 185H":
        print("Skipping changing CPU E-features. CPU not Intel Core Ultra 9 185H")
        return

    if enable_pcore_only and _is_pcore_only_state():
        print("CPU already in P-cores-only state (HT/E-cores offline, turbo off); "
              "skipping sudo. State will be left unchanged on exit.")
        return True

    action = "--disable" if enable_pcore_only else "--enable"
    script_path = os.path.realpath(
        os.path.join(os.path.dirname(__file__), "set_cpu_e_features.sh")
    )
    cmd = ["sudo", script_path, action]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        cpu_modified = enable_pcore_only
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to configure CPU: {e}")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        sys.exit(1)

def cleanup_and_exit(signum=None, frame=None):
    """Cleanup function to restore CPU configuration before exiting"""
    global cpu_modified
    if cpu_modified:
        print("\nCleaning up: Restoring CPU configuration...")
        configure_cpu_for_benchmarks(False)  # This will set cpu_modified = False
    if signum is not None:
        print(f"\nReceived signal {signum}, exiting cleanly.")
        sys.exit(1)


def setup_signal_handlers():
    """Setup signal handlers for graceful cleanup"""
    signal.signal(signal.SIGINT, cleanup_and_exit)   # Ctrl+C
    signal.signal(signal.SIGTERM, cleanup_and_exit)  # Termination signal
    atexit.register(cleanup_and_exit)  # Fallback for other exit scenarios


def build_binary(args, chrono_mode):
    """Build the binary using bazel. Returns True if successful, False otherwise."""
    
    base_cmd = ['bazel', 'build', '--ui_event_filters=-warning',
                '-c', 'opt']
    if args.fixed_1000_projections:
        base_cmd.append('--config=fixed_1000_projections')
    finished_cmd = base_cmd[:] # ← work on a copy

    if args.vectorized == "avx2":
        finished_cmd.append('--config=enable_std_upper_bound_avx2')
    elif args.vectorized == "avx512":
        finished_cmd.append('--config=enable_std_upper_bound_avx512')
        
    if args.sample_projection_mode == "Slow":
        finished_cmd.append('--config=slow_sample_projections')
    
    if chrono_mode:
        # Two-tier chrono: level 2 (fine, default) instruments every scope;
        # level 1 (coarse) keeps only the top-level scopes for lower overhead.
        if getattr(args, 'chrono_level', 2) == 1:
            finished_cmd.append('--config=chrono_profile_coarse')
        else:
            finished_cmd.append('--config=chrono_profile')

    # --use_gpu=true requires the GPU code paths to be compiled in. Without
    # --config=oblique_gpu the OBLIQUE_GPU_ENABLED macro is undefined and
    # nodewise/depthwise-gpu dispatch is compiled out (binary defaults to CPU).
    if getattr(args, 'use_gpu', False):
        finished_cmd.append('--config=oblique_gpu')

    _ap_variants = [
        ('--depthwise_1_pass',
         getattr(args, 'depthwise_1_pass', False)),
        ('--symmetric_depthwise_ap',
         getattr(args, 'symmetric_depthwise_ap', False)),
    ]
    _on = [name for name, v in _ap_variants if v]
    if len(_on) > 1:
        raise ValueError(
            f"{' / '.join(_on)} are mutually exclusive "
            "(matches the C++ #error in label.h).")
    if getattr(args, 'depthwise_1_pass', False):
        finished_cmd.append('--config=depthwise_1_pass')
    if getattr(args, 'symmetric_depthwise_ap', False):
        finished_cmd.append('--config=symmetric_depthwise_ap')
    if getattr(args, 'dataset_layout', 'column') in ('row', 'dynamic_row_col_major', 'dynamic_row_col_major_bf16', 'dual_bf16', 'dual_fp32'):
        finished_cmd.append('--config=row_major_dataset_layout')
    if getattr(args, 'gpu_mode', None) == 'per_node':
        finished_cmd.append('--config=dfs_node_queue')

    # if args.enable_fast_equal_width_binning:
    #     finished_cmd.append('--config=enable_fast_equal_width_binning')

    for extra_config in getattr(args, 'bazel_config', []) or []:
        finished_cmd.append(f'--config={extra_config}')

    finished_cmd.append("--ui_event_filters=-warning")

    # Pin Intel oneAPI ICX (clang-based) regardless of branch/.bazelrc state.
    # Why: main's .bazelrc does not pin a compiler (auto-detects gcc);
    # rebased-main's pins icx but fails when oneAPI is not sourced. Injecting
    # an absolute path here makes both branches build with icx without any
    # shell prep.
    _icx = "/opt/intel/oneapi/compiler/latest/bin/icx"
    _icpx = "/opt/intel/oneapi/compiler/latest/bin/icpx"
    if os.path.isfile(_icx) and os.path.isfile(_icpx):
        finished_cmd.append(f"--repo_env=CC={_icx}")
        finished_cmd.append(f"--repo_env=CXX={_icpx}")

    finished_cmd.append('//examples:train_oblique_forest')

    global last_build_cmd
    last_build_cmd = " ".join(finished_cmd)

    # Ensure oneAPI bin is on PATH for Bazel's cc_configure regardless of
    # whether the user sourced setvars.sh in their shell.
    build_env = os.environ.copy()
    _oneapi_bin = "/opt/intel/oneapi/compiler/latest/bin"
    if os.path.isdir(_oneapi_bin) and _oneapi_bin not in build_env.get("PATH", ""):
        build_env["PATH"] = f"{_oneapi_bin}:{build_env.get('PATH', '')}"

    print("Building binary...")
    print(f"Running: {' '.join(finished_cmd)}")

    try:
        result = subprocess.run(
            finished_cmd,
            capture_output=False,
            text=True,
            check=True,
            env=build_env,
            cwd=os.getcwd()         # Explicitly set working directory
        )
        
        print("✅ Build succeeded!")
        if result.stdout:
            logging.info(f"Build stdout:\n{result.stdout}")
        if result.stderr:
            logging.info(f"Build stderr:\n{result.stderr}")
        return True
        
    except subprocess.CalledProcessError as e:
        print("❌ Build failed!")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"Build stdout:\n{e.stdout}")
        if e.stderr:
            print(f"Build stderr:\n{e.stderr}")
        return False
    
    except KeyboardInterrupt:
        print("\n❌ Build interrupted by user")
        return False
    
    except Exception as e:
        print(f"❌ Unexpected error during build: {e}")
        return False


def get_cpu_model_proc():
    """
    Returns the CPU model name. Reads /proc/cpuinfo on Linux; falls back to
    `sysctl machdep.cpu.brand_string` on macOS (no /proc there) so the results
    directory gets a real device name instead of an error string.
    """
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name"):
                    # split only on the first ':' → [key, value]
                    return line.split(":", 1)[1].strip()
    except FileNotFoundError:
        pass

    # macOS / BSD: /proc/cpuinfo doesn't exist.
    try:
        import subprocess
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True, capture_output=True, check=False).stdout.strip()
        if out:
            return out
    except (OSError, FileNotFoundError):
        pass

    return "Unknown_CPU"


def get_machine_serial():
    """Hardware serial for traceability -- the same value runtime.sh records as
    `machine_serial`. dmidecode needs root and is Linux-only; if it fails (no
    sudo, not installed, non-Linux) keep the error text in the field rather than
    aborting, since provenance is best-effort."""
    try:
        out = subprocess.run(
            ["sudo", "dmidecode", "-s", "system-serial-number"],
            text=True, capture_output=True, check=True, timeout=10).stdout.strip()
        if out:
            return out
        return "dmidecode: empty output"
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        return f"dmidecode failed: {e}"


def get_gpu_name():
    """
    Returns the GPU name via nvidia-smi, or None if unavailable.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=5)
        name = result.stdout.strip().split("\n")[0]
        return name if name else None
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def run_binary_with_cleanup(cmd):
    """Run binary command without toggling E-cores (they should stay disabled)"""
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out
    except subprocess.CalledProcessError as e:
        raise e
    except KeyboardInterrupt:
        # Handle Ctrl+C during subprocess execution
        print("\nKeyboard interrupt received during binary execution...")
        raise
