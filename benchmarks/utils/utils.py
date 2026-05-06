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
    parser.add_argument("--input_mode", choices=["uniform", "trunk", "csv"], default="csv")
    parser.add_argument("--train_csv", default="benchmarks/data/processed_wise1_data.csv")
    parser.add_argument("--label_col", default="Cancer Status")
    parser.add_argument("--experiment_name", default="")
    parser.add_argument("--feature_split_type", default="Oblique",
                       choices=["Axis Aligned", "Oblique"])
    parser.add_argument("--numerical_split_type", default="Exact",
                       choices=["Exact", "Random", "Equal Width",
                                "Dynamic Random Histogram", "Dynamic Equal Width Histogram"])
    parser.add_argument("--vectorized", choices=[None, "avx2", "avx512"], default=None)
    parser.add_argument("--tree_depth", type=int, default=-1)
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--num_trees", type=int, default=1)  # Note: different defaults in your files
    parser.add_argument("--projection_density_factor", type=int)
    parser.add_argument("--max_num_projections", type=int)
    parser.add_argument("--sample_projection_mode", choices=["Fast", "Slow"], default="Fast")
    parser.add_argument("--fixed_1000_projections", action="store_true")
    parser.add_argument("--nodewise_proj_matrix", action="store_true",
                        help="Build with -DNODEWISE_PROJ_MATRIX=1: V1 fused "
                             "per-level CPU ApplyProjection (per-node rows-"
                             "outer / projections-inner matrix fill; serial "
                             "across nodes).")
    parser.add_argument("--depthwise_1_pass", action="store_true",
                        help="Build with -DDEPTHWISE_1_PASS=1: V2 fused "
                             "per-level CPU ApplyProjection (single-pass "
                             "kernel across all (row, projection) tasks at "
                             "the level; thread-parallel, contention-free).")
    # parser.add_argument("--enable_fast_equal_width_binning", action="store_true") # This is on by default now
    parser.add_argument("--use_gpu", type=lambda x: x.lower() in ("true", "1", "yes"),
                       default=False, help="Use GPU for oblique projections (default: false)")
    parser.add_argument("--bazel_config", action="append", default=[], metavar="NAME",
                       help="Extra --config=NAME to pass to the bazel build. Repeatable: "
                            "--bazel_config=with_isnan --bazel_config=use_std_sort. "
                            "Applied after the flag-driven configs (avx2, "
                            "nodewise_proj_matrix, depthwise_1_pass, ...).")

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
    cmd = ["sudo", "./benchmarks/utils/set_cpu_e_features.sh", action]

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
        finished_cmd.append('--config=multithreaded_chrono_profile')

    # --use_gpu=true requires the GPU code paths to be compiled in. Without
    # --config=oblique_gpu the OBLIQUE_GPU_ENABLED macro is undefined and
    # nodewise/depthwise-gpu dispatch is compiled out (binary defaults to CPU).
    if getattr(args, 'use_gpu', False):
        finished_cmd.append('--config=oblique_gpu')

    if getattr(args, 'nodewise_proj_matrix', False) and getattr(args, 'depthwise_1_pass', False):
        raise ValueError(
            "--nodewise_proj_matrix and --depthwise_1_pass are mutually "
            "exclusive (matches the C++ #error in label.h).")
    if getattr(args, 'nodewise_proj_matrix', False):
        finished_cmd.append('--config=nodewise_proj_matrix')
    if getattr(args, 'depthwise_1_pass', False):
        finished_cmd.append('--config=depthwise_1_pass')

    if getattr(args, 'gpu_mode', None) == 'per_node':
        finished_cmd.append('--config=dfs_node_queue')

    # if args.enable_fast_equal_width_binning:
    #     finished_cmd.append('--config=enable_fast_equal_width_binning')

    for extra_config in getattr(args, 'bazel_config', []) or []:
        finished_cmd.append(f'--config={extra_config}')

    finished_cmd.append("--ui_event_filters=-warning")
    finished_cmd.append('//examples:train_oblique_forest')

    global last_build_cmd
    last_build_cmd = " ".join(finished_cmd)

    print("Building binary...")
    print(f"Running: {' '.join(finished_cmd)}")
    
    try:
        result = subprocess.run(
            finished_cmd, 
            capture_output=False, 
            text=True, 
            check=True,
            env=os.environ.copy(),  # Preserve current environment
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
    Reads /proc/cpuinfo and returns the first 'model name' value.
    """
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name"):
                    # split only on the first ':' → [key, value]
                    return line.split(":", 1)[1].strip()
    except FileNotFoundError:
        return "Could not access /proc/cpuinfo to get CPU model name"


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
