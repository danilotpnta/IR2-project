import os
import sys
import signal
import shlex
import psutil
import getpass
import argparse
import subprocess

from config import MODEL_PORT_MAPPING


def is_server_running(port):
    """Check if a server is already running on the specified port."""
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = proc.info["cmdline"]
            if (
                cmdline
                and f"--port {port}" in " ".join(cmdline)
                and "vllm.entrypoints.openai.api_server" in " ".join(cmdline)
            ):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False


def stop_model(model_name):
    """Stop the model server running on the specified port."""

    port = MODEL_PORT_MAPPING.get(model_name, 8000)
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = proc.info["cmdline"]
            if cmdline and isinstance(cmdline, list):
                cmdline_str = " ".join(cmdline)
                if (
                    f"--port {port}" in cmdline_str
                    and "vllm.entrypoints.openai.api_server" in cmdline_str
                ):
                    print(
                        f"> Stopping server for '{model_name}' running on port {port}."
                    )
                    proc.terminate()
                    proc.wait(timeout=5)  #
                    print(f"> Server for '{model_name}' has been stopped.")
                    return
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
            continue
    print(f"> No server for '{model_name}' is running on port {port}.")


def serve_model(
    model_name,
    use_scratch_shared_cache=True,
    gpu_memory_utilization=0.975,
    max_model_len=8192,
    script_path=os.path.abspath(__file__),
):
    """Start the model server only if it's not already running."""

    for model, port in MODEL_PORT_MAPPING.items():
        if is_server_running(port):
            print(f"> Model server for '{model}' is running on port {port}.")
            print(
                f"> To stop the server, run: \n" 
                f"  $ python {script_path} --stop_model '{model}'"
                )
            return

    if use_scratch_shared_cache:

        # Save the Hugging Face model to scratch-shared directory
        hf_cache_dir = f"/scratch-shared/{getpass.getuser()}/.cache/huggingface"
        os.makedirs(hf_cache_dir, exist_ok=True)

        os.environ["HF_HOME"] = hf_cache_dir
        print(f"> HF_HOME set to {hf_cache_dir}")

    port = MODEL_PORT_MAPPING.get(model_name, 8001)
    base_model = shlex.quote(model_name)

    if model_name == "EleutherAI/gpt-j-6B":
        max_model_len = 2048

    cmd = (
        f"python -m vllm.entrypoints.openai.api_server "
        f"--model {base_model} "
        f"--port {port} "
        f"--gpu-memory-utilization {gpu_memory_utilization} "
        f"--max-model-len {max_model_len} "
    )

    process = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)

    def terminate(signum, frame):
        """Terminate the server process and exit."""
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.terminate()
        except Exception as e:
            print(f"Error during termination: {e}")
        finally:
            sys.exit(0)

    signal.signal(signal.SIGINT, terminate)
    signal.signal(signal.SIGTERM, terminate)

    try:
        print(f"Model server started with command: {cmd}")
        process.wait()
    except KeyboardInterrupt:
        terminate(None, None)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stop_model",
        help="Stop the specified model server by name.",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--model_name",
        help="Start the specified model server by name.",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        choices=[
            "EleutherAI/gpt-j-6B",
            "meta-llama/Llama-3.1-8B",
            "neuralmagic/Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic",
        ],
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    if args.stop_model:
        stop_model(args.stop_model)
    else:
        serve_model(args.model_name)
