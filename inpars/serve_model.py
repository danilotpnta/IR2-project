import os
import sys
import signal
import shlex
import psutil
import getpass
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
    # TODO: check by calling this method from another script stops the server

    port = MODEL_PORT_MAPPING.get(model_name, 8000)
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = " ".join(proc.info["cmdline"])
            if (
                f"--port {port}" in cmdline
                and "vllm.entrypoints.openai.api_server" in cmdline
            ):
                print(f"Stopping server for '{model_name}' running on port {port}.")
                proc.terminate()
                proc.wait(timeout=5)  #
                print(f"Server for '{model_name}' has been stopped.")
                return
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
            continue
    print(f"No server for '{model_name}' is running on port {port}.")


def serve_model(
    model_name,
    use_scratch_shared_cache=True,
    gpu_memory_utilization=0.975,
    max_model_len=8192,
):
    """Start the model server only if it's not already running."""

    if use_scratch_shared_cache:

        # Save the Hugging Face model to scratch-shared directory
        hf_cache_dir = f"/scratch-shared/{getpass.getuser()}/.cache/huggingface"
        os.makedirs(hf_cache_dir, exist_ok=True)

        os.environ["HF_HOME"] = hf_cache_dir
        print(f"HF_HOME set to {hf_cache_dir}")

    port = MODEL_PORT_MAPPING.get(model_name, 8001)
    base_model = shlex.quote(model_name)

    if is_server_running(port):
        print(f"Model server for '{model_name}' is already running on port {port}.")
        return

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


if __name__ == "__main__":
    model_name = "neuralmagic/Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic"
    serve_model(model_name)
