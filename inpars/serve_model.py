import os
import sys
import signal
import shlex
import psutil 
import getpass
import subprocess

MODEL_PORT_MAPPING = {
    "EleutherAI/gpt-j-6B": 8000,
    "meta-llama/Llama-3.1-8B": 8001,
    "neuralmagic/Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic": 8002,
}

def is_server_running(port):
    """Check if a server is already running on the specified port."""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'])
            if f"--port {port}" in cmdline and "vllm.entrypoints.openai.api_server" in cmdline:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False

def stop_model(model_name):
        """Stop the model server running on the specified port."""
        # TODO: check by calling this method from another script stops the server
     
        port = MODEL_PORT_MAPPING.get(model_name, 8000)
        for proc in psutil.process_iter(['pid', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'])
                if f"--port {port}" in cmdline and "vllm.entrypoints.openai.api_server" in cmdline:
                    print(f"Stopping server for '{model_name}' running on port {port}.")
                    proc.terminate()  
                    proc.wait(timeout=5)  #
                    print(f"Server for '{model_name}' has been stopped.")
                    return
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                continue
        print(f"No server for '{model_name}' is running on port {port}.")

def serve_model(model_name, use_scratch_shared_cache=True):
    """Start the model server only if it's not already running."""

    if use_scratch_shared_cache:

        # Save the Hugging Face model to scratch-shared directory
        hf_cache_dir = f"/scratch-shared/{getpass.getuser()}/.cache/huggingface"
        os.makedirs(hf_cache_dir, exist_ok=True)

        os.environ["HF_HOME"] = hf_cache_dir
        print(f"HF_HOME set to {hf_cache_dir}")

    port = MODEL_PORT_MAPPING.get(model_name, 8001)  
    base_model = shlex.quote(model_name)

    print("base_model", base_model)

    if is_server_running(port):
        print(f"Model server for '{model_name}' is already running on port {port}.")
        return

    cmd = (
        f"python -m vllm.entrypoints.openai.api_server --model {base_model} --port {port}"
    )

    process = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
    print(f"Model server started with command: {cmd}")

    def terminate():
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)  
        sys.exit(0)

    # Handle Ctrl+C & Kill Signals
    signal.signal(signal.SIGINT, terminate)  
    signal.signal(signal.SIGTERM, terminate)  
    process.wait()


if __name__ == "__main__":
    model_name = "EleutherAI/gpt-j-6B" 
    serve_model(model_name)



