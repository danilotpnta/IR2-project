import shlex
import subprocess
import psutil 

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

def serve_model(model_name):
    """Start the model server only if it's not already running."""
    port = MODEL_PORT_MAPPING.get(model_name, 8000)  
    base_model = shlex.quote(model_name)

    if is_server_running(port):
        print(f"Model server for '{model_name}' is already running on port {port}.")
        return

    cmd = (
        f"python -m vllm.entrypoints.openai.api_server --model {base_model} --port {port}"
    )
    subprocess.Popen(cmd, shell=True)
    print(f"Model server started with command: {cmd}")


if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-8B" 
    serve_model(model_name)



