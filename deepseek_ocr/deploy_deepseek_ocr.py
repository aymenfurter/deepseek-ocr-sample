import os
import secrets
import subprocess
from dataclasses import dataclass

@dataclass
class DeploymentConfig:
    resource_group: str = os.getenv("DEEPSEEK_OCR_RESOURCE_GROUP", "gpu-deployments-rg")
    location: str = os.getenv("DEEPSEEK_OCR_LOCATION", "swedencentral")
    env_name: str = os.getenv("DEEPSEEK_OCR_ENV_NAME", "deepseek-ocr-env")
    app_name: str = os.getenv("DEEPSEEK_OCR_APP_NAME", "deepseek-ocr-app")
    ingress_name: str = os.getenv("DEEPSEEK_OCR_INGRESS_NAME", "deepseek-ocr-gateway")
    model_name: str = os.getenv("DEEPSEEK_OCR_MODEL", "deepseek-ai/DeepSeek-OCR")
    image_name: str = os.getenv("DEEPSEEK_OCR_IMAGE", "vllm/vllm-openai:latest")
    workload_profile: str = os.getenv("DEEPSEEK_OCR_WORKLOAD_PROFILE", "gpu-a100")
    workload_type: str = os.getenv("DEEPSEEK_OCR_WORKLOAD_TYPE", "Consumption-GPU-NC24-A100")
    cpu: float = float(os.getenv("DEEPSEEK_OCR_CPU", "12"))
    memory: str = os.getenv("DEEPSEEK_OCR_MEMORY", "32Gi")
    huggingface_token: str = os.getenv("HUGGING_FACE_HUB_TOKEN", "")

def run_cli(command: str, description: str, capture_output: bool = False) -> subprocess.CompletedProcess:
    print(f"Running: {description}")
    result = subprocess.run(command, shell=True, text=True, capture_output=capture_output)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)
    return result

def setup_infrastructure(cfg: DeploymentConfig):
    run_cli("az extension add --name containerapp --upgrade --yes", "Installing containerapp extension")
    run_cli(f"az group create -n {cfg.resource_group} -l {cfg.location}", "Creating Resource Group")
    
    # Create Environment
    try:
        run_cli(f"az containerapp env show -n {cfg.env_name} -g {cfg.resource_group}", "Checking Env", capture_output=True)
    except:
        run_cli(f"az containerapp env create -n {cfg.env_name} -g {cfg.resource_group} -l {cfg.location} --enable-workload-profiles", "Creating Env")

    # Add Workload Profile
    try:
        run_cli(f"az containerapp env workload-profile add -n {cfg.env_name} -g {cfg.resource_group} --workload-profile-name {cfg.workload_profile} --workload-profile-type {cfg.workload_type}", "Adding Workload Profile")
    except:
        pass # Likely exists

def deploy_deepseek_backend(cfg: DeploymentConfig) -> str:
    # Internal, Scale to 0
    yaml_config = f"""properties:
  workloadProfileName: {cfg.workload_profile}
  configuration:
    activeRevisionsMode: Single
    ingress:
      external: false
      targetPort: 8000
      transport: Auto
  template:
    containers:
    - name: vllm-inference
      image: {cfg.image_name}
      resources:
        cpu: {cfg.cpu}
        memory: {cfg.memory}
      env:
      - name: HUGGING_FACE_HUB_TOKEN
        value: "{cfg.huggingface_token}"
      command: ["python3", "-m", "vllm.entrypoints.openai.api_server"]
      args: ["--model", "{cfg.model_name}", "--trust-remote-code", "--gpu-memory-utilization", "0.9", "--max-model-len", "8192", "--host", "0.0.0.0", "--port", "8000"]
      probes:
      - type: Startup
        httpGet: {{ path: /health, port: 8000 }}
        initialDelaySeconds: 60
        periodSeconds: 10
        failureThreshold: 30
    scale:
      minReplicas: 0
      maxReplicas: 1
"""
    with open("aca-deepseek-backend.yaml", "w") as f:
        f.write(yaml_config)
    
    try:
        run_cli(f"az containerapp show -n {cfg.app_name} -g {cfg.resource_group}", "Checking Backend", capture_output=True)
        run_cli(f"az containerapp update -n {cfg.app_name} -g {cfg.resource_group} --yaml aca-deepseek-backend.yaml", "Updating DeepSeek Backend")
    except:
        run_cli(f"az containerapp create -n {cfg.app_name} -g {cfg.resource_group} --environment {cfg.env_name} --yaml aca-deepseek-backend.yaml", "Creating DeepSeek Backend")
    
    # Get internal FQDN
    res = run_cli(f"az containerapp show -n {cfg.app_name} -g {cfg.resource_group} --query properties.configuration.ingress.fqdn -o tsv", "Getting Backend FQDN", capture_output=True)
    return f"https://{res.stdout.strip()}"

def deploy_ingress_gateway(cfg: DeploymentConfig, backend_url: str, api_key: str) -> str:
    # Public, CPU, Autoscaled
    print("Deploying Ingress Gateway...")
    
    acr_name = (cfg.app_name.replace("-", "") + "acr")[:50] # Ensure valid name length
    
    # Ensure ACR
    try:
        run_cli(f"az acr show -n {acr_name} -g {cfg.resource_group}", "Checking ACR", capture_output=True)
    except:
        run_cli(f"az acr create -n {acr_name} -g {cfg.resource_group} --sku Basic --admin-enabled true", "Creating ACR")
        
    # Build Image
    ingress_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ingress")
    run_cli(f"az acr build -r {acr_name} -t deepseek-ingress:latest {ingress_path}", "Building Ingress Image")
    
    # Get Credentials
    username = run_cli(f"az acr credential show -n {acr_name} --query username -o tsv", "Fetching ACR User", capture_output=True).stdout.strip()
    password = run_cli(f"az acr credential show -n {acr_name} --query passwords[0].value -o tsv", "Fetching ACR Password", capture_output=True).stdout.strip()
    
    # Deploy
    cmd = (
        f"az containerapp create -n {cfg.ingress_name} -g {cfg.resource_group} "
        f"--environment {cfg.env_name} "
        f"--image {acr_name}.azurecr.io/deepseek-ingress:latest "
        f"--registry-server {acr_name}.azurecr.io "
        f"--registry-username {username} "
        f"--registry-password {password} "
        f"--ingress external --target-port 8080 "
        f"--env-vars TARGET_URL={backend_url} API_KEY={api_key} "
        f"--min-replicas 1 --max-replicas 10 "
        f"--cpu 0.5 --memory 1.0Gi"
    )
    
    # Handle update vs create
    try:
        run_cli(f"az containerapp show -n {cfg.ingress_name} -g {cfg.resource_group}", "Checking Ingress App", capture_output=True)
        # Update
        run_cli(f"az containerapp update -n {cfg.ingress_name} -g {cfg.resource_group} --image {acr_name}.azurecr.io/deepseek-ingress:latest", "Updating Ingress App")
        # Update env vars and scale separately or in one go? Update supports env vars.
        # But simpler to just run the create command which updates if exists? 
        # `az containerapp create` might fail if exists.
        # Let's use `update` for everything if it exists.
        update_cmd = (
            f"az containerapp update -n {cfg.ingress_name} -g {cfg.resource_group} "
            f"--image {acr_name}.azurecr.io/deepseek-ingress:latest "
            f"--set-env-vars TARGET_URL={backend_url} API_KEY={api_key} "
            f"--min-replicas 1 --max-replicas 10"
        )
        run_cli(update_cmd, "Updating Ingress Configuration")
    except:
        run_cli(cmd, "Creating Ingress App")
    
    res = run_cli(f"az containerapp show -n {cfg.ingress_name} -g {cfg.resource_group} --query properties.configuration.ingress.fqdn -o tsv", "Getting Ingress FQDN", capture_output=True)
    return f"https://{res.stdout.strip()}"

if __name__ == "__main__":
    cfg = DeploymentConfig()
    api_key = secrets.token_urlsafe(32)
    
    setup_infrastructure(cfg)
    backend_url = deploy_deepseek_backend(cfg)
    print(f"Backend deployed at: {backend_url}")
    
    ingress_url = deploy_ingress_gateway(cfg, backend_url, api_key)
    print(f"Ingress deployed at: {ingress_url}")
    
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    with open(env_path, "a") as f:
        f.write(f"\n# DeepSeek OCR Configuration\n")
        f.write(f"DEEPSEEK_OCR_ENDPOINT={ingress_url}\n")
        f.write(f"DEEPSEEK_OCR_API_KEY={api_key}\n")
        f.write(f"DEEPSEEK_OCR_BACKEND_URL={backend_url}\n")
    
    print("\nDeployment Complete!")
    print(f"API Key: {api_key}")
    print(f"Endpoint: {ingress_url}")
    print(f"Credentials appended to {env_path}")