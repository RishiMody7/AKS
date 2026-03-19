---
title: "Deploy Triton Inference Server with a vLLM Backend on AKS enabled by Azure Arc"
date: 2026-03-18
description: "Step-by-step guide to deploying Triton Inference Server with a vLLM backend on an AKS Arc cluster for vision-language model inferencing"
authors:
- datta-rajpure
- rishi-mody
tags: ["aks-arc", "ai", "performance", "open-source", "triton", "vLLM"]
---

:::note
This tutorial applies to Azure Kubernetes Service (AKS) on Azure Local (Azure Stack HCI), with at least one GPU node (e.g., NVIDIA T4, A16, or A100 GPU) and the NVIDIA GPU drivers installed.
:::


This article describes how to deploy NVIDIA Triton Inference Server with the vLLM backend on an AKS on Azure Local cluster. This configuration uses a **vision-language large language model (LLM)**, specifically the **Qwen2-VL-2B-Instruct model**, to perform AI inferencing tasks. For example, you can provide an image as input and receive a text description in response. 

<!-- truncate -->

**vLLM** is an open-source LLM inference engine that integrates with Triton as a Python backend. It employs a memory management technique called PagedAttention to handle the model’s key-value cache in GPU memory without fragmentation. This approach, combined with vLLM’s support for **continuous batching**, allows new inference requests to enter the pipeline even while previous requests are mid-execution, maximizing GPU utilization.

In a vision-language scenario, an input image can expand to thousands of tokens (often 1,000–4,000 tokens per image), and a 16 GB GPU provides a sufficiently large context window (up to 32k tokens or more) for high-resolution image analysis. The 2-billion-parameter Qwen2-VL-2B-Instruct model is well suited for this setup: it occupies only ~4.5 GB of GPU memory (BF16 precision), leaving ~10 GB free for the vLLM PagedAttention cache, and it runs about 3–5× faster than a 7B model on the same hardware.

Triton's vLLM backend also supports a decoupled (non-blocking) transaction mode to stream out partial results as the model generates tokens, enabling responsive real-time inferencing. 

**NVIDIA Triton Inference Server** is a high-performance, open-source model serving platform that provides an easy way to host AI models and handle inference requests via HTTP/REST or gRPC APIs. In this article, you will deploy Triton on your AKS Arc cluster and use it to serve the Qwen2-VL-2B-Instruct model for image-based LLM inference.

## Workflow
To deploy Triton Inference Server with the vLLM backend on an AKS on Azure Local cluster, follow this high-level workflow: 

**Create a namespace and persistent storage**: Set up a Kubernetes namespace and a PersistentVolumeClaim (PVC) to hold the model files and configurations.  
**Deploy a provisioner pod**: Launch a lightweight helper pod (with access to the GPU node’s filesystem) for preparing the model repository (configuration files and model pointers).  
**Prepare the model repository**: Inside the provisioner pod, create the required directory structure and configuration files (such as model.json and config.pbtxt) for the vLLM model. This example uses the Hugging Face model *Qwen2-VL-2B-Instruct*; the vLLM backend automatically downloads the model weights at runtime if they aren't already present on the PVC.  
**Deploy Triton Inference Server**: Create a Kubernetes Deployment and Service to run the Triton server with the vLLM backend, mounting the model repository PVC.  
**Test AI inferencing**: Send test requests (for example, an image with a prompt) to the Triton server’s */v2/models/.../generate* endpoint to verify that the model is loaded and producing responses.  
**Clean up resources**: Delete the namespace to remove all deployed resources from the cluster when done testing.  
**Troubleshoot**: If you encounter issues, use the provided troubleshooting steps (for example, checking pod status, logs, and configuration). 


## Prerequisites 

Before you begin, make sure you have the following: 

**AKS on Azure Arc cluster with a GPU node**: You need an AKS cluster enabled by Azure Arc on Azure Local (Azure Stack HCI) with at least one GPU-enabled node (for example, NVIDIA T4, A16, or similar GPU). Verify that you've installed the appropriate NVIDIA drivers and the NVIDIA device plugin or GPU Operator so pods can access the GPU. If you need to add a GPU node pool to your Azure Local cluster, see the [Microsoft documentation](https://learn.microsoft.com/azure/aks/aksarc/deploy-gpu-node-pool) for guidance.

**Azure CLI with Azure Arc extensions**: Install the Azure CLI and the aks-preview extension (for Azure Arc previews) along with either the aksarc or connectedk8s extension for Azure Arc-enabled Kubernetes management. Run `az extension list -o table` to verify installation.

**kubectl**: Install the Kubernetes CLI kubectl on your workstation. Use kubectl to apply Kubernetes manifests and manage cluster resources. 

**Helm**: Install Helm on your machine. You might need it to install the NVIDIA GPU Operator or drivers. 

**PowerShell 7+ (optional)**: If you plan to use PowerShell for sending test requests (as shown in this article), use PowerShell 7.4 or later. (Older versions like Windows PowerShell 5.1 may have issues with JSON formatting in web requests). 

**Cluster access**: Verify that you have network connectivity to your AKS Arc cluster. This might require being on the same network as the cluster or using a VPN/ExpressRoute connection to your Azure Local environment.

**(Optional) Windows 11 environment setup**: On a Windows 11 machine, use winget to install or update the required tools (PowerShell, Azure CLI, kubectl, and Helm) as follows: 
```powershell
# Install PowerShell 

winget install -e --id Microsoft.PowerShell 
pwsh -v 

# Install or Update - Azure CLI, Kubectl, Helm, Git 
winget install -e --id Microsoft.AzureCLI 
winget install -e --id Kubernetes.kubectl 
winget install -e --id Helm.Helm 
winget install -e --id Git.Git 

winget update -e --id Microsoft.AzureCLI 
winget update -e --id Kubernetes.kubectl 
winget update -e --id Helm.Helm 
winget update -e --id Git.Git 

# Install or Update – Azure CLI Extensions (AKS Arc) 
az extension add --name aksarc 
az extension add --name connectedk8s 
az extension update --name aksarc 
az extension update --name connectedk8s 
```
:::note
The -e flag ensures exact matching by ID. Omit the -e and specify version numbers if you need specific versions.
:::

## Step 1: Create a namespace and Persistent Volume Claim (PVC)
First, create a dedicated Kubernetes namespace for Triton and a **PersistentVolumeClaim** (PVC) for the model repository. The namespace logically isolates your Triton resources (server, provisioner pod, and so on), and the PVC provides persistent storage for model files and configuration. 

### Apply the namespace and PVC manifest
Run `kubectl apply` to create the namespace and PVC from a manifest file (for example, triton-pvc.yaml): 

```bash
 # Create the namespace and PVC 
kubectl apply -f triton-pvc.yaml 

# Verify the PVC status (should show 'Bound' when ready) 
kubectl get pvc -n triton-inference 
```

### Create triton-pvc.yaml 

Create a file named **triton-pvc.yaml** with the following content, which defines a new namespace (triton-inference) and a 100 Gi persistent volume claim named triton-model-repository-pvc: 

```yaml
# 1. THE NAMESPACE 
# Creates an isolated logical boundary for your Triton resources. 
# All subsequent resources must reference this namespace to communicate. 
apiVersion: v1 
kind: Namespace 
metadata: 
  name: triton-inference 
--- 

# 2. THE STORAGE (PVC) 
# Requests a persistent disk from the cluster to store your model weights. 
# This ensures that if the Pod restarts, your downloaded models remain intact. 
apiVersion: v1 
kind: PersistentVolumeClaim 
metadata: 
  name: triton-model-repository-pvc 
  namespace: triton-inference # Ensures the storage is available within your namespace 
spec: 
  # ReadWriteOnce (RWO) allows the volume to be mounted as read-write by a single node. 
  accessModes: 
    - ReadWriteOnce 
  # The StorageClass must exist in your cluster.  
  storageClassName: default  
  resources: 
    requests: 
      # Allocates 100GB of space. Ensure your underlying disk provider  
      # supports this size. 
      storage: 100Gi 
```

After applying this manifest, wait until the PVC's status is **Bound** (indicating the cluster successfully provisioned storage) before proceeding.


## Step 2: Create a Triton provisioner pod 

Next, deploy a provisioner pod. Use this temporary helper pod to set up the model repository on the PVC (for example, creating configuration files and organizing model data). Since the vLLM backend can download and serve the model weights at runtime, the provisioner's main role is to prepare configuration files (and optionally pre-download models if your cluster is in an offline environment). The provisioner mounts the same PVC, allowing you to manipulate files in the persistent storage.

### Deploy the provisioner pod 

Apply the provisioner pod manifest and wait for the pod to be running: 
```bash
# Create the provisioner pod 
kubectl apply -f triton-provisioner.yaml 

# Wait for the provisioner pod to be up and running 
kubectl get pods -n triton-inference -w 

# Optional: Describe the pod to confirm it started without issues 
kubectl describe pod triton-provisioner -n triton-inference 
```
### Create triton-provisioner.yaml 

Create a file named **triton-provisioner.yaml** with the following content. This pod spec uses a minimal container image (busybox) that will run indefinitely (until you shut it down) and mount the model repository PVC at /models inside the container: 

```yaml
# triton-provisioner.yaml 
# This pod is a helper tool used to "provision" (upload/edit) model files  
# into a Persistent Volume used by Triton. 

apiVersion: v1 
kind: Pod 
metadata: 
  name: model-provisioner 
  namespace: triton-inference  # Must match the namespace of your Triton deployment 
spec: 
  # 'Never' ensures the pod doesn't restart once you manually stop or delete it 
  restartPolicy: Never  
  containers: 
  - name: writer 
    image: busybox             # Lightweight Linux image perfect for file operations 
    # 'sleep infinity' keeps the pod running so you can 'kubectl exec' into it  
    # to move or edit your Qwen2-VL model files. 
    command: ["sh","-c","sleep infinity"]  
    volumeMounts: 
    - name: model-repo 
      mountPath: /models       # The PV will be accessible at this folder inside the pod 
  volumes: 
  - name: model-repo 
    # This must match the name of the PVC you created for your Triton server 
    persistentVolumeClaim: 
      claimName: triton-model-repository-pvc 
```

Once this pod is running, you exec into it to perform the next steps. This pod doesn't request a GPU because it handles only file operations. The Triton server pod handles the actual model inference. 

## Step 3: Prepare the model repository and configuration 

In this step, you create the necessary folder structure and configuration files that tell Triton how to load and serve the model with the vLLM backend. The example uses the Hugging Face model Qwen2-VL-2B-Instruct, which is a vision-language LLM. The vLLM backend supports direct downloading of models from Hugging Face if the model files aren't already present on disk, so you don't need to manually download the model weights unless your cluster has restricted internet connectivity.

:::note
If your cluster cannot access the internet (or you prefer to manage model files manually), you should pre-download the model to the PVC before deploying Triton. You can do this by using the Hugging Face CLI inside the provisioner pod or by adding an init container to the provisioner that performs the download using huggingface-cli. In online scenarios, however, you can rely on vLLM's lazy download feature.
:::
 

### Create the model directory and model configuration 

Use kubectl exec to open a shell inside the provisioner pod: 
```bash
kubectl exec -it triton-provisioner -n triton-inference -- /bin/sh 
```
Once inside the pod's shell, run the following commands inside the provisioner container:

1. Create the model repository directory structure. In Triton, each model has its own directory and a subdirectory for versioned model data. For this model, create a folder named qwen2_vl_2b under /models/repo (the mounted PVC), and a subfolder 1 inside it (to represent version 1 of the model):
```bash
mkdir -p /models/repo/qwen2_vl_2b/1 
```

2. Create the model.json file. This JSON configuration file specifies the model settings for the vLLM backend. In particular, it identifies the Hugging Face model name and several runtime parameters (like how much GPU memory to use for the model, maximum sequence lengths, and so on). Create a file `/models/repo/qwen2_vl_2b/1/model.json` with the following content:

```bash
cat <<'EOF' > /models/repo/qwen2_vl_2b/1/model.json 

{
    "model": "Qwen/Qwen2-VL-2B-Instruct", 
    "gpu_memory_utilization": 0.6, 
    "max_model_len": 32768, 
    "trust_remote_code": true, 
    "dtype": "half", 
    "enforce_eager": true, 
    "limit_mm_per_prompt": {"image": 1}, 
    "max_num_seqs": 1 
} 

EOF 

ls -l /models/repo/qwen2_vl_2b/1/model.json 

cat /models/repo/qwen2_vl_2b/1/model.json 

```

 

In this configuration: 

* "model" is the Hugging Face model identifier. When Triton starts, the vLLM backend uses this name to **automatically download** the model from Hugging Face if it's not already present on the PVC.
* "gpu_memory_utilization": 0.6 reserves roughly 60% of the GPU’s memory for the model’s execution, leaving the rest for Triton overhead and other processes. 
* "max_model_len": 32768 sets a large maximum sequence length (tokens) for the model’s context window, supporting high-resolution or long input content (for example, large images with many tokens after visual processing). 
* "enforce_eager": true disables the use of CUDA Graphs in vLLM, which saves ~1–2 GB of VRAM – a safer setting for complex, high-resolution tasks. 
* Other parameters (like trust_remote_code, dtype, and so on) are set as recommended for this model and enable features such as half-precision (FP16) inference. 

3. Create the Triton config.pbtxt file. This protobuf text file defines the model's interface (inputs and outputs) and how Triton should handle execution. Create a new file at `/models/repo/qwen2_vl_2b/config.pbtxt` with the following content:
```bash
cat <<'EOF' > /models/repo/qwen2_vl_2b/config.pbtxt 

name: "qwen2_vl_2b" 
backend: "vllm" 
max_batch_size: 0 
model_transaction_policy { 
  decoupled: true 
} 
input [ 
  { 
    name: "text_input" 
    data_type: TYPE_STRING 
    dims: [ 1 ] 
  } 
] 
output [ 
  { 
    name: "text_output" 
    data_type: TYPE_STRING 
    dims: [ -1 ] 
  } 
] 
instance_group [ 
  { 
    count: 1 
    kind: KIND_GPU 
  } 
] 

EOF 
ls -l /models/repo/qwen2_vl_2b 
cat /models/repo/qwen2_vl_2b/config.pbtxt 
```
 

Key points in this config: 

* It sets the model name to qwen2_vl_2b (which should match the directory name). 
* It specifies the vLLM backend and uses max_batch_size: 0 (which in Triton means “no fixed batch limit,” allowing dynamic batching). 
* model_transaction_policy `{ decoupled: true }` enables Triton’s decoupled execution mode, which is required for the streaming generate endpoint. This allows the model to stream output tokens without waiting for the entire sequence to complete. 
* Inputs/Outputs: The model expects one input named "text_input" (a string containing the prompt and, in this case, an image placeholder) and produces one output string named "text_output". The input’s dims: [1] and output’s dims: [-1] indicate a single input string and a variable-length output sequence, respectively. 
* The instance_group specifies the model should run on one GPU (count: 1, kind: KIND_GPU). Ensure your cluster’s GPU node has sufficient memory for the model and its cache (for Qwen2-VL-2B, a single 16 GB GPU is suitable). 

After creating these files (model.json and config.pbtxt), the model repository on the PVC should look like this (with qwen2_vl_2b as the model name and 1 as the version directory): 
```
1     /models/repo/ 
2     └── qwen2_vl_2b/ 
3         ├── config.pbtxt 
4         └── 1/ 
5             └── model.json 
```
You can now exit the provisioner pod’s shell (type exit). The model repository is ready for Triton. 

## Step 4: Deploy the Triton Inference Server 

Now deploy the Triton server on your AKS Arc cluster to load the model. Create a Kubernetes Deployment for the Triton Inference Server and a Service to expose its API endpoints. The Deployment uses NVIDIA's Triton container image with the vLLM backend and mounts the same PVC to load the model repository.

### Deploy the Triton server 

Apply the Triton Deployment and Service manifest: 
```bash
 kubectl apply -f triton-server.yaml 

 # Watch the Triton pod startup 
 kubectl get pods -n triton-inference -w 

 # (Optional) Check Triton logs for any errors during startup 
 kubectl logs -f deployment/triton-server -n triton-inference 
```
### Create a triton-server.yaml 

Create a file named **triton-server.yaml** with the following content, which defines both a Service and a Deployment for the Triton Inference Server: 
```yaml
# THE SERVICE (The "Phone Number" for your Model) 
# This exposes the Triton Inference Server to users outside or inside the cluster. 

apiVersion: v1 
kind: Namespace 
apiVersion: v1 
kind: Service 
metadata: 
  name: triton-server 
  namespace: triton-inference 
spec: 
  # LoadBalancer provides a public IP (on cloud providers like AKS/EKS/GKE). 
  # Change to 'ClusterIP' if you only want internal access. 
  type: LoadBalancer  
  ports: 
    # HTTP Endpoint: Standard REST queries (standard for most web apps) 
    - name: http 
      port: 8000 
      targetPort: 8000 
  selector: 
    # Connects this Service to any Pod labeled 'app: triton-server' 
    app: triton-server 
--- 

# THE DEPLOYMENT (The Inference Server) 
# This is the actual "Running Process" that serves the model to users. 
apiVersion: apps/v1 
kind: Deployment 
metadata: 
  name: triton-server 
  namespace: triton-inference 
spec: 
  # This can be flipped 1 or 0 to start and stop the Triton server during troubleshooting. 
  replicas: 1  
  selector: 
    matchLabels: 
      app: triton-server 
  template: 
    metadata: 
      labels: 
        app: triton-server 
    spec: 
      containers: 
        - name: triton-server 
          # Must match the version used in your Provisioner to ensure Engine compatibility. 
          image: nvcr.io/nvidia/tritonserver:25.02-vllm-python-py3 
          args: 
            - "tritonserver" 
            - "--model-repository=/models/repo" 
            # Verbose logging (Level 3) is great for debugging but very "noisy."  
            # In production, change --log-verbose to 0 or 1. 
            - "--log-info=true" 
            - "--log-warning=true" 
            - "--log-error=true" 
            - "--log-verbose=3" 
          ports: 
            - containerPort: 8000 
          resources: 
            limits: 
              # Triton requires at least one GPU to load the TensorRT-LLM backend. 
              nvidia.com/gpu: 1 
          volumeMounts: 
            # Mount the SAME PVC that the Provisioner used to access the built engine files. 
            - name: model-repo 
              mountPath: /models 
      volumes: 
        - name: model-repo 
          persistentVolumeClaim: 
            claimName: triton-model-repository-pvc
```
 

This manifest will start the Triton server and expose the following endpoint: 

* HTTP (port 8000) for REST API requests (for example, health checks and inference via HTTP POST).

The Triton container is configured to **load the model repository from the PVC (mounted at /models)** and run the vLLM backend. The `nvidia.com/gpu: 1` resource limit ensures the pod is scheduled on a GPU node and given one GPU for inferencing. Once you apply this manifest, Kubernetes pulls the NVIDIA Triton image and launches the server pod. It may take a few minutes for the Triton pod to reach a Running state, as the vLLM backend downloads the model from Hugging Face on startup (if not already present) and then loads it into memory. Monitor the pod status and logs during this process.

## Step 5: Test the Triton inference service 

After the Triton server is running, validate the deployment by querying the model. The Triton Service is of type LoadBalancer; on Azure Local, you may need to access it via the host network or use port forwarding if an external LoadBalancer IP isn't available. The following steps assume you use a local port forward to reach the Triton server.

### Connect to the Triton endpoint 

Use kubectl port-forward to route a local port to the Triton service’s HTTP port: 
```bash
kubectl port-forward -n triton-inference deploy/triton-server 8000:8000 
```
This will allow you to access the Triton server at http://localhost:8000 from your machine. 

### Verify server health and model readiness 

Use a tool like curl or PowerShell to check the Triton server's health endpoints and verify the model is loaded. For example, with PowerShell:

```powershell
$baseUri = "http://localhost:8000/v2" 
# 1. Is the server process alive? (Returns 200)
Invoke-RestMethod "$baseUri/health/live" 

# 2. Are all models loaded and ready? (Returns 200) 
# Note: This might take 3-5 mins for Qwen2-VL to finish loading 
Invoke-RestMethod "$baseUri/health/ready" 

# 3. List loaded models to confirm 'qwen2_vl_2b' is version 1 
Invoke-RestMethod "$baseUri/repository/index" -Method Post 
```
 

The health checks should return HTTP 200 status and a simple "OK" response if the server is running. Note that the ready endpoint may take a few minutes to respond with OK on first launch, because the vLLM backend is downloading and initializing the model in the background. The repository index call should return a list of loaded models; verify that your model qwen2_vl_2b appears (with version 1) in the list. 

### Send an inference request 

Now send a test inference request to the Triton server’s generate API, which is the endpoint used for text-generation models. The Qwen2-VL-2B-Instruct model accepts an image (as a base64-encoded string) along with a textual prompt, and returns a text-based description or answer. The example below shows how to send an image-based inference request using PowerShell (you can also use the programming language or REST tool of your choice): 

 
```powershell
# 1. Server Configuration 
$tritonUrl = "http://localhost:8000/v2/models/qwen2_vl_2b/generate" 

# 2. Ask User for Image Path 
Write-Host "`n--- Triton Vision Inference Tool ---" -ForegroundColor Cyan 
$imagePath = Read-Host "Please enter the full path to your image (for example, D:\images\cat.jpg)"
# Remove quotes if the user copy-pasted the path with them 
$imagePath = $imagePath.Replace('"', '').Trim() 
if (-not (Test-Path $imagePath)) { 
    Write-Error "File not found at: $imagePath" 
    exit 
} 

# 3. Read and Encode Image 
Write-Host "Encoding image..." -ForegroundColor Gray 
$imageBytes = [System.IO.File]::ReadAllBytes($imagePath) 
$base64Image = [Convert]::ToBase64String($imageBytes) 

# 4. Construct Payload
$promptHeader = "<|im_start|>user`n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>`n<|im_start|>assistant`n"
$body = @{ 
    text_input = $promptHeader 
    image      = $base64Image 
    parameters = @{ 
        max_tokens = 200 
        temperature = 0 
    } 
} | ConvertTo-Json -Depth 20 

# 5. Execute Request 

try { 
    Write-Host "Sending to Triton (vLLM Qwen2-VL-2B)..." -ForegroundColor Cyan 
    $response = Invoke-RestMethod -Uri $tritonUrl -Method Post -Body $body -ContentType "application/json" -TimeoutSec 120 

# 6. Clean and Trim Output 
    $fullText = $response.text_output 
    $marker = "<|im_start|>assistant`n" 
    if ($fullText -like "*$marker*") { 
        $cleanOutput = $fullText.Substring($fullText.IndexOf($marker) + $marker.Length) 
        $cleanOutput = $cleanOutput.Replace("<|im_end|>", "").Trim() 
        Write-Host "`nSuccess! Analysis:" -ForegroundColor Green 
        Write-Host "-------------------------------" 
        Write-Host $cleanOutput -ForegroundColor White 
        Write-Host "-------------------------------`n" 
    } else { 
        Write-Host "`nRaw Response:" -ForegroundColor Yellow 
        Write-Host $fullText 
    } 
} catch { 
    $streamReader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream()) 
    $errorMsg = $streamReader.ReadToEnd() 
    Write-Error "Triton Error: $errorMsg" 
} 
```
{/* TODO: Add photos of outputs */}
This script: 
* ✅ Verifies that the Triton server is live and ready using health check endpoints.  
* 📷 Prompts the user to input the path to a local image file.  
* 🔐 Reads the image and encodes it in Base64 format for transmission.  
* 🧠 Constructs a prompt using the Qwen2-VL-2B-Instruct model's expected input format, including special tokens for image embedding and prompt structure.  
* 📡 Sends the request to the Triton server's /generate endpoint using HTTP POST.  
* 🧾 Parses the model's response and extracts the assistant's generated output, displaying it in a clean format. 


If the inferencing service is working, you should receive a textual description of the image from the model. For example, given an image of a cat, the model might respond with a sentence describing the cat and its surroundings. 

## Step 6: Clean up resources 

After you have finished testing, clean up the Kubernetes resources to avoid incurring unnecessary usage of the cluster’s GPU and storage. The simplest way to remove all deployed objects is to delete the entire namespace: 

```bash
kubectl delete namespace triton-inference
```

This will delete the Triton Inference namespace and all resources within it (including the Triton server Deployment, Service, provisioner pod, and PVC). 

 

## Troubleshooting 

If you encounter issues during deployment or inferencing, use the following tips to diagnose and resolve common problems: 

### 1. Triton or provisioner pod is stuck in Pending state 

**Symptom**: The triton-provisioner pod or the triton-server pod stays in the Pending phase and does not start.

**Possible causes**: 
* No GPU node is available in the cluster, or the GPU resource is already fully allocated to other pods. 
* The persistent volume claim did not bind (storage is unavailable). 

**Solutions**: 
* Verify that your cluster has a node with the required GPU and that it’s not already running another GPU pod. You can run kubectl describe nodes | grep nvidia.com/gpu to see the GPU resource status on each node. If no GPUs are reported, check that your GPU nodes are correctly labeled and have the NVIDIA device plugin running. 

* Check the PVC status by running kubectl get pvc -n triton-inference. Ensure the PVC status is Bound to a volume. If not, you may need to adjust your storage class or provider to supply the requested 100 Gi volume. 

### 2. Model fails to load in Triton 

**Symptom**: The Triton server pod starts but logs show errors loading the model (or the model does not appear in the /v2/repository/index output). 
**Possible causes**: 
* The model files were not present and the Triton server could not download them (for example, due to lack of internet connectivity or incorrect model name). 
* Misconfiguration in config.pbtxt or model.json (for example, mismatched names or incorrect paths). 

**Solutions**: 
* Check the Triton server log for messages related to model loading. You can fetch the logs with kubectl logs -n triton-inference deployment/triton-server --tail=50 to see recent log entries. 
* If you suspect an issue with model files, exec into the provisioner pod (or a similar utility pod) and list the contents of the model directory to ensure config.pbtxt and model.json are in the correct location. For example: 
* Verify that you see the config.pbtxt file and the 1/model.json file in the output. If the files are missing or misnamed, re-create them as described in Step 3. 
* If your cluster is offline (no internet access) and the model was not downloaded, use an alternative method to provide the model weights (see the Note in Step 3 about pre-downloading the model). Ensure the model.json refers to the correct model identifier, and that the model files are present under the version directory if you manually placed them. 

### 3. Inference requests fail or time out 

**Symptom**: The Triton server is running but client requests to the model (at the /generate endpoint) return errors or no response. 

**Possible causes**: 
* The Triton server is not yet fully initialized or the model is still loading. 
* The request payload format is incorrect (JSON fields or values don’t match the expected schema). 
* The request is too large (for example, image size or prompt length exceeds the model's max_model_len or other limits).

**Solutions**: 
* Re-check the server health endpoints to ensure the status is READY. For example, Invoke-RestMethod http://localhost:8000/v2/health/ready should return 200 OK and a simple "OK" body. If not, the model might still be loading – allow the process to finish (watch the Triton logs for a line indicating the model is loaded). 
* If the server is ready, verify that your inference request JSON matches the required format. The text_input field should be a string, and include the special tokens and image placeholder as shown in the example. The image field should contain a Base64-encoded image without any newline or metadata (just the raw Base64 string). The parameters should be a JSON object with any desired generation parameters. 
* If requests are timing out or failing after the model is loaded, check the Triton server logs for errors when handling the request: 
* Errors here can indicate issues like improperly formatted inputs or memory allocation problems. Adjust your request or configuration based on the error messages. 

### 4. GPU not recognized in provisioner pod 

**Symptom**: Inside the provisioner pod, GPU commands (like nvidia-smi) do not detect any GPU, or the pod fails to schedule on a GPU node. 
Cause: The provisioner pod YAML does not request a GPU by default (since it isn’t performing GPU operations). Kubernetes will schedule it on a non-GPU node if available.  

**Solution**: If you need GPU inside the provisioner (for example, to run GPU-enabled download or conversion tools), modify the triton-provisioner.yaml to request a GPU resource (for example, add resources.limits.nvidia.com/gpu: 1 under the container spec) and re-create the pod. Otherwise, ensure that your provisioner runs on a node that has access to the internet (for model download) or pre-load the model data onto the PVC before deploying Triton. 

If you continue to experience issues, consult the Triton Inference Server documentation and ensure that your environment meets all prerequisites. The [Microsoft documentation](https://learn.microsoft.com/azure/aks/aksarc/troubleshoot-aksarc/#gpu-nodes) may also help with cluster-specific GPU issues. 