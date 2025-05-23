# download_models.py
import os
import argparse
from huggingface_hub import snapshot_download, login
from huggingface_hub.utils import HfHubHTTPError
import getpass

def authenticate_huggingface(token=None):
    """Authenticate with Hugging Face Hub"""
    if token:
        print("Using provided token...")
        try:
            login(token=token)
            print("✅ Successfully authenticated with provided token")
            return True
        except Exception as e:
            print(f"❌ Failed to authenticate with provided token: {e}")
            return False
    
    # Try to use existing login
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"✅ Already authenticated as: {user_info['name']}")
        return True
    except:
        pass
    
    # Prompt for token
    print("🔑 Hugging Face authentication required for some models (especially Llama 3)")
    print("You can get your token from: https://huggingface.co/settings/tokens")
    
    choice = input("Do you want to enter your HF token now? (y/n): ").lower()
    if choice == 'y':
        token = getpass.getpass("Enter your Hugging Face token: ")
        try:
            login(token=token)
            print("✅ Successfully authenticated")
            return True
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            return False
    
    print("⚠️  Continuing without authentication. Some models may fail to download.")
    return False

def download_model(model_id, local_dir, require_auth=False):
    """Download a model from Hugging Face Hub"""
    print(f"📥 Downloading {model_id} to {local_dir}...")
    
    # Check if already downloaded
    if os.path.exists(local_dir) and os.listdir(local_dir):
        print(f"📁 {model_id} already exists, skipping download")
        return True
    
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.safetensors.index.json"]  # Skip some unnecessary files
        )
        print(f"✅ Successfully downloaded {model_id}")
        return True
    except HfHubHTTPError as e:
        if "401" in str(e) or "403" in str(e):
            print(f"🔒 {model_id} requires authentication or access approval")
            if require_auth:
                print(f"Please request access at: https://huggingface.co/{model_id}")
            return False
        else:
            print(f"❌ HTTP Error downloading {model_id}: {e}")
            return False
    except Exception as e:
        print(f"❌ Error downloading {model_id}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download models for testing")
    parser.add_argument("--output_dir", type=str, default="./models", 
                       help="Directory to save models")
    parser.add_argument("--models", type=str, nargs="+", 
                       choices=["openhermes", "phi3", "gemma", "llama3", "all"],
                       default=["all"], help="Which models to download")
    parser.add_argument("--hf_token", type=str, default=None,
                       help="Hugging Face token for authentication")
    parser.add_argument("--skip_auth", action="store_true",
                       help="Skip authentication (some models may fail)")
    
    args = parser.parse_args()
    
    # Model configurations
    models_config = {
        "openhermes": {
            "repo_id": "teknium/OpenHermes-2.5-Mistral-7B",
            "local_name": "OpenHermes-2.5-Mistral-7B",
            "require_auth": False,
            "size_gb": 13
        },
        "phi3": {
            "repo_id": "microsoft/Phi-3-mini-4k-instruct",
            "local_name": "Phi-3-mini-4k-instruct",
            "require_auth": False,
            "size_gb": 7
        },
        "gemma": {
            "repo_id": "google/gemma-7b-it",
            "local_name": "Gemma-7B-Instruct",
            "require_auth": True,  # Gemma requires authentication
            "size_gb": 17
        },
        "llama3": {
            "repo_id": "meta-llama/Meta-Llama-3-8B-Instruct",
            "local_name": "Llama-3-8B-Instruct",
            "require_auth": True,  # Llama requires authentication and approval
            "size_gb": 16
        }
    }
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which models to download
    if "all" in args.models:
        models_to_download = list(models_config.keys())
    else:
        models_to_download = args.models
    
    # Calculate total size
    total_size = sum(models_config[model]["size_gb"] for model in models_to_download)
    print(f"📊 Total download size: ~{total_size} GB")
    
    # Check if any models require authentication
    auth_required = any(models_config[model]["require_auth"] for model in models_to_download)
    
    # Authenticate if needed
    if auth_required and not args.skip_auth:
        authenticated = authenticate_huggingface(args.hf_token)
        if not authenticated:
            print("⚠️  Some models may fail to download without authentication")
    
    # Download models
    success_count = 0
    failed_models = []
    
    for model_key in models_to_download:
        if model_key in models_config:
            config = models_config[model_key]
            local_dir = os.path.join(args.output_dir, config["local_name"])
            
            print(f"\n📦 Processing {config['local_name']} (~{config['size_gb']} GB)")
            
            if download_model(config["repo_id"], local_dir, config["require_auth"]):
                success_count += 1
            else:
                failed_models.append(model_key)
        else:
            print(f"❌ Unknown model: {model_key}")
    
    # Summary
    print(f"\n{'='*50}")
    print(f"📈 DOWNLOAD SUMMARY")
    print(f"{'='*50}")
    print(f"✅ Successfully downloaded: {success_count}/{len(models_to_download)} models")
    
    if failed_models:
        print(f"❌ Failed models: {', '.join(failed_models)}")
        print("\n🔧 Troubleshooting failed downloads:")
        for model in failed_models:
            config = models_config[model]
            if config["require_auth"]:
                print(f"  • {model}: Requires HF authentication and possibly access approval")
                print(f"    Request access: https://huggingface.co/{config['repo_id']}")
    
    # Print model paths for testing
    print(f"\n📝 Model paths for testing scripts:")
    print("="*50)
    for model_key in models_to_download:
        if model_key in models_config and model_key not in failed_models:
            config = models_config[model_key]
            local_dir = os.path.join(args.output_dir, config["local_name"])
            print(f"{config['local_name']}:{os.path.abspath(local_dir)}")
    
    # Generate test command
    successful_models = [m for m in models_to_download if m not in failed_models]
    if successful_models:
        print(f"\n🚀 Quick test command:")
        print("="*50)
        model_paths = []
        for model_key in successful_models:
            config = models_config[model_key]
            local_dir = os.path.abspath(os.path.join(args.output_dir, config["local_name"]))
            model_paths.append(f'"{config["local_name"]}:{local_dir}"')
        
        test_cmd = f"python src/simple_evaluation.py \\\n  --model_paths {' '.join(model_paths)} \\\n  --condition_type depression \\\n  --num_questions 10"
        print(test_cmd)

if __name__ == "__main__":
    main()