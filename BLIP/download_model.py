#!/usr/bin/env python3
"""
Download BLIP models for image captioning.
This script downloads the required BLIP model files.
"""

import os
import sys
import requests
import hashlib
from pathlib import Path
from typing import Dict, Optional

# Model URLs and checksums
MODELS = {
    'model_base_14M.pth': {
        'url': 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_14M.pth',
        'size': '14M',
        'description': 'Base model (14M parameters) - fastest, lowest quality'
    },
    'model_base.pth': {
        'url': 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth',
        'size': '113M', 
        'description': 'Base model - good balance of speed and quality'
    },
    'model_large.pth': {
        'url': 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large.pth',
        'size': '447M',
        'description': 'Large model - better quality, slower'
    },
    'model_base_capfilt_large.pth': {
        'url': 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth', 
        'size': '447M',
        'description': 'Base model with caption filtering - highest quality'
    }
}

def download_file(url: str, filename: str, chunk_size: int = 8192) -> bool:
    """Download a file with progress indication."""
    try:
        print(f"Downloading {filename}...")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}% ({downloaded:,}/{total_size:,} bytes)", end='')
                    else:
                        print(f"\rDownloaded: {downloaded:,} bytes", end='')
        
        print(f"\n✓ Downloaded {filename}")
        return True
        
    except Exception as e:
        print(f"\n✗ Failed to download {filename}: {e}")
        if os.path.exists(filename):
            os.remove(filename)
        return False

def check_blip_installation() -> bool:
    """Check if BLIP models directory exists."""
    models_dir = Path('./models')
    if not models_dir.exists():
        print("BLIP models directory not found.")
        print("Please install BLIP by cloning the repository:")
        print("  git clone https://github.com/salesforce/BLIP.git")
        print("  cd BLIP")
        print("  pip install -r requirements.txt")
        print("\nOr create a models directory with the BLIP implementation.")
        return False
    return True

def main():
    """Main function to download BLIP models."""
    print("BLIP Model Downloader")
    print("=" * 50)
    
    # Check BLIP installation
    if not check_blip_installation():
        print("\nWarning: BLIP models directory not found.")
        print("The downloaded models may not work without the BLIP implementation.")
        
        answer = input("\nContinue downloading models anyway? (y/N): ").lower()
        if answer != 'y':
            return
    
    # Show available models
    print("\nAvailable models:")
    for i, (model_name, info) in enumerate(MODELS.items(), 1):
        status = "✓" if os.path.exists(model_name) else " "
        print(f"  {status} {i}. {model_name} ({info['size']}) - {info['description']}")
    
    print("\nRecommended: model_base.pth (option 2) for best balance of speed and quality")
    
    # Get user choice
    choice = input("\nEnter model number to download (1-4), 'all' for all models, or 'q' to quit: ").lower()
    
    if choice == 'q':
        return
    elif choice == 'all':
        models_to_download = list(MODELS.keys())
    elif choice.isdigit() and 1 <= int(choice) <= len(MODELS):
        model_names = list(MODELS.keys())
        models_to_download = [model_names[int(choice) - 1]]
    else:
        print("Invalid choice.")
        return
    
    # Download selected models
    success_count = 0
    for model_name in models_to_download:
        if os.path.exists(model_name):
            print(f"✓ {model_name} already exists, skipping")
            success_count += 1
            continue
            
        model_info = MODELS[model_name]
        if download_file(model_info['url'], model_name):
            success_count += 1
    
    print(f"\nDownload complete: {success_count}/{len(models_to_download)} models ready")
    
    if success_count > 0:
        print("\nModels downloaded successfully!")
        print("You can now start the BLIP service with: python REST.py")
    else:
        print("\nNo models were downloaded successfully.")

if __name__ == '__main__':
    main()