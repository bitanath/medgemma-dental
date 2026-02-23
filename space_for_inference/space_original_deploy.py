#!/usr/bin/env python3
"""
Deploy the Simple Dental Diagnosis Demo to HuggingFace Spaces with ZeroGPU.

Usage:
    python space_original_deploy.py

Requirements:
    - HF_TOKEN environment variable must be set with a valid HuggingFace token
    - Token must have write permissions to create Spaces
"""

import os
import sys

from huggingface_hub import HfApi, SpaceHardware, create_repo


def main():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN environment variable is not set.")
        print("Please set it with: export HF_TOKEN=your_token_here")
        sys.exit(1)

    SPACE_NAME = "dental-diagnosis-original"
    REPO_ID = f"justacoderwhocodes/{SPACE_NAME}"
    SPACE_TITLE = "Dental Diagnosis Original"
    SPACE_README = """---
title: Dental Diagnosis Original
emoji: 🦷
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.6.0
app_file: app.py
pinned: false
python_version: "3.12"
---

# Dental Diagnosis Original

Simple AI-powered dental X-ray analysis using Google MedGemma.

## Features
- Direct image-to-diagnosis analysis
- No intermediate detection or cropping steps

## Model Used
- google/medgemma-1.5-4b-it
"""

    print(f"Deploying Space: {REPO_ID}")
    print("=" * 50)

    api = HfApi(token=hf_token)

    print("\n1. Creating Space repository...")
    try:
        create_repo(
            repo_id=REPO_ID,
            repo_type="space",
            token=hf_token,
            space_sdk="gradio",
            private=False,
            exist_ok=True,
        )
        print(f"   ✓ Space repository created: https://huggingface.co/spaces/{REPO_ID}")
    except Exception as e:
        print(f"   ✗ Error creating repository: {e}")
        sys.exit(1)

    print("\n2. Reading space_original.py content...")
    try:
        with open("space_original.py", "r") as f:
            app_content = f.read()
        print(f"   ✓ Read {len(app_content)} characters from space_original.py")
    except Exception as e:
        print(f"   ✗ Error reading file: {e}")
        sys.exit(1)

    print("\n3. Reading requirements.txt...")
    try:
        with open("requirements.txt", "r") as f:
            requirements_content = f.read()
        print(f"   ✓ Read requirements.txt")
    except Exception as e:
        print(f"   ✗ Error reading file: {e}")
        sys.exit(1)

    print("\n4. Uploading files to Space...")

    try:
        api.upload_file(
            path_or_fileobj=app_content.encode(),
            path_in_repo="app.py",
            repo_id=REPO_ID,
            repo_type="space",
            commit_message="Upload app.py",
        )
        print("   ✓ Uploaded app.py")
    except Exception as e:
        print(f"   ✗ Error uploading app.py: {e}")
        sys.exit(1)

    try:
        api.upload_file(
            path_or_fileobj=requirements_content.encode(),
            path_in_repo="requirements.txt",
            repo_id=REPO_ID,
            repo_type="space",
            commit_message="Upload requirements.txt",
        )
        print("   ✓ Uploaded requirements.txt")
    except Exception as e:
        print(f"   ✗ Error uploading requirements.txt: {e}")
        sys.exit(1)

    try:
        api.upload_file(
            path_or_fileobj=SPACE_README.encode(),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="space",
            commit_message="Upload README.md",
        )
        print("   ✓ Uploaded README.md")
    except Exception as e:
        print(f"   ✗ Error uploading README.md: {e}")
        sys.exit(1)

    print("\n5. Requesting ZeroGPU hardware...")
    try:
        api.request_space_hardware(
            repo_id=REPO_ID,
            hardware=SpaceHardware.ZERO_A10G,
        )
        print("   ✓ Requested ZeroGPU hardware")
    except Exception as e:
        print(f"   ✗ Error requesting hardware: {e}")
        print("   Note: You may need to manually set hardware in Space settings if you don't have PRO")

    print("\n" + "=" * 50)
    print("Deployment complete!")
    print(f"\nYour Space will be available at:")
    print(f"  https://huggingface.co/spaces/{REPO_ID}")
    print("\nNote: It may take a few minutes for the Space to build and start.")


if __name__ == "__main__":
    main()
