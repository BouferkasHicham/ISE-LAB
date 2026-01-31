#!/usr/bin/env python3
"""
Environment Check Script for The Refactoring Swarm.

This script validates that all prerequisites are met before starting
the refactoring process. Run this script first to ensure your
environment is properly configured.

Usage:
    python check_setup.py
"""

import sys
import os
import subprocess
from pathlib import Path


def print_header():
    """Print the check header."""
    print("\n" + "=" * 60)
    print("🔍 REFACTORING SWARM - ENVIRONMENT CHECK")
    print("=" * 60 + "\n")


def check_python_version():
    """Check Python version is 3.10 or 3.11."""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major == 3 and version.minor in [10, 11]:
        print(f"✅ Python Version: {version_str}")
        return True
    elif version.major == 3 and version.minor >= 12:
        print(f"⚠️  Python Version: {version_str}")
        print(f"   Recommended: Python 3.10 or 3.11")
        print(f"   Note: 3.12+ may work but is not officially supported by the lab.")
        return True  # Allow to continue but warn
    else:
        print(f"❌ Python Version: {version_str}")
        print(f"   Required: Python 3.10 or 3.11")
        return False


def check_env_file():
    """Check .env file exists and has API key."""
    if not os.path.exists(".env"):
        print("❌ .env file missing")
        print("   Run: cp .env.example .env")
        print("   Then add your GOOGLE_API_KEY")
        return False
    
    print("✅ .env file found")
    
    with open(".env", "r") as f:
        content = f.read()
    
    if "GOOGLE_API_KEY" not in content:
        print("❌ GOOGLE_API_KEY not found in .env")
        return False
    
    # Check if key is actually set (not just placeholder)
    if 'GOOGLE_API_KEY="your_api_key_here"' in content or "GOOGLE_API_KEY=''" in content:
        print("⚠️  GOOGLE_API_KEY is set to placeholder value")
        print("   Please add your actual API key from: https://aistudio.google.com/apikey")
        return False
    
    print("✅ GOOGLE_API_KEY configured")
    return True


def check_dependencies():
    """Check required Python packages are installed."""
    required_packages = [
        ("langchain", "langchain"),
        ("langchain_google_genai", "langchain_google_genai"),
        ("langgraph", "langgraph"),
        ("google.genai", "google.genai"),
        ("pylint", "pylint"),
        ("pytest", "pytest"),
        ("dotenv", "dotenv"),
        ("pandas", "pandas"),
        ("colorama", "colorama")
    ]
    
    missing = []
    
    for package_name, import_name in required_packages:
        try:
            if import_name == "dotenv":
                from dotenv import load_dotenv
            elif import_name == "google.genai":
                from google import genai
            else:
                __import__(import_name)
        except ImportError:
            missing.append(package_name)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages installed")
    return True


def check_directories():
    """Check and create required directories."""
    directories = ["logs", "sandbox", "src/agents", "src/tools", "src/config", "src/utils"]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"✅ Directory exists: {directory}")
    
    return True


def check_main_files():
    """Check main project files exist."""
    required_files = [
        "main.py",
        "requirements.txt",
        "src/__init__.py",
        "src/utils/logger.py"
    ]
    
    all_exist = True
    
    for filepath in required_files:
        if os.path.exists(filepath):
            print(f"✅ File exists: {filepath}")
        else:
            print(f"❌ Missing file: {filepath}")
            all_exist = False
    
    return all_exist


def check_pylint():
    """Check pylint is working."""
    try:
        result = subprocess.run(
            ["pylint", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            version = result.stdout.split("\n")[0]
            print(f"✅ Pylint: {version}")
            return True
    except Exception as e:
        pass
    
    print("❌ Pylint not working properly")
    return False


def check_pytest():
    """Check pytest is working."""
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✅ Pytest: {version}")
            return True
    except Exception as e:
        pass
    
    print("❌ Pytest not working properly")
    return False


def check_gemini_api():
    """Check Gemini API connection (optional, requires API key)."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key == "your_api_key_here":
            print("⚠️  Skipping API test (no valid key)")
            return True
        
        from google import genai
        from google.genai import types
        
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Say 'Hello' in one word"
        )
        
        if response.text:
            print("✅ Gemini API connection successful")
            return True
        
    except Exception as e:
        print(f"⚠️  Gemini API test failed: {str(e)[:50]}")
        print("   This may work later if your API key is valid")
    
    return True  # Don't fail on API check


def main():
    """Run all environment checks."""
    print_header()
    
    checks = [
        ("Python Version", check_python_version),
        ("Environment File", check_env_file),
        ("Dependencies", check_dependencies),
        ("Directories", check_directories),
        ("Main Files", check_main_files),
        ("Pylint", check_pylint),
        ("Pytest", check_pytest),
        ("Gemini API", check_gemini_api),
    ]
    
    results = []
    
    for name, check_func in checks:
        print(f"\n📋 Checking {name}...")
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Error checking {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"   {status} {name}")
    
    print(f"\n   Checks passed: {passed}/{total}")
    
    if passed == total:
        print("\n🚀 ALL CHECKS PASSED! You're ready to start coding.")
        print("\nNext steps:")
        print("   1. Add your GOOGLE_API_KEY to .env (if not done)")
        print("   2. Run: python main.py --target_dir ./sandbox/test_cases")
        return 0
    else:
        print("\n⚠️  SOME CHECKS FAILED. Please fix the issues above.")
        print("\nCommon fixes:")
        print("   - Install dependencies: pip install -r requirements.txt")
        print("   - Create .env file: cp .env.example .env")
        print("   - Get API key: https://aistudio.google.com/apikey")
        return 1


if __name__ == "__main__":
    sys.exit(main())