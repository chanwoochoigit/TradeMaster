#!/usr/bin/env python3
"""
Mask SAC MCAD Setup Verification Script
=======================================

This script verifies that all required dependencies for the Mask SAC MCAD pipeline
are correctly installed and compatible.

Usage:
    python verify_mask_sac_setup.py
"""

import sys
import importlib
from pathlib import Path

def check_version(package_name, required_version=None, min_version=None):
    """Check if a package is installed and optionally verify version"""
    try:
        module = importlib.import_module(package_name)
        
        # Get version
        version = None
        for attr in ['__version__', 'version', 'VERSION']:
            if hasattr(module, attr):
                version = getattr(module, attr)
                if callable(version):
                    version = version()
                break
        
        if version is None:
            print(f"✅ {package_name}: Installed (version unknown)")
            return True
            
        # Check version requirements
        if required_version and str(version) != required_version:
            print(f"⚠️  {package_name}: {version} (expected {required_version})")
            return False
        elif min_version and version < min_version:
            print(f"⚠️  {package_name}: {version} (minimum {min_version} required)")
            return False
        else:
            print(f"✅ {package_name}: {version}")
            return True
            
    except ImportError:
        print(f"❌ {package_name}: Not installed")
        return False
    except Exception as e:
        print(f"❌ {package_name}: Error checking - {e}")
        return False

def verify_setup():
    """Verify all Mask SAC MCAD dependencies"""
    
    print("🔍 Verifying Mask SAC MCAD Pipeline Setup")
    print("=" * 50)
    
    # Critical dependencies with exact version requirements
    critical_deps = [
        ("gym", "0.17.3"),
        ("mmengine", "0.7.2"), 
        ("timm", "0.9.2"),
    ]
    
    # Important dependencies with minimum version requirements  
    important_deps = [
        ("einops", None, "0.6.0"),
        ("torch", None, "1.13.0"),
        ("tensorflow", None, "2.11.0"),
    ]
    
    # General dependencies (version flexible)
    general_deps = [
        "seaborn",
        "tensorboard", 
        "iopath",
        "prettytable",
        "pandas",
        "numpy",
        "scipy",
        "matplotlib",
        "sklearn",
    ]
    
    print("\n📋 Critical Dependencies (exact versions required):")
    critical_ok = True
    for package, version in critical_deps:
        if not check_version(package, required_version=version):
            critical_ok = False
    
    print("\n📋 Important Dependencies (minimum versions):")
    important_ok = True
    for item in important_deps:
        if len(item) == 3:
            package, _, min_ver = item
            if not check_version(package, min_version=min_ver):
                important_ok = False
        else:
            package = item
            if not check_version(package):
                important_ok = False
    
    print("\n📋 General Dependencies:")
    general_ok = True
    for package in general_deps:
        if not check_version(package):
            general_ok = False
    
    # Test key imports
    print("\n🧪 Testing Key Imports:")
    import_tests = [
        ("mmengine.config", "Config"),
        ("einops", "rearrange"), 
        ("pm.registry", "AGENT"),
        ("pm.utils", "export_allocation_history"),
    ]
    
    import_ok = True
    for module_name, item in import_tests:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, item):
                print(f"✅ {module_name}.{item}")
            else:
                print(f"❌ {module_name}.{item} not found")
                import_ok = False
        except ImportError as e:
            print(f"❌ {module_name}: {e}")
            import_ok = False
    
    # Check file structure
    print("\n📁 Checking Project Structure:")
    required_files = [
        "mask_sac_mcad_complete_pipeline.py",
        "tools/export_allocations.py", 
        "tools/earnmore/train.py",
        "configs/regime_dates.py",
        "pm/agent/sac/mask_sac.py",
    ]
    
    structure_ok = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} not found")
            structure_ok = False
    
    # Final verdict
    print("\n" + "=" * 50)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 50)
    
    if critical_ok and important_ok and import_ok and structure_ok:
        print("🎉 ALL CHECKS PASSED!")
        print("✅ Your setup is ready for Mask SAC MCAD training.")
        print("\nTo run the pipeline:")
        print("  python mask_sac_mcad_complete_pipeline.py --quick-test")
        return True
    else:
        print("❌ SETUP INCOMPLETE")
        print("\nIssues found:")
        if not critical_ok:
            print("  - Critical dependencies missing/wrong version")
        if not important_ok:
            print("  - Important dependencies missing/outdated")
        if not import_ok:
            print("  - Import tests failed")
        if not structure_ok:
            print("  - Project structure incomplete")
        if not general_ok:
            print("  - Some general dependencies missing")
            
        print("\nTo fix:")
        print("  pip install -r mask_sac_requirements.txt")
        print("  See MASK_SAC_SETUP.md for detailed instructions")
        return False

if __name__ == "__main__":
    success = verify_setup()
    sys.exit(0 if success else 1) 