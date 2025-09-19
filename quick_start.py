#!/usr/bin/env python3
"""
Quick Start Script for MovieLens Experiments
Simplified launcher for immediate execution
"""

import os
import sys
import subprocess

def check_requirements():
    """Quick check for essential requirements"""
    print("🔍 Quick Requirements Check")
    print("-" * 30)
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ required")
        return False
    print("✅ Python version OK")
    
    # Check dataset
    dataset_files = ['./Dataset/ratings.csv', './Dataset/movies.csv']
    missing = [f for f in dataset_files if not os.path.exists(f)]
    
    if missing:
        print("❌ MovieLens dataset missing")
        print("   Download from: https://grouplens.org/datasets/movielens/")
        return False
    print("✅ Dataset found")
    
    # Check basic dependencies
    try:
        import pandas, numpy, matplotlib
        print("✅ Basic dependencies OK")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Quick start launcher"""
    print("🚀 MovieLens Experiments - Quick Start")
    print("=" * 50)
    
    if not check_requirements():
        print("\n❌ Requirements not met. Please fix and try again.")
        return False
    
    print("\n✅ All requirements satisfied!")
    print("\n🔬 Starting experiments...")
    print("   This will take several minutes...")
    print("   Results will be saved to ./results/ and ./figures/")
    
    try:
        # Run main experiment
        result = subprocess.run([sys.executable, 'movielens_experiments.py'], 
                              check=True, capture_output=False)
        
        print("\n🎉 Experiments completed successfully!")
        print("\n📊 Check these folders for results:")
        print("   📁 ./results/ - CSV and JSON results")
        print("   📈 ./figures/ - Visualizations")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Experiment failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️ Experiment interrupted by user")
        return False

if __name__ == "__main__":
    success = main()
    input("\nPress Enter to exit...")
    sys.exit(0 if success else 1)
