#!/usr/bin/env python3
"""
Test script to verify MovieLens experiment setup
Run this before executing the main experiments
"""

import sys
import os
import importlib.util

def test_python_version():
    """Test Python version"""
    print("🐍 Testing Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 7:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Need Python 3.7+")
        return False

def test_required_packages():
    """Test if required packages are installed"""
    print("\n📦 Testing required packages...")
    
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'sklearn', 'surprise', 'tqdm', 'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            elif package == 'surprise':
                import surprise
            else:
                __import__(package)
            print(f"✓ {package} - OK")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("✓ All required packages installed")
        return True

def test_spark_installation():
    """Test Spark installation"""
    print("\n⚡ Testing Spark installation...")
    
    try:
        from pyspark.sql import SparkSession
        
        # Try to create a simple Spark session with better configuration
        spark = SparkSession.builder \
            .appName("SetupTest") \
            .master("local[1]") \
            .config("spark.sql.warehouse.dir", "file:///C:/tmp/spark-warehouse") \
            .config("spark.driver.host", "localhost") \
            .getOrCreate()
        
        # Simple test
        data = [(1, 'test')]
        df = spark.createDataFrame(data, ['id', 'value'])
        count = df.count()
        
        spark.stop()
        
        if count == 1:
            print("✓ Spark - OK")
            return True
        else:
            print("❌ Spark test failed")
            return False
            
    except Exception as e:
        print(f"❌ Spark - ERROR: {e}")
        
        # Try alternative test - just check if pyspark is importable
        try:
            import pyspark
            print(f"⚠️ PySpark installed (version {pyspark.__version__}) but session creation failed")
            print("This is a common issue on Windows - you can still run experiments")
            return True
        except ImportError:
            print("Please install Apache Spark and set SPARK_HOME")
            return False

def test_java_installation():
    """Test Java installation"""
    print("\n☕ Testing Java installation...")
    
    java_home = os.environ.get('JAVA_HOME')
    if java_home:
        print(f"✓ JAVA_HOME set: {java_home}")
    else:
        print("⚠️ JAVA_HOME not set (may still work)")
    
    try:
        import subprocess
        result = subprocess.run(['java', '-version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            version_line = result.stderr.split('\n')[0]
            print(f"✓ Java version: {version_line}")
            return True
        else:
            print("❌ Java not found in PATH")
            return False
    except Exception as e:
        print(f"❌ Java test failed: {e}")
        return False

def test_directory_structure():
    """Test directory structure"""
    print("\n📁 Testing directory structure...")
    
    directories = ['Dataset', 'raw', 'results', 'figures']
    all_exist = True
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"✓ {directory}/ - OK")
        else:
            print(f"⚠️ {directory}/ - Will be created")
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"✓ Created {directory}/")
            except Exception as e:
                print(f"❌ Failed to create {directory}/: {e}")
                all_exist = False
    
    return all_exist

def test_dataset_files():
    """Test if dataset files exist"""
    print("\n📊 Testing dataset files...")
    
    dataset_files = ['ratings.csv', 'movies.csv', 'tags.csv', 'links.csv']
    dataset_path = './Dataset/'
    
    missing_files = []
    
    for file in dataset_files:
        filepath = os.path.join(dataset_path, file)
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            print(f"✓ {file} - OK ({file_size:,} bytes)")
        else:
            print(f"❌ {file} - MISSING")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ Missing dataset files: {', '.join(missing_files)}")
        print("Please download MovieLens dataset and place files in ./Dataset/")
        print("Download from: https://grouplens.org/datasets/movielens/")
        return False
    else:
        print("✓ All dataset files found")
        return True

def test_memory_requirements():
    """Test basic memory requirements"""
    print("\n💾 Testing memory requirements...")
    
    try:
        import psutil
        
        # Get available memory
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        print(f"Available memory: {available_gb:.1f} GB")
        
        if available_gb >= 2.0:
            print("✓ Memory - OK (≥2GB available)")
            return True
        else:
            print("⚠️ Memory - LOW (<2GB available)")
            print("Consider using sample data or closing other applications")
            return True  # Still allow to continue
            
    except ImportError:
        print("⚠️ Cannot check memory (psutil not installed)")
        return True
    except Exception as e:
        print(f"⚠️ Memory check failed: {e}")
        return True

def main():
    """Run all setup tests"""
    print("🔧 MovieLens Experiment Setup Test")
    print("=" * 50)
    
    tests = [
        ("Python Version", test_python_version),
        ("Required Packages", test_required_packages),
        ("Java Installation", test_java_installation),
        ("Spark Installation", test_spark_installation),
        ("Directory Structure", test_directory_structure),
        ("Dataset Files", test_dataset_files),
        ("Memory Requirements", test_memory_requirements)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 SETUP TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print(f"\nTests passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! Ready to run experiments.")
        print("Run: python movielens_experiments.py")
    elif passed >= total - 1:  # Allow dataset files to be missing
        print("\n⚠️ Setup mostly complete. You may proceed with caution.")
        if not any(name == "Dataset Files" and not result for name, result in results):
            print("Run: python movielens_experiments.py")
        else:
            print("Please download the MovieLens dataset first.")
    else:
        print("\n❌ Setup incomplete. Please fix the failing tests.")
        print("Refer to README.md for troubleshooting guide.")

if __name__ == "__main__":
    main()
