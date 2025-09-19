# MovieLens Recommendation System Experiments

**Comparative Performance Analysis of SVD and ALS Recommendation Models on MovieLens using Apache Spark**

A comprehensive experimental framework comparing **Sequential (Surprise SVD)** and **Distributed (Spark ALS)** recommendation approaches with rigorous statistical analysis.

---

## 🎯 **Project Objectives**

1. **Compare Sequential vs Distributed** recommendation algorithms
2. **Analyze Performance Metrics**: Speed-up, Efficiency, Scalability
3. **Evaluate Recommendation Quality**: RMSE, Precision@10, Recall@10, NDCG@10
4. **Statistical Rigor**: 3 repetitions with median ± standard deviation reporting
5. **Publication-Quality Results**: High-resolution visualizations and comprehensive analysis

---

## 📋 **Requirements**

### **System Requirements**
- Python 3.7+
- Apache Spark 3.0+
- Java 8 or 11

### **Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Dataset**
Download MovieLens dataset and place files in `./Dataset/`:
- `ratings.csv`, `movies.csv`, `tags.csv`, `links.csv`

**Download from**: https://grouplens.org/datasets/movielens/

---

## 🚀 **Quick Start**

### **🎯 Fastest Way (Recommended)**
```bash
python quick_start.py
```

### **⚡ Direct Execution**
```bash
python movielens_experiments.py
```

### **📊 Complete Pipeline**
```bash
python run_complete_experiment.py
```

### **📓 Jupyter Notebook**
```bash
jupyter notebook movielens_local_experiments.ipynb
```

---

## 📁 **Project Structure**

```
📦 MovieLens Experiments (Simplified)
├── 📊 Dataset/                    # MovieLens data files (required)
│   ├── ratings.csv               # User ratings data
│   ├── movies.csv                # Movie metadata
│   ├── tags.csv                  # User-generated tags
│   └── links.csv                 # External links
├── 🔧 Essential Files
│   ├── movielens_experiments.py  # 🎯 Main experiment script
│   ├── quick_start.py           # 🚀 Fastest launcher
│   ├── run_complete_experiment.py  # Complete pipeline
│   ├── test_setup.py            # Setup verification
│   ├── movielens_local_experiments.ipynb  # Jupyter notebook
│   ├── requirements.txt          # Python dependencies
│   └── README.md                 # Documentation
└── 📁 Generated Results
    ├── raw/                     # Data splits
    ├── results/                 # CSV/JSON results
    └── figures/                 # Visualizations (20 charts)
```

---

## 🔬 **Experiment Configuration**

### **Models Compared**
- **SVD (Sequential)**: 50 factors, 20 epochs
- **ALS (Distributed)**: rank=20, regParam=0.1, maxIter=10
- **Core Configurations**: local[1], local[2], local[4], local[8]

### **Statistical Methodology**
- **3 independent runs** per configuration
- **Median ± standard deviation** reporting
- **Different random seeds** for reproducibility

### **Evaluation Metrics**
- **Accuracy**: RMSE, MAE
- **Ranking Quality**: Precision@10, Recall@10, NDCG@10, MAP@10
- **Performance**: Training time, Prediction time
- **Scalability**: Speed-up, Efficiency

---

## 📊 **Generated Visualizations**

### **Core Performance Analysis** (4 charts)
- Training Time vs Cores (with error bars)
- Speed-up vs Cores (with ideal scaling)
- RMSE vs Cores (with SVD baseline)
- Efficiency vs Cores

### **Additional Analysis** (16 charts)
- Ranking metrics comparison (Precision, Recall, NDCG)
- Cost-benefit analysis (Resource optimization)
- Detailed timing breakdown (Performance bottlenecks)
- Dataset characteristics (Data distribution)
- Comprehensive model comparison (Multi-dimensional analysis)

**Total: 20 high-quality charts** in PNG (300 DPI) + PDF formats

---

## 📈 **Expected Results**

### **Key Findings**
- **SVD**: Better accuracy (lower RMSE) but single-threaded
- **ALS**: Scalable with multiple cores but higher RMSE
- **Trade-offs**: Speed vs accuracy analysis
- **Optimal Configuration**: Resource efficiency insights

### **Generated Files**
- `./results/movielens_experiment_results.csv` - Complete results
- `./results/detailed_results.json` - Detailed data
- `./results/experiment_summary_report.txt` - Statistical summary
- `./figures/*.png` - High-quality visualizations
- `./figures/*.pdf` - Publication-ready vector graphics

---

## 🎛️ **Configuration Options**

```python
# Experiment settings
n_runs = 3               # Repetitions per experiment
test_size = 0.2          # Train/test split ratio
sample_size = 5000       # Sample size for quick testing

# SVD parameters
n_factors = 50           # Latent factors
n_epochs = 20            # Training epochs

# ALS parameters  
rank = 20                # Latent factors
regParam = 0.1           # Regularization
maxIter = 10             # Maximum iterations

# Core configurations
core_configs = ["local[1]", "local[2]", "local[4]", "local[8]"]
```

---

## 🔧 **Troubleshooting**

### **Common Issues**
1. **Spark/Java Setup**: Ensure SPARK_HOME and JAVA_HOME are set
2. **Dataset Missing**: Download MovieLens dataset to ./Dataset/
3. **Memory Issues**: Reduce sample_size or increase JVM memory
4. **Dependencies**: Run `pip install -r requirements.txt`

### **Quick Solutions**
```bash
# Test setup
python test_setup.py

# Run with sample data (faster)
# Edit sample_size in movielens_experiments.py

# Windows Spark issues
export SPARK_DRIVER_MEMORY=4g
export SPARK_EXECUTOR_MEMORY=4g
```

---

## 📋 **Success Criteria**

✅ **All metrics computed**: Runtime, Speed-Up, Efficiency, RMSE, Precision@K, Recall@K  
✅ **K=10 evaluation**: Precision@10, Recall@10, NDCG@10  
✅ **3 repetitions**: Each experiment repeated with median reporting  
✅ **Statistical rigor**: Median ± std dev with error bars  
✅ **Publication quality**: High-resolution plots and summary tables  
✅ **Memory handling**: Spark configurations run successfully  

---

## 🎓 **Academic Usage**

### **For Big Data Course**
- Demonstrates distributed computing concepts
- Compares sequential vs parallel algorithms
- Analyzes scalability and efficiency
- Provides statistical rigor for research

### **For Publications**
- All visualizations are publication-ready
- Statistical methodology is rigorous
- Results are reproducible
- Comprehensive performance analysis

---

## 🏆 **Project Status**

**✅ COMPLETE AND READY FOR USE**

The project provides a comprehensive, statistically rigorous comparison of recommendation algorithms suitable for academic research and practical applications.

---

**🎯 Ready to run experiments and generate publication-quality results!**