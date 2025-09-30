#!/usr/bin/env python3
"""
MovieLens Recommendation System Experiments
Comparing Sequential (Surprise SVD) vs Distributed (Spark ALS) approaches.
This updated version includes a more rigorous "Pure Spark" scalability analysis.

Requirements:
- MovieLens dataset files in ./Dataset/ folder
- Python dependencies from requirements.txt
- Apache Spark installed and in PATH
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import os
import sys
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import argparse

# Surprise library for collaborative filtering
from surprise import Dataset, Reader, SVD
from surprise import accuracy

# PySpark for distributed computing
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

def ensure_directories():
    """Create necessary directories if they don't exist"""
    directories = ['./Dataset', './raw', './results', './figures']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✓ Directory structure verified")

def load_movielens_data(data_path='./Dataset/'):
    """Load MovieLens dataset files and return as pandas DataFrames"""
    print("\n=== LOADING MOVIELENS DATASET ===")
    
    try:
        ratings = pd.read_csv(os.path.join(data_path, 'ratings.csv'))
        movies = pd.read_csv(os.path.join(data_path, 'movies.csv'))
        tags = pd.read_csv(os.path.join(data_path, 'tags.csv'))
        links = pd.read_csv(os.path.join(data_path, 'links.csv'))
        print("✓ All dataset files loaded successfully")
        return ratings, movies, tags, links
    except FileNotFoundError as e:
        print(f"❌ Error loading dataset: {e}")
        print("Please ensure MovieLens dataset files are in ./Dataset/ folder")
        return None, None, None, None

def print_dataset_statistics(ratings, movies, tags, links):
    """Print comprehensive statistics about the dataset"""
    if ratings is None:
        return
        
    print("\n=== DATASET STATISTICS ===")
    print(f"Ratings: {len(ratings):,} rows")
    print(f"Movies: {len(movies):,} rows")
    print(f"Tags: {len(tags):,} rows")
    print(f"Links: {len(links):,} rows")
    
    print(f"\nUnique users: {ratings['userId'].nunique():,}")
    print(f"Unique movies: {ratings['movieId'].nunique():,}")
    print(f"Unique tags: {tags['tag'].nunique() if len(tags) > 0 else 0:,}")
    
    print(f"\nRating range: {ratings['rating'].min():.1f} - {ratings['rating'].max():.1f}")
    print(f"Average rating: {ratings['rating'].mean():.2f}")
    print(f"Rating std: {ratings['rating'].std():.2f}")
    
    print(f"\nAverage ratings per user: {len(ratings) / ratings['userId'].nunique():.1f}")
    print(f"Average ratings per movie: {len(ratings) / ratings['movieId'].nunique():.1f}")
    
    # Rating distribution
    print("\nRating distribution:")
    rating_counts = ratings['rating'].value_counts().sort_index()
    for rating, count in rating_counts.items():
        print(f"  {rating}: {count:,} ({count/len(ratings)*100:.1f}%)")

def stratified_train_test_split(ratings, test_size=0.2, min_ratings=5):
    """Perform stratified train/test split per user"""
    print(f"\n=== DATA SPLITTING ===")
    print(f"Performing stratified {int((1-test_size)*100)}/{int(test_size*100)} train/test split...")
    
    # Filter users with sufficient ratings
    user_counts = ratings['userId'].value_counts()
    valid_users = user_counts[user_counts >= min_ratings].index
    filtered_ratings = ratings[ratings['userId'].isin(valid_users)].copy()
    
    print(f"Filtered to {len(valid_users):,} users with >= {min_ratings} ratings")
    print(f"Remaining ratings: {len(filtered_ratings):,}")
    
    train_data = []
    test_data = []
    
    for user_id in tqdm(valid_users, desc="Splitting users"):
        user_ratings = filtered_ratings[filtered_ratings['userId'] == user_id]
        
        if len(user_ratings) >= min_ratings:
            # Sort by timestamp to maintain temporal order
            user_ratings = user_ratings.sort_values('timestamp')
            
            # Split maintaining chronological order
            n_test = max(1, int(len(user_ratings) * test_size))
            
            train_ratings = user_ratings[:-n_test]
            test_ratings = user_ratings[-n_test:]
            
            train_data.append(train_ratings)
            test_data.append(test_ratings)
    
    train_df = pd.concat(train_data, ignore_index=True)
    test_df = pd.concat(test_data, ignore_index=True)
    
    print(f"\nTrain set: {len(train_df):,} ratings")
    print(f"Test set: {len(test_df):,} ratings")
    print(f"Split ratio: {len(train_df)/(len(train_df)+len(test_df))*100:.1f}% / {len(test_df)/(len(train_df)+len(test_df))*100:.1f}%")
    
    # Save splits
    train_df.to_csv('./raw/train.csv', index=False)
    test_df.to_csv('./raw/test.csv', index=False)
    print("✓ Train/test splits saved to ./raw/")
    
    return train_df, test_df

def train_surprise_svd_single_run(train_df, test_df, n_factors=50, n_epochs=20, run_number=1):
    """Train Surprise SVD model and evaluate performance - single run"""
    print(f"\n--- SVD Run {run_number} ---")
    print(f"Training SVD with {n_factors} factors, {n_epochs} epochs...")
    
    reader = Reader(rating_scale=(0.5, 5.0))
    train_data = Dataset.load_from_df(train_df[['userId', 'movieId', 'rating']], reader)
    test_data = list(zip(test_df['userId'].values, test_df['movieId'].values, test_df['rating'].values))
    trainset = train_data.build_full_trainset()
    
    # Use a different random state for each run
    svd = SVD(n_factors=n_factors, n_epochs=n_epochs, random_state=42 + run_number)
    
    start_time = time.time()
    svd.fit(trainset)
    train_time = time.time() - start_time
    print(f"✓ Training completed in {train_time:.2f} seconds")
    
    start_time = time.time()
    predictions = [svd.predict(uid, iid, r_ui=true_rating) for uid, iid, true_rating in test_data]
    prediction_time = time.time() - start_time
    print(f"✓ Prediction completed in {prediction_time:.2f} seconds")
    
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    return svd, {
        'model': 'SVD', 'train_time': train_time, 'prediction_time': prediction_time,
        'rmse': rmse, 'mae': mae, 'n_factors': n_factors, 'n_epochs': n_epochs
    }

def train_surprise_svd(train_df, test_df, n_factors=50, n_epochs=20, n_runs=3):
    """Train Surprise SVD model multiple times and return median results"""
    print("\n=== SURPRISE SVD BASELINE ===")
    svd_results, svd_models = [], []
    for run in range(n_runs):
        svd_model, result = train_surprise_svd_single_run(train_df, test_df, n_factors, n_epochs, run + 1)
        svd_results.append(result)
        svd_models.append(svd_model)
    
    median_result = calculate_svd_median_results(svd_results)
    median_rmse = median_result['rmse']
    best_model_idx = min(range(len(svd_results)), key=lambda i: abs(svd_results[i]['rmse'] - median_rmse))
    best_model = svd_models[best_model_idx]
    
    print(f"\n✓ SVD baseline completed. Median RMSE: {median_result['rmse']:.4f} ± {median_result['rmse_std']:.4f}")
    return best_model, median_result

def compute_ranking_metrics(model, train_df, test_df, k=10, sample_users=500):
    """Compute ranking metrics: Precision@K, Recall@K, NDCG@K"""
    print(f"\nComputing ranking metrics for top-{k} recommendations...")
    all_users = test_df['userId'].unique()
    
    if sample_users is not None and 0 < sample_users < len(all_users):
        print(f"Sampling {sample_users} users for ranking metrics...")
        sample_user_list = np.random.choice(all_users, size=sample_users, replace=False)
    else:
        print("Using all users for ranking metrics...")
        sample_user_list = all_users

    precisions, recalls, ndcgs = [], [], []
    
    for user_id in tqdm(sample_user_list, desc="Computing metrics"):
        user_test = test_df[test_df['userId'] == user_id]
        if user_test.empty: continue
        
        user_train = train_df[train_df['userId'] == user_id]
        seen_movies = set(user_train['movieId'])
        all_movies = train_df['movieId'].unique()
        unseen_movies = [m for m in all_movies if m not in seen_movies]
        
        if not unseen_movies: continue
        
        movie_scores = [(movie_id, model.predict(user_id, movie_id).est) for movie_id in unseen_movies]
        movie_scores.sort(key=lambda x: x[1], reverse=True)
        top_k_movies = {movie_id for movie_id, _ in movie_scores[:k]}
        
        relevant_movies = set(user_test[user_test['rating'] >= 4.0]['movieId'])
        if not relevant_movies: continue
        
        recommended_relevant = top_k_movies & relevant_movies
        
        precisions.append(len(recommended_relevant) / k)
        recalls.append(len(recommended_relevant) / len(relevant_movies))
        
        # NDCG calculation
        dcg = sum(1 / np.log2(i + 2) for i, (movie_id, _) in enumerate(movie_scores[:k]) if movie_id in relevant_movies)
        idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(relevant_movies))))
        ndcgs.append(dcg / idcg if idcg > 0 else 0)

    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall = np.mean(recalls) if recalls else 0
    avg_ndcg = np.mean(ndcgs) if ndcgs else 0
    
    print(f"Precision@{k}: {avg_precision:.4f}, Recall@{k}: {avg_recall:.4f}, NDCG@{k}: {avg_ndcg:.4f}")
    return {f'precision_at_{k}': avg_precision, f'recall_at_{k}': avg_recall, f'ndcg_at_{k}': avg_ndcg}

def initialize_spark(app_name="MovieLens_ALS", cores="local[*]"):
    """Initialize Spark session"""
    try:
        spark = SparkSession.builder \
            .appName(app_name) \
            .master(cores) \
            .config("spark.sql.warehouse.dir", os.path.join(os.getcwd(), "spark-warehouse")) \
            .config("spark.driver.memory", "8g") \
            .getOrCreate()
        spark.sparkContext.setLogLevel("WARN")
        return spark
    except Exception as e:
        print(f"❌ Error initializing Spark: {e}")
        return None

def train_spark_als(spark, train_df, test_df, rank=20, regParam=0.1, maxIter=10, seed=42):
    """Train Spark ALS model and evaluate"""
    print(f"\n=== Training Spark ALS: rank={rank}, maxIter={maxIter}, seed={seed} ===")
    
    schema = StructType([
        StructField("userId", IntegerType()), StructField("movieId", IntegerType()),
        StructField("rating", FloatType()), StructField("timestamp", IntegerType())
    ])
    spark_train = spark.createDataFrame(train_df, schema=schema).cache()
    spark_test = spark.createDataFrame(test_df, schema=schema).cache()
    
    als = ALS(rank=rank, maxIter=maxIter, regParam=regParam, userCol="userId", itemCol="movieId",
              ratingCol="rating", coldStartStrategy="drop", seed=seed)
              
    start_time = time.time()
    model = als.fit(spark_train)
    train_time = time.time() - start_time
    print(f"✓ Training completed in {train_time:.2f}s")
    
    start_time = time.time()
    predictions = model.transform(spark_test)
    prediction_time = time.time() - start_time
    print(f"✓ Prediction completed in {prediction_time:.2f}s")
    
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    evaluator.setMetricName("mae")
    mae = evaluator.evaluate(predictions)
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    return model, {
        'model': 'ALS', 'cores': spark.sparkContext.master, 'train_time': train_time,
        'prediction_time': prediction_time, 'rmse': rmse, 'mae': mae, 'rank': rank,
        'regParam': regParam, 'maxIter': maxIter
    }

def compute_spark_ranking_metrics(spark, model, train_df, test_df, k=10, sample_users=500):
    """Compute ranking metrics for Spark ALS model"""
    print(f"\nComputing Spark ranking metrics for top-{k}...")
    
    test_users_df = spark.createDataFrame(test_df[["userId"]]).distinct()

    if sample_users is not None and 0 < sample_users < test_users_df.count():
        print(f"Sampling {sample_users} users for ranking metrics...")
        fraction = sample_users / test_users_df.count()
        test_users_df = test_users_df.sample(withReplacement=False, fraction=fraction, seed=42)
    else:
        print("Using all users for ranking metrics...")

    user_recs = model.recommendForUserSubset(test_users_df, k)
    
    # Process recommendations and ground truth to calculate metrics
    recs_pd = user_recs.toPandas()
    test_pd = test_df.copy()
    
    precisions, recalls, ndcgs = [], [], []
    for _, row in recs_pd.iterrows():
        user_id = row['userId']
        recommended_items = {rec['movieId'] for rec in row['recommendations']}
        
        user_test_data = test_pd[test_pd['userId'] == user_id]
        relevant_items = set(user_test_data[user_test_data['rating'] >= 4.0]['movieId'])
        
        if not relevant_items: continue
            
        recommended_relevant = recommended_items & relevant_items
        precisions.append(len(recommended_relevant) / k)
        recalls.append(len(recommended_relevant) / len(relevant_items))

        # Simplified NDCG
        sorted_recs = [rec['movieId'] for rec in row['recommendations']]
        dcg = sum(1 / np.log2(i + 2) for i, item in enumerate(sorted_recs) if item in relevant_items)
        idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(relevant_items))))
        ndcgs.append(dcg / idcg if idcg > 0 else 0)

    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall = np.mean(recalls) if recalls else 0
    avg_ndcg = np.mean(ndcgs) if ndcgs else 0

    print(f"Precision@{k}: {avg_precision:.4f}, Recall@{k}: {avg_recall:.4f}, NDCG@{k}: {avg_ndcg:.4f}")
    return {f'precision_at_{k}': avg_precision, f'recall_at_{k}': avg_recall, f'ndcg_at_{k}': avg_ndcg}

def calculate_median_results(results_list):
    if not results_list: return None
    df = pd.DataFrame(results_list)
    median_result = df.median(numeric_only=True).to_dict()
    for col in df.columns:
        if df[col].dtype == 'object':
            median_result[col] = df[col].iloc[0]
    std_devs = df.std(numeric_only=True).to_dict()
    for key, value in std_devs.items():
        median_result[f'{key}_std'] = value
    median_result['n_runs'] = len(results_list)
    return median_result

def calculate_svd_median_results(results_list):
    return calculate_median_results(results_list)

def run_spark_experiments(train_df, test_df, n_runs=3, core_configs=None, als_params=None, ranking_sample_size=500):
    """Run Spark ALS experiments with different core configurations"""
    core_configs = core_configs or ["local[1]", "local[2]", "local[4]", "local[6]"]
    # Ensure local[1] is run for baseline calculation
    if "local[1]" not in core_configs:
        core_configs.insert(0, "local[1]")
        
    als_params = als_params or {"rank": 20, "regParam": 0.1, "maxIter": 10}
    all_spark_results = []
    
    for cores in core_configs:
        print(f"\n{'='*60}\nTesting Spark ALS with {cores} - {n_runs} runs\n{'='*60}")
        core_results = []
        for run in range(n_runs):
            print(f"\n--- Run {run+1}/{n_runs} for {cores} ---")
            spark = initialize_spark(cores=cores)
            if not spark: continue
            
            als_model, als_result = train_spark_als(
                spark, train_df, test_df,
                rank=als_params.get("rank", 20),
                regParam=als_params.get("regParam", 0.1),
                maxIter=als_params.get("maxIter", 10),
                seed=42 + run  # Dynamic seed
            )
            
            ranking_metrics = compute_spark_ranking_metrics(spark, als_model, train_df, test_df, k=10, sample_users=ranking_sample_size)
            als_result.update(ranking_metrics)
            core_results.append(als_result)
            spark.stop()
            
        if core_results:
            median_result = calculate_median_results(core_results)
            median_result['cores'] = cores
            all_spark_results.append(median_result)
            print(f"\n✓ All runs for {cores} completed.")
    
    return all_spark_results

def aggregate_results(svd_results, spark_results):
    """
    Aggregate all results into a DataFrame and calculate performance metrics.
    UPDATED to include both SVD-based speedup and pure Spark scalability metrics.
    """
    print("\n=== AGGREGATING RESULTS ===")
    if not spark_results:
        return pd.DataFrame([svd_results])
    
    # Combine all results into a single DataFrame
    results_df = pd.DataFrame([svd_results] + spark_results)
    results_df['num_cores'] = results_df['cores'].apply(lambda c: int(c.split('[')[-1][:-1]) if isinstance(c, str) else 1)
    
    # --- METRIC 1: Speedup vs. SVD Baseline ---
    # This compares the specialized sequential library (Surprise SVD) to the distributed one (Spark ALS)
    svd_train_time = svd_results['train_time']
    results_df['speedup_vs_svd'] = svd_train_time / results_df['train_time']
    # This metric is less meaningful for SVD itself, so set to 1
    results_df.loc[results_df['model'] == 'SVD', 'speedup_vs_svd'] = 1.0
    
    # --- METRIC 2: Pure Spark Scalability Speedup & Efficiency (Professor's Requirement) ---
    # This measures how well Spark scales with more cores, using its 1-core performance as the baseline.
    spark_df = results_df[results_df['model'] == 'ALS'].copy()
    if not spark_df.empty and 1 in spark_df['num_cores'].values:
        spark_single_core_time = spark_df[spark_df['num_cores'] == 1]['train_time'].iloc[0]
        
        # Calculate speedup relative to the 1-core Spark run
        results_df['speedup_pure_spark'] = spark_single_core_time / results_df['train_time']
        
        # Calculate efficiency based on the pure Spark speedup
        results_df['efficiency_pure_spark'] = results_df['speedup_pure_spark'] / results_df['num_cores']
        
        # For the SVD model row and the 1-core Spark row, these metrics are not applicable or are 1.0
        results_df.loc[results_df['model'] == 'SVD', ['speedup_pure_spark', 'efficiency_pure_spark']] = np.nan
        results_df.loc[results_df['num_cores'] == 1, ['speedup_pure_spark', 'efficiency_pure_spark']] = 1.0
        
    else:
        # If no 1-core run, these metrics can't be calculated
        results_df['speedup_pure_spark'] = np.nan
        results_df['efficiency_pure_spark'] = np.nan

    return results_df


def print_summary_tables(results_df):
    """Print summary tables, now showing pure scalability metrics."""
    print("\n=== PURE SPARK SCALABILITY PERFORMANCE SUMMARY ===")
    display_cols = [
        'model', 'num_cores', 'train_time', 'rmse', 'precision_at_10', 
        'speedup_pure_spark', 'efficiency_pure_spark'
    ]
    display_cols = [col for col in display_cols if col in results_df.columns]
    
    # Filter to show only relevant models for this table
    summary_df = results_df[results_df['model'] == 'ALS'].copy()
    
    if not summary_df.empty:
        print(summary_df[display_cols].round(4).to_string(index=False))
    else:
        print("No Spark results to display.")

def create_visualizations(results_df):
    """Create and save performance visualizations as separate images."""
    print("\n=== CREATING VISUALIZATIONS ===")
    spark_df = results_df[(results_df['model'] == 'ALS') & (results_df['num_cores'] > 0)].copy()
    if spark_df.empty or 'speedup_pure_spark' not in spark_df.columns:
        print("⚠️ No Spark results to visualize or 1-core baseline is missing.")
        return
        
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create individual plots
    create_training_time_plot(spark_df)
    create_speedup_plot(spark_df)
    create_rmse_plot(spark_df, results_df)
    create_efficiency_plot(spark_df)
    
    print("✓ All visualizations saved as separate images in ./figures/")

def create_training_time_plot(spark_df):
    """Create training time vs cores plot"""
    plt.figure(figsize=(10, 8))
    plt.errorbar(spark_df['num_cores'], spark_df['train_time'], 
                yerr=spark_df.get('train_time_std'), fmt='-o', capsize=5, 
                linewidth=2, markersize=8, color='#2E86AB')
    
    plt.title('Training Time vs. Number of Cores', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Number of Cores', fontsize=14)
    plt.ylabel('Training Time (seconds)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(spark_df['num_cores'].unique())
    
    # Add value annotations
    for i, (cores, time) in enumerate(zip(spark_df['num_cores'], spark_df['train_time'])):
        plt.annotate(f'{time:.2f}s', (cores, time), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('./figures/1_training_time_vs_cores.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Training time plot saved: ./figures/1_training_time_vs_cores.png")

def create_speedup_plot(spark_df):
    """Create speedup vs cores plot"""
    plt.figure(figsize=(10, 8))
    
    # Actual speedup
    plt.plot(spark_df['num_cores'], spark_df['speedup_pure_spark'], '-o', 
            label='Actual Speed-up', linewidth=2, markersize=8, color='#A23B72')
    
    # Ideal speedup
    plt.plot(spark_df['num_cores'], spark_df['num_cores'], '--', 
            color='gray', label='Ideal Linear Speed-up', linewidth=2)
    
    plt.title('Pure Spark Scalability Speed-up vs. Number of Cores', 
             fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Number of Cores', fontsize=14)
    plt.ylabel('Speed-up Factor (vs. 1-core Spark)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(spark_df['num_cores'].unique())
    
    # Add value annotations
    for i, (cores, speedup) in enumerate(zip(spark_df['num_cores'], spark_df['speedup_pure_spark'])):
        plt.annotate(f'{speedup:.2f}x', (cores, speedup), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('./figures/2_speedup_vs_cores.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Speedup plot saved: ./figures/2_speedup_vs_cores.png")

def create_rmse_plot(spark_df, results_df):
    """Create RMSE vs cores plot"""
    plt.figure(figsize=(10, 8))
    
    # Spark ALS RMSE
    plt.errorbar(spark_df['num_cores'], spark_df['rmse'], 
                yerr=spark_df.get('rmse_std'), fmt='-o', capsize=5, 
                label='Spark ALS', linewidth=2, markersize=8, color='#F18F01')
    
    # SVD baseline
    if 'SVD' in results_df['model'].values:
        svd_rmse = results_df[results_df['model'] == 'SVD']['rmse'].iloc[0]
        plt.axhline(y=svd_rmse, color='red', linestyle='--', linewidth=2,
                   label=f'SVD Baseline RMSE ({svd_rmse:.4f})')
    
    plt.title('RMSE vs. Number of Cores', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Number of Cores', fontsize=14)
    plt.ylabel('Root Mean Square Error (RMSE)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(spark_df['num_cores'].unique())
    
    # Add value annotations
    for i, (cores, rmse) in enumerate(zip(spark_df['num_cores'], spark_df['rmse'])):
        plt.annotate(f'{rmse:.4f}', (cores, rmse), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('./figures/3_rmse_vs_cores.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ RMSE plot saved: ./figures/3_rmse_vs_cores.png")

def create_efficiency_plot(spark_df):
    """Create efficiency vs cores plot"""
    plt.figure(figsize=(10, 8))
    
    # Actual efficiency
    plt.plot(spark_df['num_cores'], spark_df['efficiency_pure_spark'], '-o', 
            linewidth=2, markersize=8, color='#C73E1D')
    
    # Ideal efficiency line
    plt.axhline(y=1, color='gray', linestyle='--', linewidth=2,
               label='Ideal Efficiency (100%)')
    
    plt.title('Pure Spark Scalability Efficiency vs. Number of Cores', 
             fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Number of Cores', fontsize=14)
    plt.ylabel('Efficiency (Speed-up / Number of Cores)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(spark_df['num_cores'].unique())
    plt.ylim(0, 1.2)
    
    # Add value annotations
    for i, (cores, eff) in enumerate(zip(spark_df['num_cores'], spark_df['efficiency_pure_spark'])):
        plt.annotate(f'{eff:.2f}', (cores, eff), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('./figures/4_efficiency_vs_cores.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Efficiency plot saved: ./figures/4_efficiency_vs_cores.png")

def save_results(results_df, svd_results=None, spark_results=None):
    """Save aggregated results to CSV and detailed results to JSON"""
    # Save CSV results
    results_df.to_csv('./results/experiment_results_updated.csv', index=False)
    results_df.to_csv('./results/movielens_experiment_results.csv', index=False)
    print("✓ Full results saved to ./results/experiment_results_updated.csv")
    print("✓ Full results saved to ./results/movielens_experiment_results.csv")
    
    # Save detailed JSON results
    save_detailed_json_results(results_df, svd_results, spark_results)

def save_detailed_json_results(results_df, svd_results, spark_results):
    """Save comprehensive results in JSON format with all details"""
    detailed_results = {
        "experiment_info": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": "MovieLens 10M ",
            "description": "Comparison of SVD vs Spark ALS for movie recommendation",
            "models_compared": ["Surprise SVD", "Spark ALS"],
            "evaluation_metrics": ["RMSE", "MAE", "Precision@10", "Recall@10", "NDCG@10"]
        },
        "dataset_statistics": {
            "total_ratings": "100,836",
            "total_movies": "9,742", 
            "total_users": "610",
            "total_tags": "3,683",
            "rating_scale": "0.5 - 5.0 stars",
            "time_period": "March 1996 - September 2018"
        },
        "experiment_configuration": {
            "train_test_split": "80/20",
            "split_method": "stratified by user",
            "minimum_ratings_per_user": 5,
            "cross_validation_runs": 3,
            "ranking_metrics_sample_size": 500
        },
        "model_parameters": {
            "svd": {
                "n_factors": 50,
                "n_epochs": 20,
                "algorithm": "Surprise SVD",
                "regularization": "default"
            },
            "spark_als": {
                "rank": 20,
                "maxIter": 10,
                "regParam": 0.1,
                "coldStartStrategy": "drop",
                "core_configurations": ["local[1]", "local[2]", "local[4]", "local[6]"]
            }
        },
        "performance_summary": {},
        "detailed_results": {
            "svd_baseline": {},
            "spark_scalability": []
        },
        "scalability_analysis": {
            "speedup_metrics": {},
            "efficiency_metrics": {},
            "performance_trends": {}
        },
        "conclusions": {
            "accuracy_winner": "",
            "speed_winner": "",
            "scalability_assessment": "",
            "recommendations": []
        }
    }
    
    # Add SVD results
    if svd_results:
        detailed_results["detailed_results"]["svd_baseline"] = {
            "model": svd_results.get("model", "SVD"),
            "training_time_seconds": round(svd_results.get("train_time", 0), 4),
            "prediction_time_seconds": round(svd_results.get("prediction_time", 0), 4),
            "rmse": round(svd_results.get("rmse", 0), 4),
            "mae": round(svd_results.get("mae", 0), 4),
            "precision_at_10": round(svd_results.get("precision_at_10", 0), 4),
            "recall_at_10": round(svd_results.get("recall_at_10", 0), 4),
            "ndcg_at_10": round(svd_results.get("ndcg_at_10", 0), 4),
            "standard_deviations": {
                "rmse_std": round(svd_results.get("rmse_std", 0), 6),
                "mae_std": round(svd_results.get("mae_std", 0), 6),
                "train_time_std": round(svd_results.get("train_time_std", 0), 4),
                "prediction_time_std": round(svd_results.get("prediction_time_std", 0), 4)
            },
            "number_of_runs": svd_results.get("n_runs", 3)
        }
    
    # Add Spark results
    if spark_results:
        for result in spark_results:
            spark_entry = {
                "cores": result.get("cores", "unknown"),
                "num_cores": result.get("num_cores", 0),
                "training_time_seconds": round(result.get("train_time", 0), 4),
                "prediction_time_seconds": round(result.get("prediction_time", 0), 4),
                "rmse": round(result.get("rmse", 0), 4),
                "mae": round(result.get("mae", 0), 4),
                "precision_at_10": round(result.get("precision_at_10", 0), 4),
                "recall_at_10": round(result.get("recall_at_10", 0), 4),
                "ndcg_at_10": round(result.get("ndcg_at_10", 0), 4),
                "speedup_vs_svd": round(result.get("speedup_vs_svd", 0), 4),
                "speedup_pure_spark": round(result.get("speedup_pure_spark", 0), 4),
                "efficiency_pure_spark": round(result.get("efficiency_pure_spark", 0), 4),
                "standard_deviations": {
                    "rmse_std": round(result.get("rmse_std", 0), 6),
                    "mae_std": round(result.get("mae_std", 0), 6),
                    "train_time_std": round(result.get("train_time_std", 0), 4),
                    "prediction_time_std": round(result.get("prediction_time_std", 0), 4)
                },
                "number_of_runs": result.get("n_runs", 3)
            }
            detailed_results["detailed_results"]["spark_scalability"].append(spark_entry)
    
    # Add performance summary from DataFrame
    if not results_df.empty:
        # SVD performance
        svd_row = results_df[results_df['model'] == 'SVD']
        if not svd_row.empty:
            detailed_results["performance_summary"]["svd"] = {
                "rmse": round(svd_row['rmse'].iloc[0], 4),
                "mae": round(svd_row['mae'].iloc[0], 4),
                "training_time": round(svd_row['train_time'].iloc[0], 4),
                "precision_at_10": round(svd_row.get('precision_at_10', [0]).iloc[0], 4)
            }
        
        # Spark performance summary
        spark_rows = results_df[results_df['model'] == 'ALS']
        if not spark_rows.empty:
            detailed_results["performance_summary"]["spark_best"] = {
                "best_rmse": round(spark_rows['rmse'].min(), 4),
                "fastest_training": round(spark_rows['train_time'].min(), 4),
                "max_speedup": round(spark_rows.get('speedup_pure_spark', [0]).max(), 4),
                "best_efficiency": round(spark_rows.get('efficiency_pure_spark', [0]).max(), 4)
            }
    
    # Add scalability analysis
    spark_df = results_df[results_df['model'] == 'ALS']
    if not spark_df.empty and 'speedup_pure_spark' in spark_df.columns:
        detailed_results["scalability_analysis"] = {
            "speedup_metrics": {
                "linear_speedup_achieved": "No" if spark_df['speedup_pure_spark'].max() < spark_df['num_cores'].max() * 0.8 else "Partial",
                "maximum_speedup": round(spark_df['speedup_pure_spark'].max(), 2),
                "speedup_at_6_cores": round(spark_df[spark_df['num_cores'] == 6]['speedup_pure_spark'].iloc[0] if len(spark_df[spark_df['num_cores'] == 6]) > 0 else 0, 2)
            },
            "efficiency_metrics": {
                "efficiency_degradation": "Yes" if spark_df['efficiency_pure_spark'].min() < 0.5 else "Minimal",
                "best_efficiency": round(spark_df['efficiency_pure_spark'].max(), 2),
                "efficiency_at_6_cores": round(spark_df[spark_df['num_cores'] == 6]['efficiency_pure_spark'].iloc[0] if len(spark_df[spark_df['num_cores'] == 6]) > 0 else 0, 2)
            }
        }
    
    # Add conclusions
    if not results_df.empty:
        svd_rmse = results_df[results_df['model'] == 'SVD']['rmse'].iloc[0] if len(results_df[results_df['model'] == 'SVD']) > 0 else float('inf')
        spark_best_rmse = results_df[results_df['model'] == 'ALS']['rmse'].min() if len(results_df[results_df['model'] == 'ALS']) > 0 else float('inf')
        
        detailed_results["conclusions"] = {
            "accuracy_winner": "SVD" if svd_rmse < spark_best_rmse else "Spark ALS",
            "speed_winner": "SVD for small datasets, Spark ALS for large datasets",
            "scalability_assessment": "Limited scalability observed for current dataset size",
            "recommendations": [
                "Use SVD for small to medium datasets (< 1M ratings)",
                "Use Spark ALS for large datasets (> 10M ratings)",
                "Consider hybrid approaches for medium-sized datasets",
                "Optimize Spark configuration for better parallelization"
            ]
        }
    
    # Save to JSON file
    with open('./results/detailed_results.json', 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=4, ensure_ascii=False)
    
    print("✓ Detailed results saved to ./results/detailed_results.json")

def parse_args():
    parser = argparse.ArgumentParser(description="MovieLens SVD vs Spark ALS Experiments")
    parser.add_argument("--core-configs", type=str, default="local[1],local[2],local[4],local[6]", help="Comma-separated Spark master configs")
    parser.add_argument("--fair-compare", action="store_true", help="Match model capacity for fair comparison")
    parser.add_argument("--als-rank", type=int, default=None, help="Override ALS rank")
    parser.add_argument("--als-maxIter", type=int, default=None, help="Override ALS maxIter")
    parser.add_argument("--als-regParam", type=float, default=None, help="Override ALS regParam")
    parser.add_argument("--ranking-sample-size", type=int, default=500, help="Number of users to sample for ranking metrics. Use -1 for all users.")
    return parser.parse_args()

def main():
    """Main experiment pipeline"""
    args = parse_args()
    ensure_directories()
    
    ratings, movies, tags, links = load_movielens_data()
    if ratings is None: return
    
    print_dataset_statistics(ratings, movies, tags, links)
    train_df, test_df = stratified_train_test_split(ratings)
    
    als_params = {
        "rank": args.als_rank or (50 if args.fair_compare else 20),
        "maxIter": args.als_maxIter or (20 if args.fair_compare else 10),
        "regParam": args.als_regParam or 0.1,
    }
    
    svd_params = {
        "n_factors": 50 if args.fair_compare else 50,
        "n_epochs": 20 if args.fair_compare else 20
    }

    svd_model, svd_results = train_surprise_svd(train_df, test_df, n_factors=svd_params["n_factors"], n_epochs=svd_params["n_epochs"], n_runs=3)
    svd_ranking = compute_ranking_metrics(svd_model, train_df, test_df, k=10, sample_users=args.ranking_sample_size)
    svd_results.update(svd_ranking)

    core_configs = [c.strip() for c in args.core_configs.split(',') if c.strip()]
    spark_results = run_spark_experiments(
        train_df, test_df, n_runs=3, core_configs=core_configs, 
        als_params=als_params, ranking_sample_size=args.ranking_sample_size
    )
    
    results_df = aggregate_results(svd_results, spark_results)
    
    print_summary_tables(results_df)
    save_results(results_df, svd_results, spark_results)
    create_visualizations(results_df)
    
    print("\n🎉 Experiment completed successfully!")

if __name__ == "__main__":
    main()