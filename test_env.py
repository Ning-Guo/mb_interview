import os
import sys

# 1. 解决 PyTorch 和 LightGBM 的 OpenMP 冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 2. 统一 PySpark 的 Python 运行环境
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

import pandas as pd
import lightgbm as lgb
import torch
from pyspark.sql import SparkSession

def test_environment():
    print("--- 环境状态自检 ---")
    
    # 1. Pandas
    df = pd.DataFrame({'feature': [1, 2, 3], 'label': [0, 1, 0]})
    print(f"✅ Pandas OK (v{pd.__version__}): DataFrame shape {df.shape}")
    
    # 2. LightGBM
    train_data = lgb.Dataset(df[['feature']], label=df['label'])
    params = {'objective': 'binary', 'verbose': -1}
    model = lgb.train(params, train_data, num_boost_round=1)
    print(f"✅ LightGBM OK (v{lgb.__version__}): Model trained.")
    
    # 3. PyTorch & 硬件加速
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        status = "MPS 硬件加速已开启"
    else:
        device = torch.device("cpu")
        status = "使用 CPU"
    
    x = torch.rand(3, 3).to(device)
    print(f"✅ PyTorch OK (v{torch.__version__}): {status}, Tensor sum: {x.sum().item():.2f}")
    
    # 4. PySpark & JVM
    spark = SparkSession.builder \
        .appName("InterviewEnvTest") \
        .master("local[2]") \
        .getOrCreate()
    
    spark_df = spark.createDataFrame([(1, "a"), (2, "b")], ["id", "val"])
    print(f"✅ PySpark OK (v{spark.version}): DataFrame row count = {spark_df.count()}")
    spark.stop()

if __name__ == "__main__":
    test_environment()