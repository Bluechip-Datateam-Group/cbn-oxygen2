
import mlflow
import pandas as pd 
import datetime  
import os  
import re  
import numpy as np
import pandas as pd
import warnings  
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('precision', 0)


from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

warnings.filterwarnings('ignore')

#Mail Libraries
import requests
import os
import io
import re

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    try:
        data = pd.read_csv('Training_data_mlflow_freq_p2p.csv')
    except Exception as e:
        logger.exception(
            "Unable to read csv. Error: %s", e)

    scaled_df = data[['month', 'day', 'hour',
       'minute', 'count_trans',
       'merchant_label_', 'tier_level_','Total_transacted_perminute', 'current_state_', 'MailGroup_',
       'bvn_flag_', 'kyc_status_','Fraud_Flag']].copy()
    
    ##scaled_df = scaled_df.groupby('Fraud_Flag').apply(lambda x: x.sample(n=1000)).reset_index(drop = True)
    
    X = scaled_df.drop('Fraud_Flag', axis=1)
    
    y = scaled_df.Fraud_Flag
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 42)

    rfc = RandomForestClassifier(n_estimators=400, min_samples_split= 2, min_samples_leaf= 1, max_features='sqrt', 
                                 max_depth= None, bootstrap= False)
    with mlflow.start_run():
        # log run parameters
        mlflow.log_param("dataset", data)
        rfc.fit(X_train, y_train)
        
        # score model
        mean_accuracy = rfc.score(X_train, y_train)
        print(f"Mean accuracy: {mean_accuracy}")

        # log run metrics
        mlflow.log_metric("mean accuracy", mean_accuracy)
        
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(rfc, "model", registered_model_name="rfc_freq_ptp")
        else:
            mlflow.sklearn.log_model(rfc, "model")