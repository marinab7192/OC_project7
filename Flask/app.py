from flask import Flask, jsonify, request
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__) 

# CHARGEMENT DES DONNEES
data = pd.read_csv('app_train_sampled.csv', nrows=10000)
df = data.copy()
df.rename(columns = {"SK_ID_CURR": "CUSTOMER ID"}, 
          inplace = True)
df = df.set_index('CUSTOMER ID')
target = df['TARGET']
df = df.drop(columns = ['TARGET'])
features = df.columns

@app.route("/")
def home():
    return "API Flask OC projet 7"

@app.route("/cust_id")
def cust_id():
    return json.dumps(data.SK_ID_CURR.unique().tolist())

@app.route("/features")
def features_tot():
    return json.dumps(features.unique().tolist())

@app.route("/data_cust")
def data_cust():
    CUSTOMER_ID = int(request.args.get("CUSTOMER_ID"))
    data_cust = df.loc[CUSTOMER_ID, :]
    data_cust_json = json.loads(data_cust.to_json())
    return json.dumps(data_cust_json)

# IMPUTATION DES VALEURS MANQUANTES PAR LA MEDIANE
imputer = SimpleImputer(strategy = 'median')
df_model = imputer.fit_transform(df)
df_model = pd.DataFrame(df_model, index = df.index, columns = features)

# RECUPERATION DU MODELE
model = pickle.load(open("model.pkl","rb")) 

@app.route("/predict")
def predict():
    CUSTOMER_ID = int(request.args.get("CUSTOMER_ID"))
    data = df_model.loc[CUSTOMER_ID:CUSTOMER_ID] 
    score_cust = model.predict_proba(data)[0,1]*100
    return jsonify({'score': score_cust})

@app.route("/feature_imp")
def feature_imp():
    feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_, features)), 
                               columns = ['Value', 'Feature'])
    feat_imp_json = json.loads(feature_imp.to_json())
    return json.dumps(feat_imp_json)  
       
@app.route("/lime")
def lime():
    CUSTOMER_ID = int(request.args.get("CUSTOMER_ID"))
    data = df_model.loc[CUSTOMER_ID:CUSTOMER_ID]
    explainer = LimeTabularExplainer(np.array(df_model), mode = "classification",
                                     class_names = [0, 1],
                                     feature_names = features)
    exp = explainer.explain_instance(data.to_numpy()[0], 
                                     model.predict_proba,
                                     num_features = 30)
    lime_data_explainations = {}
    for feat_index, ex in exp.as_map()[1]:
        lime_data_explainations[list(features)[feat_index]] = ex
    lime_features = list(lime_data_explainations.keys())[::-1]
    explanations = list(lime_data_explainations.values())[::-1] 
    return jsonify({'Features': lime_features,
                    'Explanations': explanations})

# STANDARDISATION DES DONNEES POUR VISUALISATION 
scaler = StandardScaler()
array_scaled = scaler.fit_transform(df_model)
df_scaled = pd.DataFrame(array_scaled, index = df.index, columns = features)

@app.route("/data_10_neighbors")
def data_10_neighbors():
    CUSTOMER_ID = int(request.args.get("CUSTOMER_ID"))
    neigh = NearestNeighbors(n_neighbors = 11)
    neigh.fit(df_scaled)
    idx = neigh.kneighbors(df_scaled.loc[CUSTOMER_ID:CUSTOMER_ID], 
                           11, return_distance=False).ravel()
    neighbors_idx = list(df_scaled.iloc[idx].index)
    data_neighbors = df_scaled.loc[neighbors_idx, :]
    data_neighbors_json = json.loads(data_neighbors.to_json())
    return json.dumps(data_neighbors_json)

@app.route("/data_cust_for_visu")
def data_cust_for_visu():
    CUSTOMER_ID = int(request.args.get("CUSTOMER_ID"))
    data_cust_visu = df_scaled.loc[CUSTOMER_ID, :]
    data_cust_visu_json = json.loads(data_cust_visu.to_json())
    return json.dumps(data_cust_visu_json)

# AJOUT DE LA TARGET AUX DONNEES STANDARDISEES
df_scaled_target = df_scaled
df_scaled_target['TARGET'] = target

@app.route("/data_for_visu")
def data_for_visu():
    data_visu = df_scaled_target
    data_visu = data_visu.sample(n = 2000)
    data_visu_json = json.loads(data_visu.to_json())
    return json.dumps(data_visu_json)

if __name__ == "__main__":     
    app.run(debug=True)
