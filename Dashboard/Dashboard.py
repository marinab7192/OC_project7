# IMPORT DES LIBRAIRIES
import pandas as pd
import streamlit as st
import seaborn as sns
from PIL import Image
import requests
import matplotlib.pyplot as plt

# TITRE DU DASHBOARD
st.title('Loan application')

# SELECTION DU CLIENT
with st.sidebar:
    image = Image.open('Logo.jpeg')
    st.image(image)
    st.write('Marina BOUCHER')
    st.write('\n')
    CUSTOMER_ID_req = requests.get("https://oc-project7-app-9c6f626ab48b.herokuapp.com/cust_id").json()
    CUSTOMER_ID = st.multiselect("Select one customer ID:", CUSTOMER_ID_req, 
                                 max_selections = 1)
    
# MISE EN PAGE
tabs = st.tabs(['Customer data', 'Loan decision', 'Other customers situation'])

if CUSTOMER_ID:
    
    # DONNEES DU CLIENT
    with tabs[0]:
        data_cust = pd.json_normalize(requests.get("https://oc-project7-app-9c6f626ab48b.herokuapp.com/data_cust",
                                                   params = {"CUSTOMER_ID": CUSTOMER_ID}).json())\
                                      .set_index([CUSTOMER_ID])
        data_cust.index.name = "CUSTOMER ID"
        
        data_family = data_cust.copy()
        data_family = data_family[['DAYS_BIRTH', 'CNT_CHILDREN']]
        data_family['DAYS_BIRTH'] = (data_family['DAYS_BIRTH']/365).astype(int)
        data_family.rename(columns = {"DAYS_BIRTH": "AGE AT APPLICATION TIME",
                                      "CNT_CHILDREN": "CHILDREN COUNT"}, 
                           inplace = True)
        st.write("#### Family situation", data_family)

        data_finance = data_cust.copy()
        data_finance = data_finance[['FLAG_OWN_REALTY', 'FLAG_OWN_CAR', 'DAYS_EMPLOYED']] 
        data_finance['DAYS_EMPLOYED'] = (data_finance['DAYS_EMPLOYED']/365).astype(float)
        data_finance['DAYS_EMPLOYED'] = data_finance['DAYS_EMPLOYED'].round(2)
        data_finance.rename(columns = {"FLAG_OWN_REALTY": "HOME OWNER",
                                       "FLAG_OWN_CAR": "CAR OWNER",
                                       "DAYS_EMPLOYED": "YEARS EMPLOYED"}, 
                            inplace = True)
        st.write("#### Financial situation", data_finance)
        
        data_loan = data_cust[['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY']]
        data_loan.rename(columns = {"AMT_INCOME_TOTAL": "INCOME TOTAL",
                                    "AMT_CREDIT": "CREDIT AMOUNT",
                                    "AMT_ANNUITY": "ANNUITY"}, 
                         inplace = True)
        st.write("#### Loan situation", data_loan)

    # PREDICTION SUR LE CLIENT
    with tabs[1]:
        # SCORE DE LA PREDICTION
        st.write("#### Simulation results")
        score = float(pd.json_normalize(requests.get("https://oc-project7-app-9c6f626ab48b.herokuapp.com/predict",
                                                     params = {"CUSTOMER_ID": CUSTOMER_ID}).json()).iloc[0,0])
        st.write("Default probability: {0:.0f}%".format(score))
        threshold = 50
        st.write("Default model threshold: {0:.0f}%".format(threshold))
        st.write("Decision:")
        if score < threshold:
             st.success("Loan granted")
        else:
            st.warning("Loan refused")
          
        # VISUALISATION DES FACTEURS D'INFLUENCE GLOBAUX
        st.write("#### Global/local features importance")
        features_nb = st.number_input("Choose number of features to display:",
                                      0, 30, 10)
        fig = plt.figure(figsize = (8, 6))
        feature_imp = requests.get("https://oc-project7-app-9c6f626ab48b.herokuapp.com/feature_imp").text
        feature_imp = pd.read_json(feature_imp)
        sns.barplot(x = "Value", y = "Feature", 
                    data = feature_imp.sort_values(by = "Value", 
                                                   ascending = False).head(features_nb))
        colT1,colT2 = st.columns([1, 2])
        with colT2:
            st.write("**Global features importance**")
        st.write(fig)

        # VISUALISATION DES FACTEURS D'INFLUENCE LOCAUX
        colT1,colT2 = st.columns([1, 2])
        with colT2:
            st.write("**Local features importance**")
        lime_results = pd.read_json(requests.get("https://oc-project7-app-9c6f626ab48b.herokuapp.com/lime",
                                                 params = {"CUSTOMER_ID": CUSTOMER_ID}).text)
        lime_results_sorted = lime_results.sort_values(by = "Explanations", 
                                                       ascending = False, key = abs)
        fig, ax = plt.subplots(figsize=(8,6))
        cols = ['green' if x < 0 else 'red' for x in lime_results_sorted.Explanations]
        sns.barplot(x = "Explanations", y = "Features", palette = cols,
                    data = lime_results_sorted.head(features_nb))
        st.write(fig)
        
    # VISUALISATION DU CLIENT PAR RAPPORT AUX AUTRES
    with tabs[2]:
        features_req = requests.get("https://oc-project7-app-9c6f626ab48b.herokuapp.com/features").json()
        Y = st.multiselect("Select the features to display (max 10):", features_req, 
                           'DAYS_BIRTH', max_selections = 10)

        # CHARGEMENT DES DONNEES
        data_cust_scaled = pd.json_normalize(requests.get("https://oc-project7-app-9c6f626ab48b.herokuapp.com/data_cust_for_visu",
                                               params = {"CUSTOMER_ID": CUSTOMER_ID}).json())
        neigh_data = pd.read_json(requests.get("https://oc-project7-app-9c6f626ab48b.herokuapp.com/data_10_neighbors",
                                               params = {"CUSTOMER_ID": CUSTOMER_ID}).text)
        neigh_data = neigh_data.drop(CUSTOMER_ID)
        data = pd.read_json(requests.get("https://oc-project7-app-9c6f626ab48b.herokuapp.com/data_for_visu").text)
        
        # REPRESENTATION GRAPHIQUE
        fig = plt.figure(figsize = (8, 6))
        
        df_melt = data.melt(id_vars = ['TARGET'], 
                            value_vars = Y, 
                            var_name = 'Features',
                            value_name = 'Values')
        
        sns.boxplot(data = df_melt,
                    x = 'Features', y = 'Values', hue = 'TARGET',
                    palette = sns.color_palette(('g', 'r')), 
                    showfliers = False,
                    width = 0.3,
                    labels = df_melt['TARGET'].replace({0: 'Repaid loan', 
                                                        1: 'Not repaid loan'}, 
                                                        inplace = True))
        
        sns.stripplot(data = data_cust_scaled[Y], 
                      palette = sns.color_palette(('y','y')),
                      size = 8, marker = "D", edgecolor = 'white', linewidth = 1, 
                      label = 'Applicant customer')
        
        sns.stripplot(data = neigh_data.loc[:, Y], color = 'black',
                      size = 5, marker = "D", edgecolor = 'white', linewidth = 1,
                      label = '10 nearest neighboors')
        
        handles, labels = plt.gca().get_legend_handles_labels()
        
        plt.legend(handles = [handles[0], handles[1], handles[2], handles[len(Y)+2]], 
                   labels = [labels[0], labels[1], labels[2], labels[len(Y)+2]])
        plt.xticks(rotation = 90)
        st.write(fig)

else:
    st.error('Select a customer ID in order to continue')

    




     


    
    
    
    
    

    


