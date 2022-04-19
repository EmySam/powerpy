# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 09:48:29 2022

@author: fadik_iopwcjj EmySam
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn import model_selection
from sklearn.linear_model import LinearRegression

from sklearn import ensemble
from sklearn import svm
from sklearn import neighbors
from sklearn import preprocessing
from sklearn.ensemble import VotingClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, LassoCV

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.api import anova_lm

@st.cache(persist=True, allow_output_mutation=True)
def load_data():
    import pandas as pd
    import numpy as np
    import matplotlib.ot as plt
    import seaborn as sns
    import streamlit as st

    from sklearn.preprocessing import StandardScaler
    from sklearn import linear_model
    from sklearn import preprocessing 
    from sklearn.model_selection import train_test_split

    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    
    from sklearn import model_selection
    from sklearn.linear_model import LinearRegression

    from sklearn import ensemble
    from sklearn import svm
    from sklearn import neighbors
    from sklearn import preprocessing
    from sklearn.ensemble import VotingClassifier
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import cross_validate
    from sklearn.metrics import mean_squared_error
    from sklearn.linear_model import Ridge, LassoCV

    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from statsmodels.stats.api import anova_lm
    import plotly.express as px


        


@st.cache(persist=True, allow_output_mutation=True)
def load_data():
    df1 = pd.read_csv("C:\\Users\\utilisateur\\Desktop\\DataScientest Data Analyst\\Projet fil rouge\\Dataset\\eco2mix-regional-cons-def.csv", sep = ";")
    df2 = pd.read_csv("C:\\Users\\utilisateur\\Desktop\\DataScientest Data Analyst\\Projet fil rouge\\Dataset\\temperature-quotidienne-regionale.csv", sep = ";")
    
    return df1, df2

df_energy, df_meteo = load_data()

@st.cache(persist=True)
def preproc1(df_energy):
     reno = {'Consommation (MW)':'Consommation',
             'Thermique (MW)':'Thermique',
             'Nucléaire (MW)':'Nucléaire',
             'Eolien (MW)':'Eolien',
             'Solaire (MW)':'Solaire',
             'Hydraulique (MW)':'Hydraulique',
             'Pompage (MW)':'Pompage',
             'Bioénergies (MW)':'Bioénergies',
             'Ech. physiques (MW)': 'Ech. physiques'}

     # On renomme les variables grâce à la méthode rename
     df_energy = df_energy.rename(reno, axis = 1)

     

     import datetime

 # On souhaite obtenir la colonne "Date" en format datetime "YY/MM" et "YY"

     df_energy['Date'] = pd.to_datetime(df_energy['Date'])

     df_energy['Date_YYMM'] = df_energy['Date'].dt.strftime('%Y-%m')
     df_energy['Date_YY'] = df_energy['Date'].dt.strftime('%Y')

 # On transforme les colonnes "Date_YYMM" et "Date_YY" au format datetime

     df_energy['Date_YYMM'] = pd.to_datetime(df_energy['Date_YYMM'])
     df_energy['Date_YY'] = pd.to_datetime(df_energy['Date_YY'])
     
     df_meteo['Date'] = pd.to_datetime(df_meteo['Date'])
    

     df_energy = df_energy.dropna(axis = 0, how = 'any', subset = ["Consommation"])
     df_energy = df_energy.dropna(axis = 0, how = 'any', subset = ["Eolien"])




     df_energy['Région_1'] = df_energy['Code INSEE région'].astype('str') + ' - ' + (df_energy['Région'])
     df_energy = df_energy.drop(columns = 'Code INSEE région', axis = 1)

     df_energy = df_energy.drop(columns = 'Nature', axis = 1)

     #3 Suppression variable 'Date - Heure'
     df_energy = df_energy.drop(columns = 'Date - Heure', axis = 1)

     df_energy = df_energy.dropna(axis = 0, how = 'any', subset = ["Consommation"])
     df_energy = df_energy.dropna(axis = 0, how = 'any', subset = ["Eolien"])


     df_energy['Nucléaire'][(df_energy['Région_1'] == '27 - Bourgogne-Franche-Comté')| 
                        (df_energy['Région_1'] =='53 - Bretagne')|
                        (df_energy['Région_1'] =='52 - Pays de la Loire')|
                        (df_energy['Région_1'] =="93 - Provence-Alpes-Côte d'Azur")|
                        (df_energy['Région_1'] =='11 - Île-de-France')] = df_energy['Nucléaire'][(df_energy['Région_1'] == '27 - Bourgogne-Franche-Comté')| 
                        (df_energy['Région_1'] =='53 - Bretagne')|
                        (df_energy['Région_1'] =='52 - Pays de la Loire')|
                        (df_energy['Région_1'] =="93 - Provence-Alpes-Côte d'Azur")|
                        (df_energy['Région_1'] =='11 - Île-de-France')].fillna(0)

     df_energy['Pompage'][(df_energy['Région_1'] == '24 - Centre-Val de Loire')| 
                        (df_energy['Région_1'] =='32 - Hauts-de-France')|
                        (df_energy['Région_1'] =='52 - Pays de la Loire')|
                        (df_energy['Région_1'] =="28 - Normandie")|
                        (df_energy['Région_1'] =='11 - Île-de-France')|
                        (df_energy['Région_1'] =="75 - Nouvelle-Aquitaine")]= df_energy['Pompage'][(df_energy['Région_1'] == '24 - Centre-Val de Loire')| 
                        (df_energy['Région_1'] =='32 - Hauts-de-France')|
                        (df_energy['Région_1'] =='52 - Pays de la Loire')|
                        (df_energy['Région_1'] =="28 - Normandie")|
                        (df_energy['Région_1'] =='11 - Île-de-France')|
                        (df_energy['Région_1'] =="75 - Nouvelle-Aquitaine")].fillna(0)


     df_energy = df_energy.drop(df_energy.columns [12:25], axis=1)
     df_energy = df_energy.reset_index(drop=True)


     return df_energy

df_energy = preproc1(df_energy)

@st.cache(persist=True)
def preproc2(df_energy, df_meteo):
     

     df_meteo['Date'] = pd.to_datetime(df_meteo['Date'])


     df_meteo['Région_2'] = df_meteo['Code INSEE région'].astype('str') + ' - ' + (df_meteo['Région'])  
     df_meteo = df_meteo.drop(columns = 'Code INSEE région', axis = 1)

     print(df_meteo.columns)
     #df_energy = df_energy.drop(columns = 'Nature', axis = 1)
     #df_energy = df_energy.drop(columns = 'Date - Heure', axis = 1)

     df_meteo['Date'] = pd.to_datetime(df_meteo['Date'])
     df_meteo['Date_YYMM'] = df_meteo['Date'].dt.strftime('%Y-%m')
     df_meteo['Date_YYMM'] = pd.to_datetime(df_meteo['Date_YYMM'])



     df_merge = df_meteo.merge(right = df_energy, on = ["Date", "Région"], how = "inner")

     df_merge["Heure2"] = pd.to_datetime(df_merge["Heure"], format = "%H:%M")

     # Conversion manuelle nouvelle series Heure2 en valeur numérique (équivalent de la fonction toordinal)

     df_merge['Heure2'] = df_merge['Heure2'].dt.hour * 60 + df_merge['Heure2'].dt.minute

     df_merge = df_merge.drop(["Région_1", "Région_2"], axis = 1)

     return df_merge

df_merge = preproc2(df_energy, df_meteo)

st.sidebar.title("Pages")
page = st.sidebar.radio(label="", options=["Introduction", "Visualisation", "Modèle de prédiction"])


if page == "Introduction":
    

    st.title("Problématiques")
    st.write("**Avec la multiplication  des catastrophes naturelles, l’accroissement de la population et des activités humaines toujours plus énergivores, quel est le niveau de pression que doit supporter le dispositif de production d’énergie en France?  Existe-t-il un risque important de black-out?**")
    st.write("**Les évolutions des parts de la production d’énergies renouvelables et de nucléaire sont-elles alignées avec les objectifs de transition écologique? Sont-elles suffisamment marquées pour attester de la réelle prise en compte des objectifs définis?**")
    st.write("**la production d’énergies renouvelables, portée par la transition écologique française, pourra-t-elle soutenir le besoin énergétique français dans les années à venir?**") 

elif page == "Visualisation":
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    #-----------------------------Partie 2 - Visualisation

    st.header("1. Phasage entre la production et la consommation d'énergie en France")
    st.subheader("1.A. Comparaison des courbes de la production nationale d'énergie vs la consommation nationale toutes filières confondues de 2013 à Novembre 2021 (MW)")

    #-----------------------------1/A: National

    
    from datetime import datetime
        
    # On utilise un groupby afin d'obtenir la somme des valeurs de consommation et production (pour chaque filière) par mois et années (en non plus par jour et par heure) sans tenir compte de la région
    df_energy_n = df_energy.groupby(['Date_YYMM']).agg({'Consommation':'sum',
                                                   'Thermique':'sum',
                                                   'Nucléaire':'sum',
                                                   'Eolien':'sum',
                                                   'Solaire':'sum',
                                                   'Hydraulique':'sum',
                                                   'Pompage':'sum',
                                                   'Bioénergies':'sum',
                                                   'Ech. physiques':'sum'})


    # On souhaite obtenir la production totale (toutes filières confondues)
    df_energy_n['Production_totale'] = df_energy_n.Thermique + df_energy_n.Nucléaire +	df_energy_n.Eolien +	df_energy_n.Solaire +	df_energy_n.Hydraulique	+ df_energy_n.Pompage +	df_energy_n.Bioénergies

    # On convertit l'index en la colonne "Date_YYMM"
    df_energy_n = df_energy_n.reset_index()

    # On trace les courbes de la consommation et production totale au niveau national dans le temps sur un même graphique
    st.sidebar.header("1.A. Phasage national")
    dates_1A = st.sidebar.date_input(label = "Choisir une période à visualiser",
                          min_value = datetime(2013, 1, 1),
                          max_value = datetime(2021, 11, 1),
                          value = [datetime(2013, 1, 1),datetime(2021, 11, 1)], key = '1A')
    df_energy_n = df_energy_n[(df_energy_n['Date_YYMM'].dt.date > dates_1A[0]) & (df_energy_n['Date_YYMM'].dt.date <= dates_1A[1])] 


    fig6 = go.Figure()
    
    fig6.add_trace(go.Scatter(x=df_energy_n.Date_YYMM, y=df_energy_n.Production_totale,
                    mode='lines+markers',
                    name='Production nationale'))
    fig6.add_trace(go.Scatter(x=df_energy_n.Date_YYMM, y=df_energy_n.Consommation,
                    mode='lines+markers',
                    name='Consommation nationale'))
    fig6.update_layout(height=500, width = 900)
    fig6.update_xaxes(position=0)
    
    
    st.plotly_chart(fig6, use_container_width=False, sharing="streamlit") 

    st.markdown("Au niveau national, nous remarquons que la production nationale toutes filières confondues couvre la consommation énergétique. Nous notons tout de même des périodes de tension où la courbe de la consommation reste très proche de la courbe de production.")


    #-----------------------------1/B: Régional

    st.subheader("1.B. Comparaison des courbes de la production régionale d'énergie vs la consommation régionale toutes filières confondues de 2013 à Novembre 2021 (MW)")

    # On utilise un groupby afin d'obtenir la somme des valeurs de consommation et production (pour chaque filière) par mois et années (en non plus par jour et par heure) et par régions
    df_energy_r = df_energy.groupby(['Date_YYMM','Région_1']).agg({'Consommation':'sum',
                                                   'Thermique':'sum',
                                                   'Nucléaire':'sum',
                                                   'Eolien':'sum',
                                                   'Solaire':'sum',
                                                   'Hydraulique':'sum',
                                                   'Pompage':'sum',
                                                   'Bioénergies':'sum',
                                                   'Ech. physiques':'sum'})

    # On souhaite obtenir la production totale (toutes filières confondues)
    df_energy_r['Production_totale'] = df_energy_r.Thermique +	df_energy_r.Nucléaire +	df_energy_r.Eolien +	df_energy_r.Solaire +	df_energy_r.Hydraulique	+ df_energy_r.Pompage +	df_energy_r.Bioénergies

    # On convertit le multi-index en 2 colonnes "Région" et "Date_YYMM"
    df_energy_r = df_energy_r.reset_index()

    # On trace les courbes de la consommation et production totale au niveau de chaque région dans le temps sur un même graphique

    st.sidebar.header("1.B. Phasage régional")
    région = st.sidebar.selectbox(label = "Choisir un région à visualiser", options = df_energy_r['Région_1'].unique(), key = "1B")
    df_energy_r = df_energy_r[df_energy_r['Région_1'] == région]

    dates_1B = st.sidebar.date_input(label = "Choisir une période à visualiser",
                          min_value = datetime(2013, 1, 1),
                          max_value = datetime(2021, 11, 1),
                          value = [datetime(2013, 1, 1),datetime(2021, 11, 1)], key = '1B')
    df_energy_r = df_energy_r[(df_energy_r['Date_YYMM'].dt.date > dates_1B[0]) & (df_energy_r['Date_YYMM'].dt.date <= dates_1B[1])] 

    fig1 = go.Figure()
    
    fig1.add_trace(go.Scatter(x=df_energy_r.Date_YYMM, y=df_energy_r.Production_totale,
                    mode='lines+markers',
                    name='Production régionale'))
    fig1.add_trace(go.Scatter(x=df_energy_r.Date_YYMM, y=df_energy_r.Consommation,
                    mode='lines+markers',
                    name='Consommation régionale'))
    fig1.update_layout(height=500, width = 900)
    
    
    st.plotly_chart(fig1, use_container_width=False, sharing="streamlit") 
    
  
    st.markdown("On peut obtenir plus de détails au niveau régional où 3 groupes de régions se distinguent :") 
    st.markdown("  - les régions qui présentent un excédent de production")
     
    st.markdown("  - les régions en auto-suffisance")
     
    st.markdown("  - les régions qui présentent un déficit de production")


    #-----------------------------2/A: National

    st.header('2. Analyse par filière de production : énergie nucléaire / renouvelable')
    st.subheader("2.A. Comparaison du Mix énergétique national de la France entre les différents années de 2013 à Novembre 2021")

    # On utilise un groupby afin d'obtenir la somme des valeurs de consommation et production (pour chaque filière) par années (en non plus par jour et par heure)

    df_energy_n_er = df_energy.groupby(['Date_YY']).agg({'Consommation':'sum',
                                                   'Thermique':'sum',
                                                   'Nucléaire':'sum',
                                                   'Eolien':'sum',
                                                   'Solaire':'sum',
                                                   'Hydraulique':'sum',
                                                   'Pompage':'sum',
                                                   'Bioénergies':'sum',
                                                   'Ech. physiques':'sum'})

    # On souhaite obtenir la production totale d'énergie renouvelable 
    df_energy_n_er['Production_totale_ER'] = df_energy_n_er.Eolien + df_energy_n_er.Solaire + df_energy_n_er.Hydraulique + df_energy_n_er.Pompage + df_energy_n_er.Bioénergies

    # On convertit l'index en une colonne "Date_YY"
    df_energy_n_er = df_energy_n_er.reset_index()




    df_energy_n_er2013 =df_energy_n_er
    st.sidebar.header("2.A. ER/nucléaire national")
    Year2013 = st.sidebar.selectbox(label = "Choisir l'année A à visualiser", options = df_energy_n_er2013['Date_YY'].unique(), key = "2A")
    df_energy_n_er2013 = df_energy_n_er2013[df_energy_n_er2013['Date_YY'] == Year2013]

    df_energy_n_er2021 =df_energy_n_er
    Year2021 = st.sidebar.selectbox(label = "Choisir l'année B à visualiser", options = df_energy_n_er2021['Date_YY'].unique(), key = "2Abis")
    df_energy_n_er2021 = df_energy_n_er2021[df_energy_n_er2021['Date_YY'] == Year2021]



    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    labels = ['Nucléaire', 'Renouvelable','Thermique']
    colors = ['#4169E1', 'lightgreen', 'gold']
    values2013 = [df_energy_n_er2013['Nucléaire'].sum(), df_energy_n_er2013['Production_totale_ER'].sum(),df_energy_n_er2013['Thermique'].sum()]
    values2021 = [df_energy_n_er2021['Nucléaire'].sum(), df_energy_n_er2021['Production_totale_ER'].sum(),df_energy_n_er2021['Thermique'].sum()]

    fig2 = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]], subplot_titles=['Année A', 'Année B'])
    fig2.add_trace(go.Pie(labels=labels, values=values2013),
                  1, 1)
    fig2.add_trace(go.Pie(labels=labels, values=values2021),
                  1, 2)
    fig2.update_traces(textfont_size=16,
                      marker=dict(colors=colors))
    fig2.update_layout(margin=dict(t=0, b=0, l=0, r=0))

    st.plotly_chart(fig2, use_container_width=False, sharing="streamlit") 

    st.markdown("Le nucléaire est la principale filière productrice d'énergie en France. Cependant, on note un essor timide des énergies renouvelables entre 2013 et 2021 au détriment des filières thermique et nucléaire.") 

        #-----------------------------2/B: Régional

    st.subheader("2.B. Comparaison des filières énergie renouvelable/nucléaire au niveau régional de 2013 à Novembre 2021 (MW)")

    # On utilise un groupby afin d'obtenir la somme des valeurs de consommation et production (pour chaque filière) par années (en non plus par jour et par heure)



        # On utilise un groupby afin d'obtenir la somme des valeurs de consommation et production (pour chaque filière) par années (en non plus par jour et par heure) et par régions
    df_energy_r_er = df_energy.groupby(['Date_YY','Région_1']).agg({'Consommation':'sum',
                                                       'Thermique':'sum',
                                                       'Nucléaire':'sum',
                                                       'Eolien':'sum',
                                                       'Solaire':'sum',
                                                       'Hydraulique':'sum',
                                                       'Pompage':'sum',
                                                       'Bioénergies':'sum',
                                                       'Ech. physiques':'sum',})

        # On souhaite obtenir la production totale (toutes filières confondues)
    df_energy_r_er['Production_totale'] = df_energy_r_er.Thermique + df_energy_r_er.Nucléaire + df_energy_r_er.Eolien + df_energy_r_er.Solaire + df_energy_r_er.Hydraulique + df_energy_r_er.Pompage + df_energy_r_er.Bioénergies

        # On souhaite obtenir la production totale d'énergie renouvelable 
    df_energy_r_er['Production_totale_ER'] = df_energy_r_er.Eolien + df_energy_r_er.Solaire + df_energy_r_er.Hydraulique + df_energy_r_er.Pompage + df_energy_r_er.Bioénergies

        # On convertit le multi-index en 2 colonnes "Région" et "Date_YY"
    df_energy_r_er = df_energy_r_er.reset_index()

    st.sidebar.header("2.B. ER/nucléaire régional")
    région_r = st.sidebar.selectbox(label = "Choisir une région à visualiser", options = df_energy_r_er['Région_1'].unique(), key = "2B")
    df_energy_r_er = df_energy_r_er[df_energy_r_er['Région_1'] == région_r]


    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=df_energy_r_er.Date_YY,
        y=df_energy_r_er["Production_totale_ER"],
        name='Production_totale_ER',
        marker_color='lightgreen'
    ))
    fig3.add_trace(go.Bar(
        x=df_energy_r_er.Date_YY,
        y=df_energy_r_er["Nucléaire"],
        name='Nucléaire',
        marker_color="#4169E1"
    ))

    fig3.update_layout(barmode='group', xaxis_tickangle=75)
    plt.legend()
    plt.show()

    st.plotly_chart(fig3, use_container_width=False, sharing="streamlit") 
    st.markdown("L'augmentation de la part des énergies renouvelables dans le mix énergétique français se confirme au niveau régional à travers une tendance à la hausse de la production des énergies vertes. On note tout de même que le nucléaire reste la source d'énergie dominante par rapport aux énergies vertes et les régions avec un niveau de production d'énergie bas se démarquent par l'absence d'unité de production nucléaire.")

    #-----------------------------3/A: National

    st.header('3. Focus sur les énergies renouvelables')
    st.subheader("3.A. Répartition de la production d'énergie renouvelable entre régions de 2013 à Novembre 2021")

    df_energy_r_ero = df_energy.groupby(['Date_YY','Région_1']).agg({'Consommation':'sum',
                                                   'Thermique':'sum',
                                                   'Nucléaire':'sum',
                                                   'Eolien':'sum',
                                                   'Solaire':'sum',
                                                   'Hydraulique':'sum',
                                                   'Pompage':'sum',
                                                   'Bioénergies':'sum',
                                                   'Ech. physiques':'sum',})

    # On souhaite obtenir la production totale (toutes filières confondues)
    df_energy_r_ero['Production_totale'] = df_energy_r_ero.Thermique + df_energy_r_ero.Nucléaire + df_energy_r_ero.Eolien + df_energy_r_ero.Solaire + df_energy_r_ero.Hydraulique + df_energy_r_ero.Pompage + df_energy_r_ero.Bioénergies

    # On souhaite obtenir la production totale d'énergie renouvelable 
    df_energy_r_ero['Production_totale_ER'] = df_energy_r_ero.Eolien + df_energy_r_ero.Solaire + df_energy_r_ero.Hydraulique + df_energy_r_ero.Pompage + df_energy_r_ero.Bioénergies

    # On convertit le multi-index en 2 colonnes "Région" et "Date_YY"
    df_energy_r_ero = df_energy_r_ero.reset_index()





    # On trie pour obtenir les 5 régions principales productrices d'énergie verte pour 2013

    df_energy_ro_er2013 =df_energy_r_ero
    df_energy_ro_er2013.sort_values('Production_totale_ER', axis = 0, ascending=False, inplace = True)
    st.sidebar.header("3.A. Focus ER national")
    Year2013b = st.sidebar.selectbox(label = "Choisir une année à visualiser", options = df_energy_ro_er2013['Date_YY'].unique(), key = "3A")
    df_energy_ro_er2013 = df_energy_ro_er2013[df_energy_ro_er2013['Date_YY'] == Year2013b]

    # On trie pour obtenir les 5 régions principales productrices d'énergie verte pour 2021
    df_energy_ro_er2021 =df_energy_r_ero
    df_energy_ro_er2021.sort_values('Production_totale_ER', axis = 0, ascending=False, inplace = True)
    Year2021b = st.sidebar.selectbox(label = "Choisir une année à visualiser", options = df_energy_ro_er2021['Date_YY'].unique(), key = "3Abis")
    df_energy_ro_er2021 = df_energy_ro_er2021[df_energy_ro_er2021['Date_YY'] == Year2021b]


    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    labelsb = ['84 - Auvergne-Rhône-Alpes', '76 - Occitanie',"93 - Provence-Alpes-Côte d'Azur",'44 - Grand Est',
              '75 - Nouvelle-Aquitaine','Autres']
    colorsb = ['#FAEBD7', '#20B2AA', '#6B8E23', '#B22222', '#F0E68C', '#FF7F50']
    values2013b = [df_energy_ro_er2013['Production_totale_ER'].iloc[0].sum(),
             df_energy_ro_er2013['Production_totale_ER'].iloc[1].sum(),
             df_energy_ro_er2013['Production_totale_ER'].iloc[2].sum(),
             df_energy_ro_er2013['Production_totale_ER'].iloc[3].sum(),
             df_energy_ro_er2013['Production_totale_ER'].iloc[4].sum(),
             df_energy_ro_er2013['Production_totale_ER'].iloc[5:].sum()]
    values2021b = [df_energy_ro_er2021['Production_totale_ER'].iloc[0].sum(),
             df_energy_ro_er2021['Production_totale_ER'].iloc[1].sum(),
             df_energy_ro_er2021['Production_totale_ER'].iloc[2].sum(),
             df_energy_ro_er2021['Production_totale_ER'].iloc[3].sum(),
             df_energy_ro_er2021['Production_totale_ER'].iloc[4].sum(),
             df_energy_ro_er2021['Production_totale_ER'].iloc[5:].sum()]
    
    st.metric(label="Total production - year A", value=df_energy_ro_er2013["Production_totale_ER"].sum())
    fig4 = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]], subplot_titles=['Année A', 'Année B'])
    fig4.add_trace(go.Pie(labels=labelsb, values=values2013b),
                  1, 1)
    st.metric(label="Total production - year B", value=df_energy_ro_er2021["Production_totale_ER"].sum(), delta=df_energy_ro_er2021["Production_totale_ER"].sum()-df_energy_ro_er2013["Production_totale_ER"].sum())
    fig4.add_trace(go.Pie(labels=labelsb, values=values2021b),
                  1, 2)
    fig4.update_traces(textfont_size=16,
                      marker=dict(colors=colorsb))
    fig4.update_layout(margin=dict(t=0, b=0, l=0, r=0))
    st.plotly_chart(fig4, use_container_width=False, sharing="streamlit") 
    st.markdown("Entre 2013 et 2021, les régions principales productrices d'énergie renouvelable restent les mêmes : Auvergne-Rhônes-Alpes, Occitanie, Nouvelle-Aquitaine, Provence-Alpes-Côte d'Azur et Grand Est")
    #-----------------------------3/B: Régional

    st.subheader("3.B. Comparaison de la part des différentes énergies renouvelables au niveau régional de 2013 à Novembre 2021 (MW)")


    df_energy_r_eru = df_energy.groupby(['Date_YY','Région_1']).agg({'Consommation':'sum',
                                                   'Thermique':'sum',
                                                   'Nucléaire':'sum',
                                                   'Eolien':'sum',
                                                   'Solaire':'sum',
                                                   'Hydraulique':'sum',
                                                   'Pompage':'sum',
                                                   'Bioénergies':'sum',
                                                   'Ech. physiques':'sum',})

    # On souhaite obtenir la production totale (toutes filières confondues)
    df_energy_r_eru['Production_totale'] = df_energy_r_eru.Thermique + df_energy_r_eru.Nucléaire + df_energy_r_eru.Eolien + df_energy_r_eru.Solaire + df_energy_r_eru.Hydraulique + df_energy_r_eru.Pompage + df_energy_r_eru.Bioénergies

    # On souhaite obtenir la production totale d'énergie renouvelable 
    df_energy_r_eru['Production_totale_ER'] = df_energy_r_eru.Eolien + df_energy_r_eru.Solaire + df_energy_r_eru.Hydraulique + df_energy_r_eru.Pompage + df_energy_r_eru.Bioénergies

    # On convertit le multi-index en 2 colonnes "Région" et "Date_YY"
    df_energy_r_eru = df_energy_r_eru.reset_index()

    st.sidebar.header("3.B. Focus ER régional")
    région_er = st.sidebar.selectbox(label = "Choisir une région à visualiser", options = df_energy_r_eru['Région_1'].unique(), key = "3B")
    df_energy_r_eru = df_energy_r_eru[df_energy_r_eru['Région_1'] == région_er]


    import plotly.express as px


    fig5 = go.Figure()
    fig5.add_trace(go.Bar(
        x=df_energy_r_eru.Date_YY,
        y=df_energy_r_eru["Hydraulique"],
        name='Hydraulique',
        marker_color='#41D1CC'
    ))
    fig5.add_trace(go.Bar(
        x=df_energy_r_eru.Date_YY,
        y=df_energy_r_eru["Solaire"],
        name='Solaire',
        marker_color="#FF6347"
    ))
    fig5.add_trace(go.Bar(
        x=df_energy_r_eru.Date_YY,
        y=df_energy_r_eru["Eolien"],
        name='Eolien',
        marker_color="#B8860B"
    ))
    fig5.add_trace(go.Bar(
        x=df_energy_r_eru.Date_YY,
        y=df_energy_r_eru["Bioénergies"],
        name='Bioénergies',
        marker_color="#BA55D3"
    ))


    fig5.update_layout(barmode='group', xaxis_tickangle=75)

    plt.legend()
    plt.show()

    st.plotly_chart(fig5, use_container_width=False, sharing="streamlit")# visualize function
    st.markdown ("L'énergie renouvelable est majoritairement hydraulique avec un faible volume de bioénergies et d'éolienne. L'énergie solaire représente un volume quasi négligeable et ne concernent que certaines régions.")
elif page == "Modèle de prédiction":
    
    st.header("Régression linéaire")
    
    @st.cache(persist=True)
    def create_df_for_ML(df_merge):

        #df_merge = df_merge.dropna(axis = 0, how = 'all', subset =['Consommation'])

        df_merge_national = df_merge.drop(['Thermique', 'Nucléaire','Eolien', 'Solaire', 'Hydraulique', 'Pompage', 'Bioénergies',
                                        'Ech. physiques'], axis = 1)

        df_merge_national["Date"] = df_merge_national['Date'].dt.month

        #df_merge_national = df_merge_national.drop(["Région_1", "Région_2"], axis = 1)

        df_merge_national = df_merge_national.groupby(by = ["Date", "Heure"]).agg({"Consommation" : "sum", "TMin (°C)" : "mean", "TMax (°C)" : "mean", "TMoy (°C)" : "mean", "Heure2" : "mean"})

        df_merge_national = df_merge_national.reset_index()

        return df_merge_national





    @st.cache(persist=True)
    def create_df_regional(df_merge, region):


        df_merge_reg = df_merge[df_merge["Région"] == region]

        return create_df_for_ML(df_merge_reg)



    def lin_reg(df_merge):
        data = df_merge.drop(["Consommation", "Heure"], axis = 1)
        target = df_merge['Consommation']
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.2)
        # Instanciation d'un régresseur lr de la classe LinearRegression.
        lrm = LinearRegression()
        # Ajustement du modèle
        lrm.fit(X_train, y_train)

        score_train = lrm.score(X_train, y_train)
        cross_val = cross_val_score(lrm, X_train, y_train).mean()
        score_test = lrm.score(X_test, y_test)

        pred_test = lrm.predict(X_test)
        fig, ax = plt.subplots()
        ax.scatter(pred_test, y_test)
        ax.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()));
        st.pyplot(fig)
        return score_train, cross_val, score_test, lrm


    def predict(df_merge, hyp, lrm):
        X_futur = df_merge.drop(["Consommation", "Heure"], axis = 1)
        X_futur = X_futur[X_futur["Date"] <= 6]
        X_futur[["TMin (°C)", "TMax (°C)", "TMoy (°C)"]] = X_futur[["TMin (°C)", "TMax (°C)", "TMoy (°C)"]] * (hyp/100.0 + 1.0)

        pred_futur = lrm.predict(X_futur) / 6

        st.subheader('Prédiction de consommation')
        st.write('Prédiction de la consommation au 1er semestre 2023, en considérant une hypothèse de réchauffement climatique de ' + str(hypo_rech) + '% (par rapport aux températures moyennes des 6 dernières années)')

        fig, ax = plt.subplots()
        ax.scatter(X_futur["TMoy (°C)"], pred_futur)
        ax.plot((X_futur["TMoy (°C)"].min(), X_futur["TMoy (°C)"].max()), (pred_futur.max(), pred_futur.min()));
        st.pyplot(fig)






    dfm = preproc2(df_energy, df_meteo)

    # st.subheader('DF mergé')
    # st.dataframe(dfm.iloc[:10,:])

    st.sidebar.header("Prédiction de consommation")
    hypo_rech = st.sidebar.slider("Hypothèse de réchauffement climatique pour S1 2023 (%)", min_value=0, max_value=100, value=10)


    st.sidebar.header("Scope")
    if st.sidebar.checkbox("Analyser et prédire par région", True, key=1):

        # CAS REGIONAL

        region_selectionnee = st.sidebar.selectbox('Sélectionnez une région ',dfm['Région'].unique())

        dfreg = create_df_regional(dfm, region_selectionnee)

        st.subheader('Dataframe régional ' + region_selectionnee + ' prêt pour ML')
        st.dataframe(dfreg.iloc[:10,:])

        st.subheader('ML: Régression linéaire')
        s, cvs, stest, model = lin_reg(dfreg)
        st.write('Score de la Linear Regression (ensemble d\'entraînement) pour ' + region_selectionnee + ' : ' + str(s))
        st.write('Cross-val score pour ' + region_selectionnee + ' : ' + str(cvs))
        st.write('Score de la Linear Regression (ensemble de test) pour ' + region_selectionnee + ' : ' + str(stest))

        predict(dfreg, hypo_rech, model)


    else:
        # CAS NATIONAL
        dfnat = create_df_for_ML(dfm)

        st.subheader('Dataframe national prêt pour ML')
        st.dataframe(dfnat.iloc[:10,:])

        st.subheader('ML: Régression linéaire')
        s, cvs, stest, model = lin_reg(dfnat)
        st.write('Score de la Linear Regression (ensemble d\'entraînement) national : ' + str(s))
        st.write('Cross-val score national : ' + str(cvs))
        st.write('Score de la Linear Regression (ensemble de test) national : ' + str(stest))

        predict(dfnat, hypo_rech, model)
        
    st.header("Série temporelle")
    import streamlit as st
    from PIL import Image
    import numpy as np

    st.write ('**Consommation**')
    image = Image.open('image7.png')

    st.image(image, caption="consommation de l'éléctricité en France")


    st.write ('**Premier model**')
    image = Image.open('image6.png')

    st.image(image, caption="premier model")


    st.write('**Selection de model par rapport aux combinaison possible**')
    image1 = Image.open('image5.png')

    st.image(image1, caption='selection du model qui minimise BIC et AIC')

    st.write('**Model optimal**')
    image2 = Image.open('image3.png')

    st.image(image2, caption='model optimal minimisant le critere BIC et AIC')

    st.write ("**Prevision pour l'annee 2022**")
    image8 = Image.open('image1.png')
    st.image(image8, caption='Prevision 2022')

    st.write ("**Justification des erreurs**")
    image3 = Image.open('image4.png')

    st.image(image3, caption='repartition et normalisation des erreurs')

 

