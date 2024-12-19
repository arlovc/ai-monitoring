# packages
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import pickle
from IPython import embed as dbstop
from pygit2 import Repository

def app():
    repo = Repository('/data/joramvandriel/ai-monitoring-streamlit-dashboard/')
    # Getting back the objects:
    with open('objs_' + repo.head.shorthand + '_branch.pkl','rb') as f:  # Python 3: open(..., 'rb')
        dF = pickle.load(f)
        gini = pickle.load(f)
        drift = pickle.load(f)
        F1 = pickle.load(f)
        outl = pickle.load(f)
        tix = pickle.load(f)
        tid = pickle.load(f)
        feats = pickle.load(f)
        cats = pickle.load(f)
        
    # selecteer categoriale features
    dc = dF.select_dtypes(include=['object'])

    ## Plotten

    # Feature dropdown list
    cx = st.sidebar.selectbox('Select model feature', cats)
    # Tijdspunt slider
    sx = st.sidebar.select_slider('Selecteer sample set', tid, value=tid[len(tid)-1])
    # Tijdspunt filter
    tf = st.sidebar.select_slider('Filter periode', tid, value=(tid[2],tid[len(tid)-1]))

    # filter data op tijdrange
    F1_ = F1[(F1['Time']>=tf[0]) & (F1['Time']<=tf[1])]

    # functie voor plot
    def plotly_drift(metric,cx,sx):
        fig = px.line(F1_[F1_['Feature']==cx], x="Time", y=metric)
        fig.update_yaxes(range=[F1_[F1_['Feature']==cx][metric].mean()-(4*F1_[F1_['Feature']==cx][metric].std()), F1_[F1_['Feature']==cx][metric].mean()+(4*F1_[F1_['Feature']==cx][metric].std())])
        fig.update_traces(mode='markers+lines')
        fig.add_vline(x=sx, line_width=2, line_dash="dash", line_color="red")
        return fig
        
    # column layout
    col1, col2 = st.columns(2)
    
    # Drift plot over tijd
    col1.markdown("### F1-score Random Forest")
    col1.plotly_chart(plotly_drift("F1",cx,sx))
 
    # Distribution plot per tijdspunt
    col2.markdown("### Distribution plots")
    fig = px.histogram(dc.loc[(dc['datT']=='dev') | (dc['datT']==tix[tid.index(sx)])], x=cx, color="datT", marginal="violin")
    col2.plotly_chart(fig)
