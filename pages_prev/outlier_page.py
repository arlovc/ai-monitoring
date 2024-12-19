# packages
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import pickle
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
      
    # selecteer numerieke features
    df = pd.concat([dF.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']),dF[['datT','dataset_date']]],axis=1)

    ## Plotten

    # Feature dropdown list
    fx = st.sidebar.selectbox('Select model feature', feats)
    # Tijdspunt slider
    sx = st.sidebar.select_slider('Selecteer out of sample set', tid, value=tid[len(tid)-1])
    # Tijdspunt filter
    tf = st.sidebar.select_slider('Filter periode', tid, value=(tid[2],tid[len(tid)-1]))
    
    # filter data op tijdrange
    outl_ = outl[(outl['Time']>=tf[0]) & (outl['Time']<=tf[1])]

    # functie voor plot
    def plotly_outl(metric,fx,sx):
        fig = px.line(outl_[outl_['Feature']==fx], x="Time", y=metric)
        fig.update_yaxes(range=[outl_[outl_['Feature']==fx][metric].mean()-(4*outl_[outl_['Feature']==fx][metric].std()), outl_[outl_['Feature']==fx][metric].mean()+(4*outl_[outl_['Feature']==fx][metric].std())])
        fig.update_traces(mode='markers+lines')
        fig.add_vline(x=sx, line_width=2, line_dash="dash", line_color="red")
        return fig
    
    # dbstop()
    # column layout
    col1, col2 = st.columns(2)
    
    # Outlier percentage plot over tijd
    col1.markdown("### Percentage detected outliers")
    col1.plotly_chart(plotly_outl("poutliers",fx,sx))
 
    # Distribution plot per tijdspunt
    col2.markdown("### Distribution plots")
    fig = px.histogram(df.loc[(df['datT']=='dev') | (df['datT']==tix[tid.index(sx)])], x=fx, color="datT", marginal="violin")
    col2.plotly_chart(fig)
