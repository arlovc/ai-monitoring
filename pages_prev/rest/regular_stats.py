# packages

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from IPython import embed as dbstop

def app():
    
    ds = pd.read_csv("df_for_dash.csv")
    df = pd.concat([ds.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']),ds[['datT','dataset_date']]],axis=1)

    ## Get feature names
    
    feats = [col for col in df]
    feats = feats[0:len(feats)-2]    
    
    ## Tijdspunten out of sample
    
    tix = list(df.datT.unique())
    tix.remove('dev')
    tix = [x for x in tix if str(x) != 'nan'] # --> tijdslabels
    tid = list(df.dataset_date.unique())
    tid = [x for x in tid if str(x) != 'nan'] # --> datum

    ## Drift maat berekenen
    
    avg = []
    for tx,ti in enumerate(tix):
        for fi in feats:
            # Difference in mean
            AVG = df.loc[df['datT']=='dev'][fi].mean() - df.loc[df['datT']==ti][fi].mean()
            avg.append(
                {
                    'Time': tid[tx],
                    'Feature': fi,
                    'AVG': AVG
                }
            )

    avg = pd.DataFrame(avg).pivot(index = 'Time', columns = 'Feature', values = 'AVG')
    avg['Time'] = avg.index
    avg['Time'] = pd.to_datetime(avg['Time'])

    ## Plotten
    
    # Feature dropdown list
    fx = st.sidebar.selectbox('Select model feature', feats)
    # Tijdspunt slider
    sx = st.sidebar.select_slider('Selecteer out of sample set', tid, value=tid[len(tid)-1])
    
    # column layout
    col1, col2 = st.columns(2)
    
    # Drift plot over tijd
    col1.markdown("### Difference in Mean")
    fig = px.line(avg, x="Time", y=fx, title=fx)
    fig.update_yaxes(range=[avg[fx].mean()-(3*avg[fx].std()), avg[fx].mean()+(3*avg[fx].std())])
    fig.update_traces(mode='markers+lines')
    fig.add_vline(x=sx, line_width=2, line_dash="dash", line_color="red")
    col1.plotly_chart(fig)

    # Distribution plot per tijdspunt
    col2.markdown("### Distribution plots")
    fig = px.histogram(df.loc[(df['datT']=='dev') | (df['datT']==tix[tid.index(sx)])], x=fx, color="datT", marginal="violin")
    col2.plotly_chart(fig)
