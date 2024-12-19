# packages
from scipy.stats import ks_2samp
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

    kst = []
    for tx,ti in enumerate(tix):
        for fi in feats:
            # Kolmogorov-Smirnov test
            KS = ks_2samp(df.loc[df['datT']=='dev'][fi].to_numpy(), df.loc[df['datT']==ti][fi].to_numpy())
            kst.append(
                {
                    'Time': tid[tx],
                    'Feature': fi,
                    'KS': KS.statistic
                }
            )

    kst = pd.DataFrame(kst).pivot(index = 'Time', columns = 'Feature', values = 'KS')
    kst['Time'] = kst.index
    kst['Time'] = pd.to_datetime(kst['Time'])

    ## Plotten

    st.markdown("### Kolmogorov-Smirnov test statistic")
    
    # Feature dropdown list
    fx = st.sidebar.selectbox('Select model feature', feats)

    # plot the value
    fig = px.line(kst, x="Time", y=fx, title=fx)
    fig.update_yaxes(range=[kst[fx].mean()-(3*kst[fx].std()), kst[fx].mean()+(3*kst[fx].std())])
    fig.update_traces(mode='markers+lines')
    st.plotly_chart(fig)