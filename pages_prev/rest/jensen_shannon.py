# packages
from scipy.spatial import distance
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from IPython import embed as dbstop

def app():
    
    ds = pd.read_csv("df_for_dash.csv")
    df = pd.concat([ds.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']),ds[['datT','dataset_date']]],axis=1)
    
    ##### JENSEN-SHANNON DIVERGENCE FUNCTION #####

    def jsdist(df1,df2):
        if len(df1)>len(df2):
            np.random.shuffle(df1)
            df1 = df1[0:len(df2)]
        elif len(df1)<len(df2):
            np.random.shuffle(df2)
            df2 = df2[0:len(df1)]
        return distance.jensenshannon(df1, df2) ** 2

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
    
    jsd = []
    for tx,ti in enumerate(tix):
        for fi in feats:
            # Jensen-Shannon divergence
            JSD = jsdist(df.loc[df['datT']=='dev'][fi].to_numpy(), df.loc[df['datT']==ti][fi].to_numpy())
            jsd.append(
                {
                    'Time': tid[tx],
                    'Feature': fi,
                    'JSD': JSD
                }
            )

    jsd = pd.DataFrame(jsd).pivot(index = 'Time', columns = 'Feature', values = 'JSD')
    jsd['Time'] = jsd.index
    jsd['Time'] = pd.to_datetime(jsd['Time'])

    ## Plotten
    
    st.markdown("### Jensen Shannon Divergence")

    # Feature dropdown list
    fx = st.sidebar.selectbox('Select model feature', feats)

    # plot the value
    fig = px.line(jsd, x="Time", y=fx, title=fx)
    fig.update_yaxes(range=[jsd[fx].mean()-(3*jsd[fx].std()), jsd[fx].mean()+(3*jsd[fx].std())])
    fig.update_traces(mode='markers+lines')
    st.plotly_chart(fig)