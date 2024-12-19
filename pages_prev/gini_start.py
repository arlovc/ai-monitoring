import pandas as pd
import streamlit as st
import plotly.express as px
from IPython import embed as dbstop

def app():
    
    gini = pd.read_csv("gini_for_dash.csv")
    gini = gini.sort_values('MeanDecreaseGini')
    ## Plotten
    
    st.markdown("### Top-10 Gini values")
    st.markdown("In dit dashboard wordt data drift berekend tussen distributies van de out of sample datasets van het model in productie, en de originele implementatie dataset waarop het model is getrained.")
    st.markdown("Alleen de features die het meest bijdragen aan de performance van het model worden getest. Hiervoor wordt de top-10 features gekozen met de hoogste Gini waarde.")
    
    fig = px.bar(
        gini, x="MeanDecreaseGini", y="Variable", 
        title="Ranked Gini values",
        width=800, height=1000,
        color='show',   # if values in column z = 'some_group' and 'some_other_group'
        color_discrete_map={
            'Overig': 'lightslategray',
            'Drift test': 'darkgreen'
        }
    )
    st.plotly_chart(fig)
