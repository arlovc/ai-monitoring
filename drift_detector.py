import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from abc import ABC, abstractmethod
from typing import Tuple
from IPython import embed

from scipy.stats import ks_2samp
from scipy.stats import chi2_contingency 
from scipy.spatial import distance

#   ____                    ____ _               
#  | __ )  __ _ ___  ___   / ___| | __ _ ___ ___ 
#  |  _ \ / _` / __|/ _ \ | |   | |/ _` / __/ __|
#  | |_) | (_| \__ \  __/ | |___| | (_| \__ \__ \
#  |____/ \__,_|___/\___|  \____|_|\__,_|___/___/
                                              

class DriftDetector(ABC):
    """Abstract base class for Drift Detectors.
    Parameters:
        drift_dataset: a pd.DataFrame, recommended from a DriftDataset
    """
    def __init__(self, data: pd.DataFrame):
        self.data = self.measure(data)
    
    @abstractmethod
    def name(self) -> str:
        """Name of the detector. This is used when displaying detector results"""
        pass
    
    @abstractmethod
    def description(self) -> str:
        """Description of the detector. This is folded into an expander when displaying detector results.
        This string can contain Markdown styling.
        """
        pass
    
    @abstractmethod
    def measure(self, data: pd.DataFrame):
        """Calculates drift measurements. This is called in __init__ and returns to self.data"""
        pass
    
    @abstractmethod
    def plot(self, feature: str, vline_date: str, date_range: Tuple[str, str]) -> go.Figure:
        """Plots results of drift measurement for the given feature. 
        This should be based on self.data (Hint: use self._copy_data())
        
        NOTE: MonitoringDashboard will always pass the declared variables. 
        If the specific implementation of DriftDetector does not use them, they 
        should still be declared, but simply ignored in the body of 
        the plot() function.
        
        Parameters:
            feature: The feature for which drift should be plotted.
            dataset_date: The date for which the vertical line should be plotted.
            date_range: demarks the start and end of the range for which the data
                        is to be plotted.
        """
        pass
    
    # Render functions
    def render(self, *args) -> st.container:
        """Test functie of ik een st.container terug kan keren met de plot erin"""
        container = st.container()
        container.plotly_chart(self.plot(*args))
        
    # Helper functions
    def _copy_data(self):
        """Returns a copy of self.data"""
        return self.data.copy()
        
    def _add_vline(self, fig: go.Figure, vline_date: str, color: str='Midnightblue') -> go.Figure:  # TODO basiskleur aanpassen naar een RIjksoverheidskleur
        """Adds a vertical line at the given dataset_date to a ploty Figure.
        
        Parameters:
            fig: the plotly Figure on which the vline is to be added
            vline_date: the date at which the line is to be added
            color (Optional): the color of the vline
        """
        fig.add_vline(x=vline_date, line_width=2, line_dash='dash', line_color=color)
        return fig
        
    def _filter_dates(self, data: pd.DataFrame, date_range: Tuple[str, str]) -> pd.DataFrame:
        """Filters a dataset based on a given date range (expected to be a tuple)
        
        Parameters:
            data: the dataset which is to be filtered
            date_range: demarks the start and end of the range for which the data
                        is to be plotted.
        """
        start_date, end_date = date_range
        mask = (data.dataset_date >= start_date) & (data.dataset_date <= end_date)
        return data[mask] 
    
    def _default_layout(self, fig: go.Figure) -> go.Figure:
        fig.update_layout(
            width=800,
            height=450,
            plot_bgcolor="rgba(0, 0, 0, 0)",
            yaxis={'tickformat':'6.3r'}
        )
        return fig
           

#   ___                 _                           _        _   _                 
#  |_ _|_ __ ___  _ __ | | ___ _ __ ___   ___ _ __ | |_ __ _| |_(_) ___  _ __  ___ 
#   | || '_ ` _ \| '_ \| |/ _ \ '_ ` _ \ / _ \ '_ \| __/ _` | __| |/ _ \| '_ \/ __|
#   | || | | | | | |_) | |  __/ | | | | |  __/ | | | || (_| | |_| | (_) | | | \__ \
#  |___|_| |_| |_| .__/|_|\___|_| |_| |_|\___|_| |_|\__\__,_|\__|_|\___/|_| |_|___/
#                |_|                                                              

class MeanStdDetector(DriftDetector):
    """Drift detector based on mean and average of each variable in each dataset"""
    
    def name(self) -> str:
        return """Mean and Standard Deviation"""
    
    def description(self) -> str:
        desc = """Calculates the mean and standard deviation of all numeric columns.
        The mean and standard deviation are shown as a line and dashed lines respectively.
        """
        return desc
    
    def measure(self, data: pd.DataFrame):
        """Calculate mean and standard deviation for all numerical columns."""
        data = data.copy()
        data = data.groupby(['dataset_index', 'dataset_date']).agg(['mean', 'std'])   # this results in a dataframe with MultiIndex, which plotly doesn't understand. 
        data.columns = ['_'.join(pair) for pair in data.columns] # resetting the columns
        data = data.reset_index() 

        return data
    
    # TODO kleuren fixen naar Rijksoverheidskleuren
    def plot(self, feature: str, vline_date: str=None, date_range: Tuple[str, str]=None) -> go.Figure:
        """Plots the mean and standard deviations. 
        Returns:
            fig -- a plotly Figure
        """
        df = self._copy_data()
          
        # reading out mean and standard deviation for training set
        mean = df[df.dataset_index == 0][f'{feature}_mean'].values
        std = df[df.dataset_index == 0][f'{feature}_std'].values
        
        if date_range:
            df = self._filter_dates(df, date_range) 

        color = 'firebrick'     # TODO Rijksoverheidskleur toevoegen                            

        fig = go.Figure()
        # adding mean trace with error bars
        fig.add_trace(go.Scatter(
            x=df['dataset_date'],
            y=df[f'{feature}_mean'],
            mode='markers',
            name='Mean',
            line=dict(color=color),
            error_y=dict(type='data',
                         array=df[f'{feature}_std'],
                         visible=True,
                         color=color,
                         thickness=1)
        ))  
        # adding lines for training mean, mean + std and mean - std
        fig.add_hline(float(mean), line_width=2, line_color='black', 
                      annotation_text='Training mean', annotation_position='top left')
        fig.add_hline(float(mean + std), line_width=1, line_color='black', line_dash='dash', 
                      annotation_text='Training mean + std', annotation_position='top left')
        fig.add_hline(float(mean - std), line_width=1, line_color='black', line_dash='dash', 
                      annotation_text='Training mean - std', annotation_position='bottom left')
        
        # adding vline
        if vline_date:
            fig = self._add_vline(fig, vline_date)
        
        # adding titles and setting background color to transparent
        fig.update_layout(
            title=feature,
            xaxis_title="Dataset date",
        )
        fig = self._default_layout(fig)
        
        return fig
    
# NOOT: We doen nu absolute aantallen voor KS, is dat wenselijk? Gaat dan niet alles wat meer data heeft automatisch afwijken van de originele trainingsdata?
class KSDetector(DriftDetector):
    """Drift detector based on Kolmogorov-Smirnov test"""
    # overriding __init__ since I need another value in my constructor, passing back to original with super().__init__()
    def __init__(self, data: pd.DataFrame, pvalue: bool=False):
        """Parameters:
            pvalue (optional): whether to display the pvalue, otherwise the test statistic
        """
        self.pvalue = pvalue  # registring self.pvalue before calling constructor of superclass as it is used in self.measure() (which is called by superclass)
        super().__init__(data)
    
    def name(self) -> str:
        stat = "pvalue" if self.pvalue else "test statistic"

        return f"Kolmogorov-Smirnov ({stat})"
    
    def description(self) -> str:
        value_string = "p-value" if self.pvalue else "test statistic"
        
        desc = f"""Calculates the Kolmogorov-Smirnov test statistic of all numeric columns.
        The two-sample Kolmogorov-Smirnov test finds the largest vertical distance between the empirical cdfs for two samples.
        Unusually large distances indicate that the two samples are not consistent with having come from the same distribution.
        This test is nonparametric in the sense that the distribution of the test statistic under the null doesn't depend on which specific distribution was specified under the null.
        
        The {value_string} is shown as a line.
        """
        return desc
    
    def measure(self, data: pd.DataFrame):
        """Calculate per feature the KS test between train and sample test for all numerical columns."""
        data = data.copy()

        # define reference set
        refset = data.loc[data['dataset_index']==0] # ervan uitgaande dat dit ook generiek genoeg is?
        # custom function for KS test
        def kstest(x,y):
            features = list(x.select_dtypes('number').columns)
            features.remove('dataset_index') # generiek genoeg? wel nodig, want anders blijft dataset_index ondanks de groupby er alsnog tussen staan, en dan werkt reset_index() niet
            result = {feature : ks_2samp(x[feature],y[feature]).pvalue if self.pvalue else ks_2samp(x[feature], y[feature]).statistic for feature in features}
            return pd.Series(result)
        data = data.groupby(['dataset_index', 'dataset_date']).apply(kstest, y=refset) # apply met extra argument (de refset) ipv agg did the trick
        data = data.reset_index()
        
        return data
    
    # TODO kleuren fixen naar Rijksoverheidskleuren
    def plot(self, feature: str, vline_date: str=None, date_range: Tuple[str, str]=None) -> go.Figure:
        """Plots the mean and standard deviations. 
        Returns:
            fig -- a plotly Figure
        """
        df = self._copy_data()

        if date_range:
            df = self._filter_dates(df, date_range) 

        color = 'firebrick'     # TODO Rijksoverheidskleur toevoegen                            
        
        stat = "pvalue" if self.pvalue else "statistic"

        # adding trace of the desired statistic
        fig = px.line(df, x="dataset_date", y=f"{feature}", title=feature, markers=True)
        fig.update_traces(line_color=color) # sets the color of the line
 
        # adding vline
        if vline_date:
            fig = self._add_vline(fig, vline_date)
        
        # adding titles and setting background color to transparent
        fig.update_layout(
            xaxis_title="Dataset date"
        )
        fig = self._default_layout(fig)
        # y-axis range aanpassen om grafiek beter te interpreteren
        fig.update_yaxes(range=[df[feature].mean()-(4*df[feature].std()), df[feature].mean()+(4*df[feature].std())]) # y-scale rondom gemiddelde +/- 4 std
        if stat=="pvalue":
            maxp = 0.1 if df[feature].max() < 0.1 else df[feature].max() # range p-values tussen 0.1 tenzij hoger dan 0.1
            fig.update_yaxes(range=[-0.01,maxp])
        return fig

# TODO: Jensen-Shannon detector klopt niet, dus laten kloppen of verwijderen --> update: verwijderd en hier commit van gemaakt; dus altijd terug te halen :)

class Chi2Detector(DriftDetector):
    """Drift detector based on Chi-square test of independence"""
        # overriding __init__ since I need another value in my constructor, passing back to original with super().__init__()
    def __init__(self, data: pd.DataFrame, pvalue: bool=False):
        """Parameters:
            pvalue (optional): whether to display the pvalue, otherwise the test statistic
        """
        self.pvalue = pvalue  # registring self.pvalue before calling constructor of superclass as it is used in self.measure() (which is called by superclass)
        super().__init__(data)
    
    def name(self) -> str:
        stat = "pvalue" if self.pvalue else "test statistic"
        return f"Chi-square test ({stat})"
    
    def description(self) -> str:
        value_string = "p-value" if self.pvalue else "test statistic"
        
        desc = f"""Calculates the Chi-square test statistic of all categorical columns.
        The two-sample Chi-square test is a non-parametric (distribution-free) method used to compare the relationship between the two categorical (nominal) variables in a contingency table.
        The alternative hypothesis is that the two variables are related. So if we use it to test for drift within one variable, comparing two samples at different time points, a high test
        statistic means less drift. With the other detectors for numerical features, we test the alternative hypothesis that there is a difference (i.e. drift).
        
        The Chi-square {value_string} is shown as a line.
        """
        return desc
    
    def measure(self, data: pd.DataFrame):
        """Calculate per feature the Chi-square test between contingency table of train and sample for all categorical columns."""
        data = data.copy()

        # define reference set
        refset = data.loc[data['dataset_index']==0] # ervan uitgaande dat dit ook generiek genoeg is?
        # custom function for KS test       
        def chisq(x,y):
            features = list(x.select_dtypes(exclude='number').columns)
            features.remove('dataset_markering')
            features.remove('dataset_date')
            df = pd.concat([x,y],axis=0)
            
            pvalue = self.pvalue == True # chi2_contingency heeft op 0 statistic en op 1 pvalue. Dit wordt 1 als self.pvalue True is, anders 0
            result = { fi : chi2_contingency(pd.crosstab(df.dataset_markering,df[fi]).to_numpy())[pvalue] for fi in features }
            return pd.Series(result) 
        
        data = data.groupby(['dataset_index', 'dataset_date']).apply(chisq, y=refset) # apply met extra argument (de refset) ipv agg did the trick
        data = data.reset_index()
        
        return data
    
    # TODO kleuren fixen naar Rijksoverheidskleuren
    def plot(self, feature: str, vline_date: str=None, date_range: Tuple[str, str]=None) -> go.Figure:
        """Plots the mean and standard deviations. 
        Returns:
            fig -- a plotly Figure
        """
        df = self._copy_data()

        if date_range:
            df = self._filter_dates(df, date_range) 

        color = 'firebrick'     # TODO Rijksoverheidskleur toevoegen                            
        
        stat = "pvalue" if self.pvalue else "statistic"

        # adding trace of the desired statistic
        fig = px.line(df, x="dataset_date", y=f"{feature}", title=feature, markers=True)
        fig.update_traces(line_color=color) # sets the color of the line
 
        # adding vline
        if vline_date:
            fig = self._add_vline(fig, vline_date)
        
        # adding titles and setting background color to transparent
        fig.update_layout(
            xaxis_title="Dataset date"
        )
        fig = self._default_layout(fig)
        
        # testen of dit ook werkt voor chi square
        fig.update_yaxes(range=[df[feature].mean()-(4*df[feature].std()), df[feature].mean()+(4*df[feature].std())]) # y-scale rondom gemiddelde +/- 4 std
        if stat=="pvalue":
            maxp = 0.1 if df[feature].max() < 0.1 else df[feature].max() # range p-values tussen 0.1 tenzij hoger dan 0.1
            fig.update_yaxes(range=[-0.01,maxp])

        return fig

class DistributionDetector(DriftDetector):
    """Plots a histogram of a selected dataset"""
    
    def name(self) -> str:
        return "Distribution"
    
    def description(self) -> str:
        desc = """Distribution of selected dataset. The vertical axis shows the proportion of the number of datapoints in this bin compared to the total data."""
        return desc
    
    def measure(self, data: pd.DataFrame):
        """Returns data unchanged"""
        return data
    
    def plot(self, feature: str, vline_date: str=None, date_range: Tuple[str, str]=None) -> go.Figure:
        """Plots a histogram of the dataset selected with vline_date.
        Returns:
            fig -- a plotly figure
        """
        df = self._copy_data()
        
        # filter on train (index = 0) and the selected date from the vline
        df = df[(df.dataset_index == 0) | (df.dataset_date == vline_date)]
        
        # Histogram with violin marginals
        fig = px.histogram(df, x=feature, color='dataset_date', marginal='violin', barmode='group', histnorm='probability') # TODO Rijksoverheidskleuren
        
        # Adding titles and setting background color to transparent
        fig.update_layout(
            title=f'Distribution of training data and {feature} on {vline_date}',
            xaxis_title=feature,
        )
        
        fig = self._default_layout(fig)
        
        return fig
        

class ValueCountsDetector(DriftDetector):
    """Plots the number of datapoints of a selected featurte"""
    
    def name(self) -> str:
        return "Number of data points"
    
    def description(self) -> str:
        desc = """The number of datapoints of a selected feature, plotted over time. """
        return desc
    
    def measure(self, data: pd.DataFrame):
        """Returns grouped dataframe with value counts per feature and time point"""
        data = data.copy()
        data = data.groupby(['dataset_index', 'dataset_date']).count()   # this results in a dataframe with MultiIndex, which plotly doesn't understand. 
        data = data.reset_index() 
        return data
    
    # TODO kleuren fixen naar Rijksoverheidskleuren
    def plot(self, feature: str, vline_date: str=None, date_range: Tuple[str, str]=None) -> go.Figure:
        """Plots the number of datapoints at each time point. 
        Returns:
            fig -- a plotly Figure
        """
        df = self._copy_data()

        if date_range:
            df = self._filter_dates(df, date_range) 

        color = 'firebrick'     # TODO Rijksoverheidskleur toevoegen                            

        fig = go.Figure()
        # adding trace for value count
        fig.add_trace(go.Scatter(
            x=df['dataset_date'],
            y=df[f'{feature}'],
            mode='lines+markers',
            name='Value count',
            line=dict(color=color)
        ))  
        
        # adding vline
        if vline_date:
            fig = self._add_vline(fig, vline_date)
        
        # adding titles and setting background color to transparent
        fig.update_layout(
            title=feature,
            xaxis_title="Dataset date")
        
        fig = self._default_layout(fig)
        

        return fig
        
        
class GiniDetector(DriftDetector):
    """Feature selector based on top-10 Gini values"""
    
    def name(self) -> str:
        return """Gini feature importance"""
    
    def description(self) -> str:
        desc = """
        Predictive features of the model are ranked based on the Gini values of feature importance during training. The dashboard picks the top-10 for monitoring.
        """
        return desc
    
    def measure(self, data: pd.DataFrame):
        # Joram: Nu pakken we standaard de eerste 10; wordt al in prepareData.py gedaan; alternatief is om dat hier als methode te doen; waarbij je zelf top-x pakt
        # Luc: Dat zou inderdaad gebruiksvriendelijker zijn! Alleen nog de vraag hoe we dan die state gaan opslaan...
        """Returns data unchanged"""
        return data
    
    # Noot: ik accepteer hier [feature], [vline_date] en [date_range] args, maar dat betekent niet dat ik daar iets mee moet doen :)
    def plot(self, feature: str, vline_date: str=None, date_range: Tuple[str, str]=None) -> go.Figure:
        """Plots a bar graph with feature importances (Gini)"""
        data = self._copy_data()
        
        fig = px.bar(
            data, x="MeanDecreaseGini", y="Variable", 
            title="Ranked Gini values",
            color='plot_color',   # if values in column z = 'some_group' and 'some_other_group'
            # TODO deze namen zijn niet handig voor generiek, iets aan doen
            color_discrete_map={
                'omitted': 'lightslategray',
                'selected': 'darkgreen'
            }
        )
        
        # adding titles and setting background color to transparent
        fig.update_layout(
            title=self.name(),
            width=800,
            height=1000,
            plot_bgcolor="rgba(0, 0, 0, 0)"
        )
        return fig

            

