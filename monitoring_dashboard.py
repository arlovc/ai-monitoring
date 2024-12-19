import math
import pickle
import streamlit as st
from pathlib import Path
from typing import List, Tuple

from drift_detector import DriftDetector

class LoopIterator:
    """Utility class that allows looping over its elements indefinitely"""
    def __init__(self, elements):
        self.elements = elements
        self.iter = iter(self.elements)
    
    def __next__(self):
        try:
            return next(self.iter)
        except StopIteration:  # reset the iterator if the iterator is exhausted
            self.iter = iter(self.elements)  
            return next(self.iter)   



class MonitoringDashboard:
    """Creates a drift monitoring dashboard.
    
    Pages are created from a dictionary.
    This has the form:
    {
    'Page title': page_dict
    }
        where the page_dict is another dictionary, of the form:
        {
         'description': str with description of the page
         'columns': [list of columns for the selector]
         'date_range': [list of dates appearing in the drift dataset (excluding train data)]
         'display_columns': int with number of columns to display DriftDetectors in
         'detectors': [list of initialized instances of DriftDetector]
         }
    This information is used to automatically generate the drift monitoring dashboard.
    NOTE: The if the key is not available in the page_dict, the element is omitted.
    
    Parameters:
        dashboard_dict_path (Optional): Path to a pickle file containing a dashboard_dict in the above format.
                                        defaults to 'dashboard_dict.pickle', assumed to be in the main folder.
        title (Optional): title of the dashboard.
        num_cols (Optional): the number of columns to render drift detectors in.
    """
    def __init__(self, 
                 dashboard_dict_path: Path=Path('dashboard_dict.pickle'),
                 title: str='Data Drift Monitoring', 
                 num_cols: int=2):
        self.dashboard_dict_path = dashboard_dict_path
        self.title=title
        self.default_columns = num_cols

    @st.cache
    def load(self):
        """Loads the data"""
        with open(self.dashboard_dict_path, 'rb') as file:
            return pickle.load(file)   # running this in the constructor seems to avoid having to add a load function with streamlit @st.cache weirdness
        
        
    def _page_select(self, dashboard_dict: dict):
        """Creates the page select in the sidebar"""
        pages = [key for key in dashboard_dict.keys()]  # reading out the titles from the dashboard_dict
        page = st.sidebar.radio(
            'Select page',
            pages
        )
        
        return page
    
    def _render_detector(self, detector: DriftDetector, column, *args):
        """Renders a detector for a given feature and set of filters"""
        with column:
            with st.expander(detector.name()):     # expander with name and reveals description on expansion
                st.write(detector.description())
            detector.render(*args)
#     def _render_selector(self, selector: FeatureSelector, *args):
#         selector=selector[0] # het is er nu maar eentje, daarom niet apart nog in een nieuwe _create_grid; maar dan blijft het een list, dus hier 'uitpakken'; beetje lelijk, kan beter en generieker
#         """Renders feature selection plot at start page"""
#         with st.expander(selector.name()):     # expander with name and reveals description on expansion
#             st.write(selector.description())
#         selector.render(*args)

            
    def _create_grid(self, 
                     detectors: List[DriftDetector], 
                     display_cols: int,
                     *args):
        """Creates the grid of drift detectors for the given feature.
        Detectors are rendered left to right over [display_cols] columns        
        """
        columns = st.columns(display_cols)  # creating columns based on [display_cols]
        iter_cols = LoopIterator(columns)   # LoopIterator so I can keep looping over the columns
        pairs = [(detector, next(iter_cols)) for detector in detectors]  # pairing detectors with columns from left to right
        for detector, column in pairs: 
            self._render_detector(detector, column, *args)
        
        
    def _make_page(self, page: str, dashboard_dict: dict):
        """Generates a page from the dashboard_dict"""
        page_dict = dashboard_dict[page] 
        
        ## sidebar settings
        
        # feature selector in sidebar
#         if 'columns' in page_dict.keys():
        cols = 'columns' in page_dict.keys()
        feature = st.sidebar.selectbox('Select model feature', page_dict['columns']) if cols else None       
        
        # selecting dataset and filtering on range
        if 'date_range' in page_dict.keys():
            drange = page_dict['date_range']
            vline_date = st.sidebar.select_slider('Selecteer sample set', drange, value=drange[-1])  # slider for selecting time point  #TODO: possible to select outside of the filter period below this way, needs fix?
            date_range = st.sidebar.select_slider('Filter periode', drange, value=(drange[0], drange[-1]))   # slider for selecting period
        else: 
            vline_date, date_range = None, None
        
        # Header and description of the page 
        st.subheader(page)
        if 'description' in page_dict.keys():
            st.write(page_dict['description'])
        
        # Reading out how many display columns the user wants to create, else default to self.default_columns
        display_columns = page_dict['display_columns'] if 'display_columns' in page_dict.keys() else self.default_columns
            
        # creating grid of detectors
        if 'detectors' in page_dict.keys():
            self._create_grid(page_dict['detectors'], display_columns, feature, vline_date, date_range)
        if 'selectors' in page_dict.keys():
            self._render_selector(page_dict['selectors'])
        
        
    def run(self):
        """Runs the streamlit app"""        
        # Use the full page instead of a narrow central column
        st.set_page_config(layout="wide", page_title=self.title)
        
        dashboard_dict = self.load()
        st.title(self.title)
        page = self._page_select(dashboard_dict)
        self._make_page(page, dashboard_dict)
        
        
        
        