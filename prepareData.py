import pickle
import pandas as pd
from functools import partial
from pathlib import Path

from drift_dataset import DriftDataset
from drift_detector import MeanStdDetector, DistributionDetector, KSDetector, ValueCountsDetector, GiniDetector, Chi2Detector

binnenvaart_pad = Path.home()/'share'/'Binnenvaart'/'Monitoring'

#   ____        _          _       _                    
#  |  _ \  __ _| |_ __ _  (_)_ __ | | ___ _______ _ __  
#  | | | |/ _` | __/ _` | | | '_ \| |/ _ \_  / _ \ '_ \ 
#  | |_| | (_| | || (_| | | | | | | |  __// /  __/ | | |
#  |____/ \__,_|\__\__,_| |_|_| |_|_|\___/___\___|_| |_|

# postprocessing functie
def mark_datasets(df: pd.DataFrame):
    """Post processing function to add markings for labelled and unlabelled samples in each dataset"""
    
    def mark_labels(df: pd.DataFrame):
        """To be applied over groups, marking the datasets"""
        df = df.copy()
        [idx] = df.dataset_index.unique()
        if idx == 0:
            train_label = 'train'
            oos_label = 'oos'
        else:
            train_label = f'labelled_{idx}'
            oos_label = f'unlabelled_{idx}'

        begindatum = max(df.Begindatum)
        mask = df.Begindatum < begindatum
        df['dataset_markering'] = None
        df.loc[mask, 'dataset_markering'] = train_label
        df.loc[~mask, 'dataset_markering'] = oos_label
        
        return df
    
    df = df.copy()
    df = df.groupby(['dataset_index']).apply(mark_labels)
    df = df.reset_index(drop=True)
    
    return df

# features uit top 10 Gini halen
def main():
    print('Processing Gini....')
    gini = pd.read_csv(binnenvaart_pad/'meta'/'meanDecreaseGini.csv')
    features = gini.Variable[:10].tolist() + ['Begindatum', 'dataset_markering'] # laatste wordt toegevoegd door mark_datasets()

    # Gini waarden voor landing page
    gini['plot_color'] = 'omitted'
    gini.loc[0:10, "plot_color"] = "selected"
    gini = gini.sort_values('MeanDecreaseGini')  # sorteren voor plot (dit gaat van onder naar boven)

    print('Post processing data...')
    dd = DriftDataset(binnenvaart_pad/'data/feature_data', post_process=mark_datasets)
    dd.keep_columns(features) 

    # AI Binnenvaart specifiek, voorspellingsdata vergelijken alleen met de training data, niet met de out of sample op moment van de training.
    # Daarom hierboven de postprocessing functie aangemaakt, daar kan ik nu dus op filteren:

    print('Creating datasets...')
    data = dd.dataset

    # uitlezen gevallen waar de markering 'train' is of begint met 'labelled'
    train_mask = data.dataset_markering == 'train'
    labelled_mask = data.dataset_markering.str.contains('^labelled')
    mask = train_mask | labelled_mask # samenstellen mask
    labelled_data = data[mask] # subsamplen van labelled data

    # uitlezen gevallen waar het om out of sample gaat ('oos' of 'unlabelled')
    oos_mask = data.dataset_markering == 'oos'   # dataset maken met alleen out of sample data
    unlabelled = data.dataset_markering.str.contains('^unlabelled')  # out of sample data in de drift datasets begint met 'unlabelled'
    mask = oos_mask | unlabelled  # samenstellen mask
    oos = data[mask] # subsamplen van out of sample data

    # uitlezen target predictions
    # dt = DriftDataset(binnenvaart_pad/'data/target_predictions', post_process=mark_datasets)
    dt = DriftDataset(binnenvaart_pad/'data/target_predictions')
    dt.keep_columns(['predRF']) 
    targets = dt.dataset

    #   ____       _  __ _     ____       _            _                                                     _              
    #  |  _ \ _ __(_)/ _| |_  |  _ \  ___| |_ ___  ___| |_ ___  _ __ ___    __ _  __ _ _ __  _ __ ___   __ _| | _____ _ __  
    #  | | | | '__| | |_| __| | | | |/ _ \ __/ _ \/ __| __/ _ \| '__/ __|  / _` |/ _` | '_ \| '_ ` _ \ / _` | |/ / _ \ '_ \ 
    #  | |_| | |  | |  _| |_  | |_| |  __/ ||  __/ (__| || (_) | |  \__ \ | (_| | (_| | | | | | | | | | (_| |   <  __/ | | |
    #  |____/|_|  |_|_|  \__| |____/ \___|\__\___|\___|\__\___/|_|  |___/  \__,_|\__,_|_| |_|_| |_| |_|\__,_|_|\_\___|_| |_|

    print('Creating dashboard_dict...')
    # Noot, hier kunnen een arbitraire hoeveelheid lijst worden gemaakt, dus hier kan ook nog onderscheid gemaakt worden tussen vergelijken van labelled data met train data, oos met unlabelled, etc.
    NUMERIEK_DETECTORS = [KSDetector, partial(KSDetector, pvalue=True), DistributionDetector, ValueCountsDetector, MeanStdDetector]  
    CATEGORICAL_DETECTORS = [Chi2Detector, partial(Chi2Detector, pvalue=True), DistributionDetector, ValueCountsDetector]  
    SELECTORS = [GiniDetector] # nu nog één, dus niet zo'n spannende lijst :)

    dashboard_pages = dict()

    # Startpagina met Gini values feature overzicht; wellicht uit te breiden naar interactieve feature selection
    dashboard_pages["Feature selection"] = {
        "description": "This dashboard monitors changes in input (data) and output (predictions) of deployed machine learning models. This page shows ranked feature importance. The 10 most important features are selected for monitoring.",
        "display_columns": 1,
        "detectors": [selector(gini) for selector in SELECTORS]
    }
    # Toevoegen pagina met numerieke measures, maar alleen voor gelabelde data 
    dashboard_pages["Numeric data: labeled"] = {
        "description": "Here the numerical veriables from the operational data, for which inspections results are known, are compared against the data on which the model was trained. ",
        "columns": dd.numeric_columns(),
        "date_range": dd.date_range,
        "detectors": [detector(labelled_data) for detector in NUMERIEK_DETECTORS]
    }
    # Toevoegen pagina onderling vergelijken out of sample 
    dashboard_pages["Numeric data: out-of-sample"] = {
        "description": "Here the numerical variables from the out of sample data, for which inspection results are not known, are compared against the out of sample set at time of training.",
        "columns": dd.numeric_columns(),
        "date_range": dd.date_range,
        "detectors": [detector(oos) for detector in NUMERIEK_DETECTORS]
    }
    # Toevoegen pagina categorische features; labelled data
    dashboard_pages["Categorical data: labeled"] = {
        "description": "Here the categorical variables from the operational data, for which inspections results are known, are compared against the data on which the model was trained.",
        "columns": dd.categorical_columns(exclude=["dataset_markering"]),
        "date_range": dd.date_range,
        "detectors": [detector(labelled_data) for detector in CATEGORICAL_DETECTORS]
    }
    # Toevoegen pagina categorische features; oos data
    dashboard_pages["Categorical data: out-of-sample"] = {
        "description": "Here the categorical variables from the out of sample data, for which inspection results are not known, are compared against the out of sample set at time of training.",
        "columns": dd.categorical_columns(exclude=["dataset_markering"]),
        "date_range": dd.date_range,
        "detectors": [detector(oos) for detector in CATEGORICAL_DETECTORS]
    }

    # Toevoegen pagina vergelijk target predictions
    dashboard_pages["Target predictions"] = {
        "description": "Here the distribution of current target predictions are compared against the test predictions during the model was training/testing.",
        "columns": dt.numeric_columns(),
        "date_range": dt.date_range,
        "detectors": [detector(targets) for detector in NUMERIEK_DETECTORS]
    }

    print("Saving dashboard_dict.pickle.")
    # Writing dictionary to file
    with open('dashboard_dict.pickle', 'wb') as file:
        pickle.dump(dashboard_pages, file)

if __name__ == '__main__':
    main()

 






