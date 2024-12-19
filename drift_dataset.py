import re
import pandas as pd
from pathlib import Path
from typing import Callable, List
from IPython import embed

#   _   _      _                    __                  _   _                 
#  | | | | ___| |_ __   ___ _ __   / _|_   _ _ __   ___| |_(_) ___  _ __  ___ 
#  | |_| |/ _ \ | '_ \ / _ \ '__| | |_| | | | '_ \ / __| __| |/ _ \| '_ \/ __|
#  |  _  |  __/ | |_) |  __/ |    |  _| |_| | | | | (__| |_| | (_) | | | \__ \
#  |_| |_|\___|_| .__/ \___|_|    |_|  \__,_|_| |_|\___|\__|_|\___/|_| |_|___/
#               |_|                                                           

def noop(x):
    """'No operation': returns input unchanged"""
    return x

class DriftFilesNotFound(Exception):
    """Exception class to be thrown when no files are found in the given Path.
    Parameters:
        path -- the original Path provided to the constructor of the DriftDataset    
    """
    def __init__(self, path: Path) -> None:
        self.message = f'No files of format .csv found in "{path.absolute()}"'  # TODO mogelijk meerdere bestandsformaten toevoegen
        super().__init__(self.message)

class DateParseError(Exception):
    """Exception class to be thrown when date parsing fails.
    Parameters:
        filenames -- a list of filenames for which the dates did not parse correctly.
    """
    def __init__(self, filenames: list[str]) -> None:
        filenames = '\n'.join(filenames)
        self.message = 'Date parsing from filenames failed for the following files:\n' + filenames + '\nAllowed formats are: yyyymm(dd) or yyyy-mm(-dd) or yyyy/mm(/dd) (days are optional)'
        super().__init__(self.message)

#   ____       _  __ _   ____        _                 _   
#  |  _ \ _ __(_)/ _| |_|  _ \  __ _| |_ __ _ ___  ___| |_ 
#  | | | | '__| | |_| __| | | |/ _` | __/ _` / __|/ _ \ __|
#  | |_| | |  | |  _| |_| |_| | (_| | || (_| \__ \  __/ |_ 
#  |____/|_|  |_|_|  \__|____/ \__,_|\__\__,_|___/\___|\__|
                                                                   
        
# TODO: Meer bestandsformaten toestaan dan alleen CSV?
# TODO: Is de manier waarop we datum uit de bestandsnaam uitlezen handig?

class DriftDataset:
    """Object for reading in and collectively storing multiple datasets with the goal of
       recognising potential data drift.
    
        Parameters:
        path: pathlib.Path to folder containing the datasets on which drift is 
              to be investigated. This folder can contain any number of files, 
              but at least two are expected. The calibration date for each 
              dataset has to be part of the filename in the format "YYYYMM(DD)" 
              (days being optional).
              The following formats are recognised: .csv.
        pre_process (Optional): A function to be run as a preprocessing step. 
                                This function is run on each file, directly 
                                after it is read from disk.
                                Must take a pandas.DataFrame as input and 
                                return a pandas.DataFrame as output.                      
        post_process (Optional): A function to be run as a postprocessing step. 
                                 This function is run on the entire dataset, 
                                 after all files have been concatenated.
                                 Must take a pandas.DataFrame as input and 
                                 return a pandas.DataFrame as output.  
    """
    
    def __init__(self, path: Path, 
                 pre_process: Callable[[pd.DataFrame], pd.DataFrame]=None, 
                 post_process: Callable[[pd.DataFrame], pd.DataFrame]=None):
        self.path = path
        self.preprocessing = pre_process if pre_process is not None else noop
        self.postprocessing = post_process if post_process is not None else noop
        self.files = [file for file in self.path.glob('*.csv')] # TODO mogelijk meerdere bestandsformaten toevoegen
        if len(self.files) == 0: raise DriftFilesNotFound(self.path)   # throwing an exception if no files of the supported format were found in the provided path
        self.ordered_files = self._order_files()
        self.drift_dataset = self._make_drift_dataset()
        self.keep = set(['dataset_index', 'dataset_date'])  # set of columns that should always be in the drift dataset
        
    def _order_files(self) -> pd.DataFrame:
        """Order files chronologically. Returns a dataframe with ordered datasets."""
        # creating a dataframe for all the file paths
        files = pd.DataFrame(self.files, columns=['path'])
        files['file'] = [file.stem for file in self.files]
        files['suffix'] = [file.suffix for file in self.files]
        
        # TODO uitlezen met regex is flexibel, maar ook wel een beetje rommelig. Gewoon goede afspraken maken voor standaarden?
        # reading out the date from the file names using regex
        files['ndate'] = files.file.str.extract(r'(\d{4}[-/]?\d{2}[-/]?\d{0,2})') # assuming there is a date in format yyyymm(dd) or yyyy-mm(-dd) or yyyy/mm(/dd)
        
        if files.ndate.isnull().values.any():   # checking whether the extraction failed for any of the files, and raising an exception
            filenames = files[files.ndate.isnull()].file
            raise DateParseError(filenames)
            
        files.ndate = files.ndate.str.replace(r'[-/]', '', regex=True)  # removing the - or / if it exists in the date format
        files['strlen'] = files.ndate.str.len()  # length of string suggests the date parsing format: e.g. 6 is yyyymm, while larger gives yyyymmdd. Any under 6 are caught by the error above.
        
        # adding dataset_date
        mask = files.strlen == 6  # mask for strings of length 6, all other will have length > 6
        files['dataset_date'] = pd.to_datetime('1900-01-01')  # need a datetime to use .dt later, starting with obvious bogus date in case something parses wrong
        files.loc[mask, 'dataset_date'] = pd.to_datetime(files.loc[mask, 'ndate'], format='%Y%m')
        files.loc[~mask, 'dataset_date'] = pd.to_datetime(files.loc[~mask, 'ndate'], format='%Y%m%d')       
        files.dataset_date = files.dataset_date.dt.strftime('%Y-%m-%d')
        
        # sorting the filenames in chronological order and making a dataset index to be able to distinguish them later
        files = files.sort_values(by=['dataset_date'])
        files = files.reset_index(drop=True)
        files['dataset_index'] = files.index        
        files = files.drop(columns=['ndate', 'strlen']) # removing superfluous columns
        
        return files
    
    def _read_file(self, row: pd.DataFrame) -> pd.DataFrame:
        """Reads in a file and adds information from self.ordered_files.
        Parameters:
            row: a single row from self.ordered_files        
        """
        # TODO uitbereiden naar meerdere bestandsformaten die we hebben besloten te ondersteunen
        
        # reading in the file and adding in the dataset_index, dataset_date and original filename for future reference
        data = pd.read_csv(row.path)
        data['dataset_index'] = row.dataset_index
        data['dataset_date'] = row.dataset_date
        data = self.preprocessing(data)   # Runs preprocessing
        return data
        
    def _make_drift_dataset(self) -> pd.DataFrame:
        """Function which reads in an ordered list of files and adds the dataset date and index."""
        # NOTE: this is probably the slowest part
        # TODO (misschien) als blijkt dat dit erg lang duurt, bij veel bestanden, optie voor multithreading toepassen
        data = [self._read_file(row) for _, row in self.ordered_files.iterrows()]
        data = pd.concat(data, ignore_index=True)
        data = self.postprocessing(data)  # Runs postprocessing
        return data
    
    def drop_columns(self, columns: List[str]) -> None:
        """Drop columns from the drift_dataset.
        Will refuse to drop `dataset_index` and `dataset_date`.
        
        Parameters:
            columns: list of columns (str) to drop. 
        """
        # making sure the drift_datset columns are kept regardless
        columns = set(columns).difference(self.keep)
        
        # checking if all requested columns exist in the drift_dataset:
        all_columns = set(self.drift_dataset.columns)
        missing = columns.difference(all_columns)   # difference here checks what is left when removing all_columns from columns. Should be empty, else raising an exception.        
        if len(missing): raise ValueError(f'Requested column(s) {missing} not found in drift dataset, cannot drop what is not there.')

        self.drift_dataset = self.drift_dataset.drop(columns=columns, axis=1)
        
    def keep_columns(self, columns: List[str]) -> None:
        """Keep columns in the drift dataset, and drop the rest.
        Will refuse to drop `dataset_index` and `dataset_date`.
        
        Parameters:
            columns: list of columns (str) to keep
        """
        # making sure the drift_dataset columns are kept regardless
        columns = set(columns).union(self.keep)
        
        # checking if all requested columns exist in the drift_dataset:
        all_columns = set(self.drift_dataset.columns)
        missing = columns.difference(all_columns)  # difference here checks what is left when removing all_columns from columns. Should be empty, else raising an exception.
        if len(missing): raise ValueError(f'Requested columns(s) {missing} not found in drit dataset, cannot keep what is not there.')

        self.drift_dataset = self.drift_dataset[columns]
        
    def restore_columns(self) -> pd.DataFrame:
        """Restores all the columns in the original dataset.
        (in case something is dropped which is needed later)"""
        self.drift_dataset = self._make_drift_dataset()
    
    def numeric_columns(self, incl_indices: bool=False, exclude: List[str]=None) -> List[str]:
        """Returns list of names of all numeric columns.
        
        Parameters:
            incl_indices (Optional):  whether to include dataset_index and dataset_date
            exclude (Optional):  list of variables that can be excluded
        """
        columns = self.drift_dataset.select_dtypes('number').columns
        if exclude is not None:
            columns = set(columns).difference(set(exclude))  # difference takes what is in the first and not in the second set
        # adding or removing index columns (self.keep)
        if incl_indices:
            columns = set(columns).union(self.keep)
        else:
            columns = set(columns).difference(self.keep)

        return list(columns)
    
    def categorical_columns(self, incl_indices: bool=False, exclude: List[str]=None) -> List[str]:
        """Returns list of names of all numeric columns.
        
        Parameters:
            incl_indices (Optional):  whether to include dataset_index and dataset_date
            exclude (Optional):  list of variables that can be excluded
        """
        columns = self.drift_dataset.select_dtypes(exclude='number').columns
        if exclude is not None:
            columns = set(columns).difference(set(exclude))  # difference takes what is in the first and not in the second set
        # adding or removing index columns (self.keep)
        if incl_indices:
            columns = set(columns).union(self.keep)
        else:
            columns = set(columns).difference(self.keep)

        return list(columns)
    
    
    @property
    def dataset(self) -> pd.DataFrame:
        """Returns a combined pandas.DataFrame of the drift datasets, 
        with added columns indicating the dataset index and date."""
        return self.drift_dataset
    
    @property
    def original_files(self) -> pd.DataFrame:
        """Returns a pandas.DataFrame with an overview of the files used to 
        construct the drift datasets."""
        return self.ordered_files
    
    @property
    def date_range(self) -> List[str]:
        """Returns a list with the dataset dates, excluding the training data."""
        return self.ordered_files.dataset_date.unique().tolist()[1:]
    
    @property
    def train_date(self) -> str:
        """Returns the date of the trainig data"""
        return self.ordered_files.dataset_date.unique().tolist()[0]
    

    