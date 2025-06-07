import os
import pandas as pd

class EEGDataLoader:
    """
    Class to handle loading EEG data from CSV files.
    """

    def __init__(self, file_path):
        """
        Initializes the EEGDataLoader with a specific CSV file path.

        :param file_path: The full path to the CSV file to load.
        """
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """
        Loads the EEG data from the CSV file and stores it in the data attribute.

        :return: pandas DataFrame containing EEG data, or None if an error occurs.
        """
        try:
            self.data = pd.read_csv(self.file_path, delimiter=",")
            if self.data.empty or "A1" not in self.data.columns or "A2" not in self.data.columns:
                print(f"Warning: Invalid data format in {self.file_path}. Expected columns 'A1' and 'A2'.")
                return None
            return self.data
        except FileNotFoundError:
            print(f"Error: The file '{self.file_path}' was not found.")
            return None
        except pd.errors.EmptyDataError:
            print(f"Error: The file '{self.file_path}' is empty.")
            return None
        except pd.errors.ParserError:
            print(f"Error: There was an error parsing the file '{self.file_path}'.")
            return None
        except Exception as e:
            print(f"Error: Unexpected error loading {self.file_path}: {str(e)}")
            return None

    def get_channels(self):
        """
        Returns the available EEG channels (columns) from the loaded data.

        :return: List of column names representing EEG channels, or an empty list if no data is loaded.
        """
        if self.data is not None:
            return self.data.columns.tolist()
        else:
            print("Error: EEG data has not been loaded.")
            return []