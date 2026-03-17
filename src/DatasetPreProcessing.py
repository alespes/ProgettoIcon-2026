import os

import pandas as pds
import numpy as npy
from datetime import datetime


class AirBnBDatasetPreprocessing:

    # Class constructor
    def __init__(self, input_file_path, output_file_path):
        self.original_file_path = input_file_path
        self.processed_file_path = output_file_path
        self.data = None

    # Data loaded directly from class specified file, assuming the file format is CSV
    # Can catch specific FileNotFound exceptions or generic ones
    def load_data(self):
        # Trying to load the specified CSV file in a pandas DataFrame
        try:
            self.data = pds.read_csv(self.original_file_path, low_memory=False)
            print("Caricamento dati completato")
        except FileNotFoundError:
            print(f"Errore: File non trovato - {self.original_file_path}, probabile file mancante oppure il percorso indicato è errato")
        except Exception as e:
            print(f"Errore generico durante il caricamento dei dati: {str(e)}")

    def clean_data(self):
        if self.data is None:
            print("Impossibile proseguire con la pulizia dei dati\n"
                  "Dati non trovati")
        else:
            print("Inizio pulizia dei dati")
            # Dropping unnecessary columns
            columns_to_drop = ['NAME',
                               'country',
                               'country code',
                               'license',
                               'host name',
                               'house_rules']

            self.data.drop(columns=columns_to_drop, inplace=True)

            # Dropping entries containing any missing values on the specified column,
            # since a statistical replacement will not be consistent
            self.data.dropna(subset=['neighbourhood group',
                                     'neighbourhood',
                                     'lat',
                                     'long',
                                     'instant_bookable'],
                             inplace=True)

            # Collapsing any duplicates that originated from the previous drops
            self.data.drop_duplicates(inplace=True)

            ''' Data formatting from this point on '''

            # Removing "$" symbol in 'price' column and converting values to numerical
            self.data['price'] = self.data['price'].str.replace('$', '', regex=False)
            self.data['price'] = self.data['price'].str.replace(',', '', regex=False)
            self.data['price'] = self.data['price'].str.replace(' ', '', regex=False)
            self.data['price'] = self.data['price'].astype('Int64')
            # Same for column 'service fee'
            self.data['service fee'] = self.data['service fee'].str.replace('$', '', regex=False)
            self.data['service fee'] = self.data['service fee'].str.replace(',', '', regex=False)
            self.data['service fee'] = self.data['service fee'].str.replace(' ', '', regex=False)
            self.data['service fee'] = self.data['service fee'].astype(float)

            # Converting the 'construction year' column in type Int64
            self.data['Construction year'] = self.data['Construction year'].astype('Int64')
            # Same for 'minimum nights' column
            self.data['minimum nights'] = self.data['minimum nights'].astype('Int64')

            # Converting 'last review' column in a consistent date format
            self.data['last review'] = pds.to_datetime(self.data['last review'])
            mean_date = self.data['last review'].mean().date()
            self.data['last review'] = self.data['last review'].astype('datetime64[ns]')
            self.data['last review'] = self.data['last review'].fillna(pds.to_datetime(mean_date))

            # Converting 'room type' column in a consistent string format
            self.data['room type'] = self.data['room type'].astype(str)

            # Managing 'availability 365' column, changing all negative values to positive,
            # then collapsing all values exceeding the maximum to 365
            self.data['availability 365'] = npy.where(self.data['availability 365'] < 0,
                                                      self.data['availability 365'] * -1,
                                                      self.data['availability 365'])
            self.data['availability 365'] = npy.where(self.data['availability 365'] > 365,
                                                      365,
                                                      self.data['availability 365'])

            # Same job for 'minimum nights' column, plus filling the blanks with 1 as a default value
            self.data['minimum nights'] = self.data['minimum nights'].mask(self.data['minimum nights'] < 0,
                                                                           self.data['minimum nights'] * -1)
            self.data['minimum nights'] = self.data['minimum nights'].mask(self.data['minimum nights'] > 365,
                                                                           365)
            self.data['minimum nights'] = self.data['minimum nights'].fillna(1)

            # Fixing specific typos in 'neighbourhood group' column
            self.data['neighbourhood group'] = self.data['neighbourhood group'].replace('brookln', 'Brooklyn')

            ''' Data completion from this point on '''

            '''
               Considering the dataset is normally distributed, filling the holes with the mean is a good compromise 
               of simplicity and cheapness, yet less precise and reliable than a more complex metric
            '''
            # Filling Na/NaN values in 'price' column with the mean value
            self.data['price'] = self.data['price'].fillna(self.data['price'].mean().astype(int))
            # Same for 'service fee' column
            self.data['service fee'] = self.data['service fee'].fillna(self.data['service fee'].mean().astype(int))

            # Filling Na/Null values in 'host_identity_verified' column with the default value 'unconfirmed'
            self.data['host_identity_verified'] = self.data['host_identity_verified'].fillna('unconfirmed')

            # Filling Na/Null values in 'calculated host listing count' column with the precise value
            self.data['calculated host listings count'] = self.data.groupby('host id')[
                'calculated host listings count'].transform(lambda x: x.fillna(x.count()))

            # Filling Na/Null values in 'cancellation_policy' column with the modal value
            modal_value = self.data['cancellation_policy'].mode()[0]
            self.data['cancellation_policy'] = self.data['cancellation_policy'].fillna(modal_value)

            # Filling Na/Null values in 'Construction year' column with the corresponding neighbourhood mean
            construction_year_mean = self.data.groupby('neighbourhood group')['Construction year'].mean()
            self.data['Construction year'] = self.data.apply(
                lambda row: construction_year_mean[row['neighbourhood group']]
                if pds.isnull(row['Construction year'])
                else row['Construction year'], axis=1)
            self.data['Construction year'] = self.data['Construction year'].astype(int)

            # Filling Na/Null values in the selected columns with the corresponding median or mean value,
            # based on the consistency of the metric
            self.data['number of reviews'] = self.data['number of reviews'].fillna(self.data['number of reviews'].median())
            self.data['reviews per month'] = self.data['reviews per month'].fillna(self.data['reviews per month'].median())
            self.data['availability 365'] = self.data['availability 365'].fillna(self.data['availability 365'].median())
            self.data['review rate number'] = self.data['review rate number'].fillna(self.data['review rate number'].mean().astype(int))

            # Fixing inconsistency between the 'number of reviews' column and the 'reviews per month' column
            self.data.loc[self.data['number of reviews'] == 0, 'reviews per month'] = 0

            # Fixing dates indicating points in the future
            today = datetime.now().date()
            self.data['last review'] = self.data['last review'].apply(
                lambda x: today if pds.notna(x) and x.date() > today else x)
            self.data['last review'] = pds.to_datetime(self.data['last review'])
            mean_date = self.data['last review'].mean().date()
            self.data['last review'] = self.data['last review'].fillna(mean_date)
            self.data['last review'] = pds.to_datetime(self.data['last review'])

            print("Tutti i dati sono stati puliti")

    def save_processed_data(self):
        # Creating the directory for the output data, if not existing
        output_path = os.path.join(self.processed_file_path, 'Post_PreProcessing')
        os.makedirs(output_path, exist_ok=True)

        # Saving freshly processed data in a dedicated CSV file
        if self.data is not None:
            try:
                output_file_path = os.path.join(output_path, 'Airbnb_Processed_Data.csv')
                self.data.to_csv(output_file_path, index=False)
                print(f"Dati processati salvati in {output_file_path}.")
            except Exception as e:
                print(f"Errore durante il salvataggio dei dati processati: {str(e)}")

    def show_data(self):
        self.data.info()

def call():
    original_file_path = 'data/Airbnb_Open_Data.csv'
    output_directory = 'data'
    preprocessor = AirBnBDatasetPreprocessing(original_file_path, output_directory)
    preprocessor.load_data()
    preprocessor.show_data()
    preprocessor.clean_data()
    preprocessor.show_data()
    preprocessor.save_processed_data()