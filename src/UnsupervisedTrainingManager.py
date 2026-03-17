import pandas as pds

from src.EM_Implementation import EM_Implementation
from src.GuestPreferenceSegmentationTask import GuestPreferenceSegmentationTask


class UnsupervisedTrainingManager:
    # Class constructor
    def __init__(self, input_file_path):
        self.file_path = input_file_path
        self.data = None

        self.load_data()

    # Data loaded directly from class specified file, assuming the file format is CSV
    # Can catch specific FileNotFound exceptions or generic ones
    def load_data(self):
        # Trying to load the specified CSV file in a pandas DataFrame
        try:
            self.data = pds.read_csv(self.file_path, low_memory=False)
            print("Caricamento dati completato")
        except FileNotFoundError:
            print(
                f"Errore: File non trovato - {self.file_path}, probabile file mancante oppure il percorso indicato è errato")
        except Exception as e:
            print(f"Errore generico durante il caricamento dei dati: {str(e)}")


def call():
    original_file_path = 'data/Post_PreProcessing/Airbnb_Processed_Data.csv'
    manager = UnsupervisedTrainingManager(original_file_path)

    hard_clustering_task = True
    if hard_clustering_task:
        hard_clustering = GuestPreferenceSegmentationTask(manager.data)
        hard_clustering.call()

    soft_clustering_task = True
    if soft_clustering_task:
        soft_clustering = EM_Implementation(manager.data)
        soft_clustering.call()

