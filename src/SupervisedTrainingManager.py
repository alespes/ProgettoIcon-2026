from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostRegressor, AdaBoostClassifier
import pandas as pds
from xgboost import XGBRegressor

from src.AvailabilityPredictionTask import AvailabilityPredictionTask
from src.PricePredictionTask import PricePredictionTask


class SupervisedTrainingManager:
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
            print(f"Errore: File non trovato - {self.file_path}, probabile file mancante oppure il percorso indicato è errato")
        except Exception as e:
            print(f"Errore generico durante il caricamento dei dati: {str(e)}")

def call():
    original_file_path = 'data/Post_PreProcessing/Airbnb_Processed_Data.csv'
    manager = SupervisedTrainingManager(original_file_path)

    regression_task = False
    if regression_task:
        # Price prediction task
        price_predictor = PricePredictionTask(manager.data, "price", model = RandomForestRegressor(n_estimators=100, random_state=42)) # RandomForestRegressor(n_estimators=100, random_state=42)
        price_predictor.call(True, False, True, True, True)

    classification_task = True
    if classification_task:
        # Availability prediction task
        availability_predictor = AvailabilityPredictionTask(manager.data, "instant_bookable")
        availability_predictor.call(True, False, True, True, True)
