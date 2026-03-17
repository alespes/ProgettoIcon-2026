import pandas as pds
import matplotlib.pyplot as plt
import seaborn as sns


class DataAnalyzer:
    def __init__(self, input_file_path):
        self.original_file_path = input_file_path
        self.data = None

    def load_data(self):
        # Trying to load the specified CSV file in a pandas DataFrame
        try:
            self.data = pds.read_csv(self.original_file_path, low_memory=False)
            print("Caricamento dati completato")
        except FileNotFoundError:
            print(f"Errore: File non trovato - {self.original_file_path}, probabile file mancante oppure il percorso indicato è errato")
        except Exception as e:
            print(f"Errore generico durante il caricamento dei dati: {str(e)}")

    def show_data(self):
        plt.figure(figsize = (10, 6))
        sns.violinplot(x=self.data['availability 365'])
        plt.show()

        plt.scatter(self.data['minimum nights'], self.data['price'], alpha=0.6)

        # Customize the plot
        plt.title('Minimum Nights vs Price', fontsize=14)
        plt.xlabel('minimum nights', fontsize=12)
        plt.ylabel('price', fontsize=12)

        # Set the axis limits
        plt.xlim(0, 365)
        plt.ylim(0, 1300)

        # Add gridlines
        plt.grid(True, linestyle='--', alpha=0.7)

        # Show the plot
        plt.tight_layout()
        plt.show()

def call():
    original_file_path = 'data/Post_PreProcessing/Airbnb_Processed_Data.csv'
    analyzer = DataAnalyzer(original_file_path)
    analyzer.load_data()
    analyzer.show_data()