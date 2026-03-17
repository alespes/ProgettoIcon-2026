# import KBManager
# import mainOntology
from . import DatasetPreProcessing
from . import DataAnalyzer
from . import SupervisedTrainingManager
from . import UnsupervisedTrainingManager

DatasetPreProcessing.call()
DataAnalyzer.call()
SupervisedTrainingManager.call()
UnsupervisedTrainingManager.call()