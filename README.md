# 🗽 Airbnb NYC - Knowledge-Based System (KBS)

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Classification-green.svg)](https://xgboost.readthedocs.io/)
[![Owlready2](https://img.shields.io/badge/Owlready2-Ontology-yellow.svg)](https://owlready2.readthedocs.io/)

**Corso:** Ingegneria della Conoscenza (AA 2025/2026) — Università degli Studi di Bari Aldo Moro  
**Autore:** Alessandro Pesari  

---

## 📖 Descrizione del Progetto

Questo progetto sviluppa un sistema ibrido **Machine Learning + Background Knowledge (ML+OntoBK)** per l'analisi del mercato degli affitti a breve termine su Airbnb nella città di New York. 

A differenza di una classica pipeline di Data Science, il sistema sfrutta un'**Ontologia OWL 2** (Knowledge Base) per "ragionare" sui dati e sui risultati del clustering non supervisionato, arricchendo il dataset con nuove feature semantiche prima di addestrare i modelli predittivi.

### 🎯 Task Principali:
1. **Clustering (K-Means & GMM):** Segmentazione delle preferenze degli ospiti (es. soggiorni brevi vs lunghi, turisti vs residenti).
2. **Knowledge Base Enrichment:** Deduzione logica di nuove feature (es. *BusinessFriendly*, *LuxuryOption*) tramite pattern *Classify-then-Assert* con Owlready2.
3. **Price Prediction (Regressione):** Predizione del prezzo tramite **Random Forest** ottimizzata con GridSearchCV. *(Risultato: R² ≈ 0.85)*
4. **Availability Prediction (Classificazione):** Predizione della prenotabilità immediata tramite **XGBoost** e feature semantiche derivate dalla KB.

---

## ⚙️ Architettura del Sistema

Il progetto segue una pipeline automatizzata a 4 step principali, coordinata dal file `main.py`:

1. `DatasetPreProcessing`: Pulizia dati, imputation intelligente e One-Hot Encoding.
2. `UnsupervisedTrainingManager`: Estrazione dei segmenti di mercato (salvati come *cluster labels*).
3. `KnowledgeBase`: Costruzione dell'ontologia, classificazione OWL ed esportazione delle feature `kb_*`.
4. `SupervisedTrainingManager`: Addestramento, Cross-Validation (10-Fold) e testing dei modelli supervisionati.

---

## 📂 Struttura della Repository

\`\`\`text
ProgettoIcon-2026/
├── data/                       # Dataset originario, pulito e file dell'ontologia (.owl)
├── results/                    # Risultati generati in automatico (Grafici e Metriche CSV)
│   ├── classification/         # ROC Curves, Feature Importance (XGBoost)
│   ├── clustering/             # Scatter plots PCA K-Means e GMM
│   └── regression/             # Actual vs Predicted plots (Random Forest)
├── src/                        # Codice sorgente Python
│   ├── main.py                 # Entry point dell'applicazione
│   ├── KnowledgeBase.py        # Logica dell'ontologia Owlready2
│   └── ...                     # Moduli dei task ML
├── DocumentazioneIcon.pdf      # Relazione accademica completa
├── requirements.txt            # Dipendenze del progetto
└── README.md
\`\`\`

---

## 🚀 Installazione ed Esecuzione

### 1. Clona la repository
\`\`\`bash
git clone https://github.com/alespes/ProgettoIcon-2026.git
cd ProgettoIcon-2026
\`\`\`

### 2. Crea un ambiente virtuale (consigliato) e installa le dipendenze
\`\`\`bash
python3 -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate
pip install -r requirements.txt
\`\`\`

### 3. Avvia la pipeline
Il sistema farà girare l'intera architettura. Per via della validazione incrociata (GridSearchCV/RandomizedSearchCV), l'esecuzione potrebbe richiedere alcuni minuti.
\`\`\`bash
python -m src.main
\`\`\`

> **Nota sui risultati:** Al termine dell'esecuzione, la cartella `results/` verrà sovrascritta con i nuovi grafici `.png` generati dinamicamente (curve ROC, importanza delle feature, cluster) e i `.csv` con le metriche dei vari test.

---

## 📄 Documentazione Accademica

Per un'analisi approfondita delle scelte architetturali, della complessità dell'ontologia e dell'interpretazione critica dei risultati sperimentali (es. analisi dell'AUC sul task di classificazione), si prega di consultare la **[Documentazione Ufficiale](Documentazione_KBS_Airbnb.md)** presente in questa repository.
