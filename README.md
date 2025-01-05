# Tema1

Acest repository conține soluția pentru tema practică de **Predicție a Soldului** din Sistemul Energetic Național (SEN) pentru luna decembrie 2024, folosind algoritmii **ID3** și **Naive Bayes**, adaptați pentru regresie prin discretizare.

---

## Structura Repository-ului

- **src/**  
  - `main.py`: Conține codul principal (Python) care:
    1. Citește fișierul `sen_data.csv` (dacă există local).
    2. Exclude datele din decembrie 2024 la antrenarea modelelor.
    3. Discretizează variabilele de intrare și variabila țintă (Sold).
    4. Antrenează două modele (ID3 și Naive Bayes) și face predicții pentru decembrie 2024.
    5. Calculează și afișează metricele de regresie (RMSE și MAE).

- **`raport_sold.tex`**  
  Fișierul \(\LaTeX\) care generează raportul în format PDF. Acesta descrie în detaliu:
  - Contextul problemei
  - Metodologia (ID3, Naive Bayes, discretizare)
  - Rezultatele obținute și concluziile

- **README.md**  
  Explicații generale despre cum se structurează proiectul și cum se rulează.

---

## Cum Rulez Codul?

1. **Instalează** Python 3.7+ (ideal 3.8+).
2. Clonează repository-ul sau descarcă fișierele:
   ```bash
   git clone https://github.com/IOANASTEFANOE/Tema1.git
   cd Tema1
