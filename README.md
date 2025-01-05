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

## Cum rulez codul? (Instrucțiuni Generale)

1. **Descărcați fișierul** `main.py` din acest repository (acesta conține tot codul necesar).  
2. **Obțineți datele** de la [Transelectrica - SEN Grafic](https://www.transelectrica.ro/widget/web/tel/sen-grafic/-/SENGrafic_WAR_SENGraficportlet).  
   - După ce generați/exportați datele de acolo, salvați fișierul sub numele `sen_data.csv` (în format CSV).  
3. **Asigurați-vă** că `sen_data.csv` se află în același folder cu `main.py` (sau modificați calea în cod, dacă-l așezați altundeva).  
4. **Pregătiți un mediu Python** (local sau un IDE online) și instalați librăriile necesare (`pandas`, `numpy`, `scikit-learn`) pentru a evita erorile de rulare.  
5. **Rulați** `main.py`:  
6. În consolă vor apărea valorile RMSE și MAE pentru datele din decembrie 2024, precum și concluzii despre comportamentul modelelor ID3 și Naive Bayes.
