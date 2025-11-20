# crop-disease-flask

Multi-model Flask project (Models A/B/C) for predicting crop disease.

- Model A: Inputs = Crop Stage + Region  -> Predict Crop Disease
- Model B: Inputs = Crop Stage + Region  -> Predict Crop Disease **and** Cause
- Model C: Inputs = Crop Stage + Region + Cause  -> Predict Crop Disease

## How to use

1. Place your dataset in `data/crop_diseases.csv` (already copied from your upload if provided).
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\\Scripts\\activate
   pip install -r requirements.txt
   ```
3. Train models:
   ```bash
   python train.py
   ```
   This will create `models/model_A.pkl`, `models/model_B.pkl`, and `models/model_C.pkl`.
4. Run the Flask app:
   ```bash
   python app.py
   ```
5. Open `http://127.0.0.1:5000` in your browser.

Notes:
- The app auto-selects model: if "Cause" is entered it uses Model C; if you tick "Predict Disease + Cause" it uses Model B; otherwise Model A.
- If you see "Unknown stage or region" errors, make sure your input matches values in the dataset (case-insensitive).