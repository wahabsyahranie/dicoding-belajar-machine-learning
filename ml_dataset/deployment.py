import joblib
import pickle

# memuat model dari file joblib
joblib_model = joblib.load('gbr_model.joblib')

# memuat model dari file picke
with open('gbr_model.pkl', 'rb') as file:
    pickle_model = pickle.load(file)
