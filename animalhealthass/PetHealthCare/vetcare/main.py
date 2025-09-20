from random_forest import bagging_predict
import numpy as np
import pandas as pd
from utils import load_and_prepare_data
from random_forest import random_forest

# Load and prepare data
data, encoders = load_and_prepare_data('animal_health_dataset.csv')
X, y = data[:, :-1], data[:, -1]
train = np.column_stack((X, y))

# Train the Random Forest and get trees
n_trees = 10
max_depth = 5
min_size = 5
sample_size = 0.8
trees = random_forest(train, X, max_depth, min_size, sample_size, n_trees)

# Get encoders
animal_encoder = encoders['Animal']
gender_encoder = encoders['Gender']
disease_encoder = encoders['Disease']

# Medication mapping with available products in India
medication_mapping = {
    'Parvovirus': {
        'description': 'Supportive care including fluid therapy and antiemetics.',
        'products': [
            {'name': 'Parvolix-PV Drops', 'link': 'https://www.all4pets.in/product/parvolix-pv-drops-100ml-for-treatment-of-parvo-virusfor-dogs-cats/'},
            {'name': "Bakson's Parvo Aid Drops", 'link': 'https://www.1mg.com/otc/bakson-s-homeopathy-parvo-drop-otc941317'}
        ]
    },
    'Feline Leukemia': {
        'description': 'Antiviral medications like AZT and supportive care.',
        'products': [
            {'name': 'Virbagen Omega (Interferon-ω)', 'link': 'https://ph.virbac.com/products/immunomodulator/virbagen-omega'},
            {'name': 'Immunol Pet Supplement', 'link': 'https://www.amazon.in/Himalaya-Immunol-Liquid-100-ml/dp/B08L7XGSRY'}
        ]
    },
    'Rabies': {
        'description': 'Immediate vaccination and administration of immunoglobulins.',
        'products': [
            {'name': 'Rabipur Vaccine', 'link': 'https://www.1mg.com/drugs/rabipur-injection-150208'},
            {'name': 'Verorab Vaccine', 'link': 'https://www.1mg.com/drugs/verorab-injection-150407'}
        ]
    },
    'Gastroenteritis': {
        'description': 'Fluid therapy, antiemetics, and dietary management.',
        'products': [
            {'name': 'Diafine Drops', 'link': 'https://goelvetpharma.com/product/diafine-for-pets-diarrhoea-parvo-virus/'},
            {'name': 'Emepet Oral Spray', 'link': 'https://animeal.in/products/emepet-oral-spray'},
            {'name': 'ORS Electrolyte Solution', 'link': 'https://www.1mg.com/otc/ors-powder-lemon-otc735155'}
        ]
    },
    'Avian Flu': {
        'description': 'Supportive care with antiviral medications like oseltamivir.',
        'products': [
            {'name': 'Oseltamivir (Tamiflu)', 'link': 'https://www.1mg.com/drugs/tamiflu-75mg-capsule-329217'},
            {'name': 'Doxycycline (Vibramycin)', 'link': 'https://www.1mg.com/drugs/vibramycin-100mg-capsule-154829'},
            {'name': 'Vitamin A-D3-E Supplements', 'link': 'https://www.indiamart.com/proddetail/vitamin-a-d3-e-liquid-supplement-25564292473.html'}
        ]
    },
    'Mite Infestation': {
        'description': 'Topical acaricides and environmental cleaning.',
        'products': [
            {'name': 'Butox® Vet', 'link': 'https://www.msd-animal-health.co.in/products/butox-vet/'},
            {'name': 'Taktic® 12.5% EC', 'link': 'https://www.msd-animal-health.co.in/products/taktic-12-5-ec/'},
            {'name': 'Scabovate Lotion', 'link': 'https://www.1mg.com/otc/scabovate-lotion-otc351239'}
        ]
    }
}

# Prompt user for input
print("\nAnimal Health Prediction System")
print("Select Animal:")
for i, label in enumerate(animal_encoder.classes_):
    print(f"{i + 1}. {label}")
animal_input = int(input("Enter the number: ")) - 1
animal_encoded = animal_encoder.transform([animal_encoder.classes_[animal_input]])[0]

age = float(input("Enter age: "))

print("\nSelect Gender:")
for i, label in enumerate(gender_encoder.classes_):
    print(f"{i + 1}. {label}")
gender_input = int(input("Enter the number: ")) - 1
gender_encoded = gender_encoder.transform([gender_encoder.classes_[gender_input]])[0]

print("\nEnter symptoms (1 = Yes, 0 = No):")
fever = int(input("Fever: "))
cough = int(input("Cough: "))
vomiting = int(input("Vomiting: "))
diarrhea = int(input("Diarrhea: "))
lethargy = int(input("Lethargy: "))
appetite = int(input("Loss of Appetite: "))
sneezing = int(input("Sneezing: "))
rash = int(input("Skin Rash: "))

# Collect symptoms
symptoms = [fever, cough, vomiting, diarrhea, lethargy, appetite, sneezing, rash]

if all(symptom == 0 for symptom in symptoms):
    predicted_disease = "Healthy"
    medication_info = {
        'description': 'No medication required. Maintain good nutrition and regular check-ups.',
        'products': []
    }
else:
    # Create test sample and predict
    sample = [animal_encoded, age, gender_encoded, fever, cough, vomiting,
              diarrhea, lethargy, appetite, sneezing, rash]
    prediction = bagging_predict(trees, sample)
    predicted_disease = disease_encoder.inverse_transform([int(prediction)])[0]
    medication_info = medication_mapping.get(predicted_disease, {
        'description': 'Consult a veterinarian for appropriate medication.',
        'products': []
    })

# Output
print("\nPredicted Disease:", predicted_disease)
# Ask if the user wants to see the recommended treatment
show_treatment = input("\nWould you like to see the recommended treatment and medications? (1 = Yes, 0 = No): ")

if show_treatment.strip() == '1':
    treatment_info = medication_mapping.get(predicted_disease)
    if treatment_info:
        print("\nRecommended Treatment:", treatment_info['description'])
        print("Available Medications:")
        for med in treatment_info['products']:
            print(f"- {med['name']}: {med['link']}")
    else:
        print("\nNo specific medications found. Please consult a veterinarian.")
else:
    print("\nThank you for using the Animal Health Prediction System!")