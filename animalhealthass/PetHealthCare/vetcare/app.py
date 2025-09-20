from flask import Flask, render_template, url_for, redirect, flash, request
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Email, Length, EqualTo, ValidationError
from flask_bcrypt import Bcrypt
from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView
from flask_cors import CORS
import numpy as np
from flask import request, jsonify

from utils import load_and_prepare_data
from random_forest import random_forest, bagging_predict

# --------------------- App Config ---------------------
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'thisisasecretkey'

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})

# --------------------- User Model ---------------------


class User(db.Model, UserMixin):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(100), nullable=False)


# --------------------- Flask-Login Setup ---------------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# --------------------- Flask-Admin Setup ---------------------
admin = Admin(app, name='VetCare Admin', template_mode='bootstrap3')
admin.add_view(ModelView(User, db.session))

# --------------------- Forms ---------------------


class RegisterForm(FlaskForm):
    email = StringField(validators=[InputRequired(), Email(), Length(max=150)],
                        render_kw={"placeholder": "Email"})
    password = PasswordField(validators=[InputRequired(), Length(min=8, max=100)],
                             render_kw={"placeholder": "Password"})
    confirm_password = PasswordField(validators=[InputRequired(), EqualTo('password')],
                                     render_kw={"placeholder": "Confirm Password"})
    submit = SubmitField('Register')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError(
                'Email already registered. Please use a different one.')


class LoginForm(FlaskForm):
    email = StringField(validators=[InputRequired(), Email(), Length(max=150)],
                        render_kw={"placeholder": "Email"})
    password = PasswordField(validators=[InputRequired(), Length(min=8, max=100)],
                             render_kw={"placeholder": "Password"})
    submit = SubmitField('Login')


# --------------------- ML Data Preparation ---------------------
data, encoders = load_and_prepare_data('animal_health_dataset.csv')
X, y = data[:, :-1], data[:, -1]
train_data = np.column_stack((X, y))

# Train the Random Forest
n_trees = 10
max_depth = 5
min_size = 5
sample_size = 0.8
trees = random_forest(train_data, X, max_depth, min_size, sample_size, n_trees)

animal_encoder = encoders['Animal']
gender_encoder = encoders['Gender']
disease_encoder = encoders['Disease']

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

# --------------------- Routes ---------------------


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if not user:
            flash("User not found. Please register first.", "warning")
            return redirect(url_for('register'))

        if bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            flash("Logged in successfully.", "success")
            return redirect(url_for('dashboard'))
        else:
            flash("Incorrect password. Try again.", "danger")
    return render_template('login.html', form=form)


@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(
            form.password.data).decode('utf-8')
        new_user = User(email=form.email.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash("Registration successful. Please login.", "success")
        return redirect(url_for('login'))
    return render_template('register.html', form=form)


@app.route('/healthcheck', methods=['GET', 'POST'])
@login_required
def healthcheck():
    if request.method == 'POST':
        try:
            animal = request.form.get('animal')
            age = float(request.form.get('age'))
            gender = request.form.get('gender')

            def get_symptom(symptom):
                val = request.form.get(symptom)
                return int(val) if val else 0

            symptoms = [get_symptom(s) for s in ['fever', 'cough', 'vomiting', 'diarrhea',
                                                 'lethargy', 'appetite', 'sneezing', 'rash']]

            animal_encoded = animal_encoder.transform([animal])[0]
            gender_encoded = gender_encoder.transform([gender])[0]

            sample = [animal_encoded, age, gender_encoded] + symptoms

            prediction = bagging_predict(trees, sample)
            predicted_label = disease_encoder.inverse_transform([int(prediction)])[
                0]

            animals = list(animal_encoder.classes_)
            genders = list(gender_encoder.classes_)

            return render_template('healthcheck.html',
                                   prediction=predicted_label,
                                   animals=animals,
                                   genders=genders)

        except Exception as e:
            flash(f"Error during prediction: {str(e)}", "danger")

    animals = list(animal_encoder.classes_)
    genders = list(gender_encoder.classes_)
    return render_template('healthcheck.html', animals=animals, genders=genders)


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        data = request.get_json()

        animal = data.get('animal')
        age = float(data.get('age'))
        gender = data.get('gender')

        # Collect symptoms safely with default 0
        symptoms = [
            int(data.get('fever', 0)),
            int(data.get('cough', 0)),
            int(data.get('vomiting', 0)),
            int(data.get('diarrhea', 0)),
            int(data.get('lethargy', 0)),
            int(data.get('appetite', 0)),
            int(data.get('sneezing', 0)),
            int(data.get('rash', 0))
        ]

        # Check for Healthy (all symptoms are 0)
        if all(symptom == 0 for symptom in symptoms):
            predicted_label = "Healthy"
            medication_info = {
                'description': 'No medication required. Maintain good nutrition and regular check-ups.',
                'products': []
            }
        else:
            # Encode categorical features
            animal_encoded = animal_encoder.transform([animal])[0]
            gender_encoded = gender_encoder.transform([gender])[0]

            # Combine all features for prediction
            sample = [animal_encoded, age, gender_encoded] + symptoms

            # Predict using the trained Random Forest
            prediction = bagging_predict(trees, sample)
            predicted_label = disease_encoder.inverse_transform([int(prediction)])[0]

            # Fetch medication info from the mapping (safe fallback)
            medication_info = medication_mapping.get(predicted_label, {
                'description': 'No medication information available.',
                'products': []
            })

        # Return the response as JSON
        return jsonify({
            'disease': predicted_label,
            'treatment_description': medication_info['description'],
            'medications': medication_info['products']
        })

    except Exception as e:
        return jsonify({'error': str(e)})




@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("Logged out successfully.", "info")
    return redirect(url_for('login'))


@app.route('/options', methods=['GET'])
def get_options():
    animals = ['Cat', 'Dog', 'Rabbit', 'Parrot', 'Hamster']
    genders = ['Male', 'Female']
    return {'animals': animals, 'genders': genders}


@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', user_email=current_user.email)


@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html', user_email=current_user.email)


@app.route('/medication', methods=['GET'])
def get_medication():
    disease = request.args.get('disease')
    if not disease:
        return jsonify({'error': 'No disease provided.'}), 400
    treatment = medication_mapping.get(disease)
    if not treatment:
        return jsonify({'error': 'No medication info found for this disease.'}), 404
    return jsonify({
        'treatment': treatment['description'],
        'medications': treatment['products']
    })


# --------------------- Main ---------------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
