from flask import Flask,render_template,redirect,request,url_for, send_file
import mysql.connector, joblib, pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


app = Flask(__name__)
app.secret_key = 'medicare' 

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    port="3306",
    database='medicare'
)

mycursor = mydb.cursor()

def executionquery(query,values):
    mycursor.execute(query,values)
    mydb.commit()
    return

def retrivequery1(query,values):
    mycursor.execute(query,values)
    data = mycursor.fetchall()
    return data

def retrivequery2(query):
    mycursor.execute(query)
    data = mycursor.fetchall()
    return data


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        c_password = request.form['c_password']
        if password == c_password:
            query = "SELECT UPPER(email) FROM users"
            email_data = retrivequery2(query)
            email_data_list = []
            for i in email_data:
                email_data_list.append(i[0])
            if email.upper() not in email_data_list:
                query = "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)"
                values = (name, email, password)
                executionquery(query, values)
                return render_template('login.html', message="Successfully Registered!")
            return render_template('register.html', message="This email ID is already exists!")
        return render_template('register.html', message="Conform password is not match!")
    return render_template('register.html')


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        
        query = "SELECT UPPER(email) FROM users"
        email_data = retrivequery2(query)
        email_data_list = []
        for i in email_data:
            email_data_list.append(i[0])

        if email.upper() in email_data_list:
            query = "SELECT UPPER(password) FROM users WHERE email = %s"
            values = (email,)
            password__data = retrivequery1(query, values)
            if password.upper() == password__data[0][0]:
                global user_email
                user_email = email

                return redirect("/home")
            return render_template('login.html', message= "Invalid Password!!")
        return render_template('login.html', message= "This email ID does not exist!")
    return render_template('login.html')


@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/upload', methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        global df
        file = request.files['file']
        df = pd.read_csv(file)
        df1 = df.head(100)  # Preview first 100 rows
        
        return render_template('upload.html', data=df1.to_html(), message="Dataset uploaded successfully! Go to Splitting!")
    return render_template('upload.html')



@app.route('/split', methods=["GET", "POST"])
def split():
    if request.method == "POST":
        global X_train, X_test, y_train, y_test
        split_size = float(request.form['split_size'])

        # Check if 'df' is defined
        if 'df' not in globals() or df.empty:
            return render_template('split.html', message="Please upload a dataset! Go to the upload section!")
        else:
            X = df.drop("target", axis=1)
            y = df["target"]
            # Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size, random_state=42)
            return render_template('model.html', message="Data split successfully! Go for Model selection!")
    return render_template('split.html')


@app.route('/model', methods=["GET", "POST"])
def model():
    if request.method == "POST":
        global algorithm
        algorithm = request.form['algorithm']
        
        if algorithm == "XGBoost":
            model = XGBClassifier()
            
        elif algorithm == "AdaBoost":
            model = AdaBoostClassifier()

        elif algorithm == "LGBM":
            model = LGBMClassifier()

        elif algorithm == "Decision_Tree":
            model = DecisionTreeClassifier()

        elif algorithm == "Logistic_Regression":
            model = LogisticRegression()

        elif algorithm == "Random_Forest":
            model = RandomForestClassifier()

        # Check if the dataset has been split
        if 'X_train' not in globals() or 'y_train' not in globals() or X_train.empty or y_train.empty:
            return render_template('model.html', message="First split the dataset! Go to the Split section!")
        
        else:

            # Train the model
            model.fit(X_train, y_train)

            # Predict on the test set
            y_pred = model.predict(X_test)

            # Calculate accuracy using the test set
            accuracy = accuracy_score(y_test, y_pred)
            accuracy = str(accuracy)[2:4]

            return render_template('model.html', accuracy = accuracy, algorithm = algorithm, message = "Model trained successfully! Go for prediction!")
    return render_template('model.html')



@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        Rndrng_Prvdr_Gndr = float(request.form['Rndrng_Prvdr_Gndr'])
        Rndrng_Prvdr_Ent_Cd = float(request.form['Rndrng_Prvdr_Ent_Cd'])
        Tot_Benes = float(request.form['Tot_Benes'])
        Med_Tot_Benes = float(request.form['Med_Tot_Benes'])
        Bene_Age_65_74_Cnt = float(request.form['Bene_Age_65_74_Cnt'])
        Bene_Age_75_84_Cnt = float(request.form['Bene_Age_75_84_Cnt'])
        Bene_Age_GT_84_Cnt = float(request.form['Bene_Age_GT_84_Cnt'])
        Bene_Feml_Cnt = float(request.form['Bene_Feml_Cnt'])
        Bene_Male_Cnt = float(request.form['Bene_Male_Cnt'])
        Bene_Race_Wht_Cnt = int(request.form['Bene_Race_Wht_Cnt'])

        file1 = "Models/Logistic Regression_model.pkl"
        file2 = "Models/Logistic Regression_selected_features.pkl"
            
        # Load the saved model
        with open(file1, 'rb') as file:
            model = pickle.load(file)

        # Load the selected features
        with open(file2, 'rb') as file:
            selected_feature_names = pickle.load(file)

        # Define a function to preprocess the input data
        def preprocess_input(input_data, selected_features):
            # Create a DataFrame with the input data
            input_df = pd.DataFrame([input_data], columns=selected_features)
            
            # Ensure the DataFrame columns match the selected features
            input_df = input_df[selected_features]
            return input_df


        # Function to test predictions for a list of input data
        def test_predictions(input_data_list):
            for index, input_data in enumerate(input_data_list):
                # Preprocess the input data
                X_input = preprocess_input(input_data, selected_feature_names)
                
                # Make the prediction
                prediction = model.predict(X_input)
                
                # Output the prediction
                if prediction[0] == 0:
                    result = "No Fraud"
                else:
                    result = "Fraud"
            return result

        # Fraud examples based on the provided table
        input_data_fraud = [
            {
                'Rndrng_Prvdr_Gndr': Rndrng_Prvdr_Gndr,
                'Rndrng_Prvdr_Ent_Cd': Rndrng_Prvdr_Ent_Cd,
                'Tot_Benes': Tot_Benes,
                'Med_Tot_Benes': Med_Tot_Benes,
                'Bene_Age_65_74_Cnt': Bene_Age_65_74_Cnt,
                'Bene_Age_75_84_Cnt': Bene_Age_75_84_Cnt,
                'Bene_Age_GT_84_Cnt': Bene_Age_GT_84_Cnt,
                'Bene_Feml_Cnt': Bene_Feml_Cnt,
                'Bene_Male_Cnt': Bene_Male_Cnt,
                'Bene_Race_Wht_Cnt': Bene_Race_Wht_Cnt
            },
        ]

        result = test_predictions(input_data_fraud)
        
        return render_template('prediction.html', prediction = result)
    return render_template('prediction.html')



if __name__ == '__main__':
    app.run(debug = True)