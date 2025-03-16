import os
import joblib
import pandas as pd
import keras

def load_model_and_scaler(model_dir):
    model_path = os.path.join(model_dir, 'best_model.keras')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Model or scaler not found in {model_dir}")
    
    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    return model, scaler

def get_valid_input(prompt, valid_values=None, cast_type=str, case_sensitive=True, min_value=None):
    while True:
        user_input = input(prompt)
    
        if not case_sensitive:
            user_input = user_input.strip().upper()
    
        try:
            casted_input = cast_type(user_input)
        except ValueError:
            print(f"Invalid input! Please enter a valid {cast_type.__name__} value.")
            continue
    
        if valid_values is not None and casted_input not in valid_values:
            print(f"Invalid input! Please enter one of the following: {valid_values}")
            continue
        
        if min_value is not None and casted_input < min_value:
            print(f"Invalid input! Please enter a value greater than or equal to {min_value}.")
            continue
        
        return casted_input


def get_user_input():
    print("Please enter the following information to make a prediction:")
    sex = get_valid_input("Sex (0 for male, 1 for female): ", valid_values=[0, 1], cast_type=int)
    age = get_valid_input("Age: ", cast_type=float, min_value=0)
    sibsp = get_valid_input("SibSp (Number of siblings/spouses aboard): ", cast_type=int, min_value=0)
    parch = get_valid_input("Parch (Number of parents/children aboard): ", cast_type=int, min_value=0)
    pclass = get_valid_input("Pclass (1, 2, or 3): ", valid_values=[1, 2, 3], cast_type=int)
    embarked = get_valid_input("Embarked (C for Cherbourg, Q for Queenstown, S for Southampton): ", valid_values=['C', 'Q', 'S'], cast_type=str, case_sensitive=False)
    
    data = {
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Embarked': [embarked]
    }

    return pd.DataFrame(data)


def prepare_data_for_prediction(df, label_encoder, scaler):
    df['Embarked'] = label_encoder.transform(df['Embarked'])
    df_scaled = scaler.transform(df)
    return df_scaled

def main():
    model_dir = 'models/titanic_model_20250316_171021'
    model, scaler = load_model_and_scaler(model_dir)

    user_input_df = get_user_input()
    label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
    input_scaled = prepare_data_for_prediction(user_input_df, label_encoder, scaler)

    prediction = model.predict(input_scaled)
    prediction_value = prediction[0][0]
    if prediction[0] > 0.5:
        print(f"Prediction: Survived - {prediction_value}")
    else:
        print(f"Prediction: Did not survive - {prediction_value}")

if __name__ == "__main__":
    main()
