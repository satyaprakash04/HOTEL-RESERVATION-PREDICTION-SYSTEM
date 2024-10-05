from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('hotel_reservation_model.pkl', 'rb') as file:
    classification = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs from the form
    no_of_adults = int(request.form['no_of_adults'])
    no_of_children = int(request.form['no_of_children'])
    no_of_weekend_nights = int(request.form['no_of_weekend_nights'])
    no_of_week_nights = int(request.form['no_of_week_nights'])
    type_of_meal_plan = int(request.form['type_of_meal_plan'])
    required_car_parking_space = int(request.form['required_car_parking_space'])
    room_type_reserved = int(request.form['room_type_reserved'])
    lead_time = int(request.form['lead_time'])
    arrival_month = int(request.form['arrival_month'])
    arrival_date = int(request.form['arrival_date'])
    market_segment_type = int(request.form['market_segment_type'])
    repeated_guest = int(request.form['repeated_guest'])
    no_of_previous_cancellations = int(request.form['no_of_previous_cancellations'])
    no_of_previous_bookings_not_canceled = int(request.form['no_of_previous_bookings_not_canceled'])
    avg_price_per_room = float(request.form['avg_price_per_room'])
    no_of_special_requests = int(request.form['no_of_special_requests'])

    # Create a numpy array with the user inputs
    user_data = np.array([[no_of_adults, no_of_children, no_of_weekend_nights, no_of_week_nights,
                          type_of_meal_plan, required_car_parking_space, room_type_reserved,
                          lead_time, arrival_month, arrival_date, market_segment_type,
                          repeated_guest, no_of_previous_cancellations,
                          no_of_previous_bookings_not_canceled, avg_price_per_room,
                          no_of_special_requests]])

    # Make prediction using the loaded model
    prediction = classification.predict(user_data)[0]

    if prediction == 1:
        result = 'Your reservation is not likely to be cancelled.'
    else:
        result = 'Your reservation is likely to be cancelled.'

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)