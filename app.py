from flask import Flask, render_template, request,jsonify

import pickle
import numpy as np

app = Flask(__name__)

# Load the trained models
#model = pickle.load(open('model.pkl', 'rb'))
model_bengaluru = pickle.load(open('model_bengaluru.pkl', 'rb'))
model_mumbai = pickle.load(open('model_mumbai.pkl', 'rb'))
model_delhi = pickle.load(open('model_delhi.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    location = request.form['location']
    
    if location == 'Bengaluru':
        bath = int(request.form['bath'])
        balcony = int(request.form['balcony'])
        has_hall=int(request.form['Has Hall'])
        has_kitchen=int(request.form['Has Kitchen'])
        num_bedrooms=int(request.form['Number of Bedrooms'])
        data = np.array([[bath,balcony,has_hall,has_kitchen,num_bedrooms]])
        prediction = model_bengaluru.predict(data)
    elif location == 'Mumbai':
        sqrt = int(request.form['sqrt'])
        bhk = int(request.form['bhk'])
        gymnasium = int(request.form['Gymnasium'])
        lift_available = int(request.form['lift_available'])
        car_parking = int(request.form['car_parking'])
        maintenance_staff = int(request.form['maintenance_staff'])
        security = int(request.form['24x7_security'])
        clubhouse = int(request.form['clubhouse'])
        intercom = int(request.form['intercom'])
        landscaped_gardens = int(request.form['landscaped_gardens'])
        indoor_games = int(request.form['indoor_games'])
        gas_connection = int(request.form['gas_connection'])
        jogging_track = int(request.form['jogging_track'])
        swimmimg_pool = int(request.form['swimming_pool'])      
               
        
        
        data = np.array([[sqrt, bhk, gymnasium, lift_available, car_parking, maintenance_staff, security, clubhouse, intercom,landscaped_gardens,
                          indoor_games,gas_connection,jogging_track,swimmimg_pool]])
        prediction = model_mumbai.predict(data)
       
    elif location == 'Delhi NCR':
        area = int(request.form['Area'])
        bhk = int(request.form['BHK'])
        bathroom=int(request.form['Bathroom'])
        parking=int(request.form['Parking'])
        data = np.array([[area, bhk,bathroom,parking]])
        prediction = model_delhi.predict(data)
    return render_template('index.html', data=int(prediction))

# List of locations from the Bengaluru dataset
location_list = [
    "Electronic City Phase II", "Chikka Tirupathi", "Uttarahalli", "Lingadheeranahalli", "Kothanur",
    "Whitefield", "Old Airport Road", "Rajaji Nagar", "Marathahalli", "Gandhi Bazar", "Whitefield",
    "7th Phase JP Nagar", "Gottigere", "Sarjapur", "Mysore Road", "Bisuvanahalli", "Raja Rajeshwari Nagar",
    "Ramakrishnappa Layout", "Manayata Tech Park", "Kengeri", "Binny Pete", "Thanisandra", "Bellandur",
    "Thanisandra", "Mangammanapalya", "Electronic City", "Whitefield", "Ramagondanahalli", "Electronic City",
    "Yelahanka", "Bisuvanahalli", "Hebbal", "Raja Rajeshwari Nagar", "Kasturi Nagar", "Kanakpura Road",
    "Electronics City Phase 1", "Kundalahalli", "Chikkalasandra", "Uttarahalli", "Murugeshpalya", "Sarjapur Road",
    "Ganga Nagar", "Yelahanka", "Kanakpura Road", "HSR Layout", "Doddathoguru", "Whitefield", "KR Puram"
]
@app.route('/suggest_locations', methods=['GET'])
def suggest_locations():
    user_input = request.args.get('input')
    suggestions = suggest_locations(location_list, user_input)
    return jsonify({'suggestions': suggestions})
    
    # Iterate through the location list
def suggest_locations(location_list, user_input):
    suggestions = []
    user_input = user_input.lower()
    for location in location_list:
        if location.lower().startswith(user_input):
            suggestions.append(location)
    return suggestions
if __name__ == '__main__':
    app.run(debug=True)


    


