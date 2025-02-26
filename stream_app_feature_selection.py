import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
label_encoder = LabelEncoder()
ordinal_encoder = OrdinalEncoder()
from encoder import MultiColumnLabelEncoder
import pickle
# Custom CSS
st.markdown("""
    <style>
    .main {
       background-image: url('https://example.com/your-image.jpg');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 12px;
    }
    .stTextInput>div>div>input {
        border: 2px solid #4CAF50;
        border-radius: 4px;
    }
    .stSelectbox>div>div>div>div {
        border: 2px solid #4CAF50;
        border-radius: 4px;
    }
    .stMarkdown h2 {
        color: #4CAF50;
    }
    .stMarkdown p {
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

tab2, tab1 = st.tabs(["Prediction", "Feature Selection"])
with tab1:
    
    st.markdown("<h2 style = 'color: orange;'>selecting features for prediction!</h2>", unsafe_allow_html=True)
    st.markdown("""
            <p style='font-size:20px;'>
                In this section, most important features are selected from the best model.
                </p>
                """, unsafe_allow_html=True)

    def select_k_best_features(data, k=10):
        X = data.drop(columns=["hv270"])
        X_final = X[['hv219' ,'hv024','hv025','hv106_01','hv115_01'
                                    ,'hv201','hv237a','hv237b',"hv009","hv220",
                                    'hv237c','hv237d','hv237e','hv237f',
                                    'hv237','hv252','hv230a',"sh121h","sh121j",
                                    "sh121k","sh121l","sh121m","sh121s","sh121n",
                                    "sh122h","hv221","hv243e","sh122i","sh121g", 
                                    "sh121i","sh121p","sh121q","sh121r","hv210",
                                    "hv243c","hv212","hv243d","hv247"]]



        #X_final = X_final.replace("don't know", 0)
        X_final["hv220"]=X_final['hv220'].replace("don't know", 0)
        #X_final["hv204"]=X_final['hv204'].replace("don't know", 0)
        #X_final["hv204"]=X_final['hv204'].replace("on premises", 0)

        X_final = X_final.replace("97+", 97)
        #X_final=X_final.drop("hv204", axis=1)
        X_final['hv220']= X_final['hv220'].astype('int64')

        #if __name__ == '__main__':

        X_final = MultiColumnLabelEncoder(columns = ['hv219', 'hv024','hv025','hv106_01','hv115_01'
                                                    ,'hv201','hv237a','hv237b', 'hv237c','hv237d',
                                                    'hv237e','hv237f',
                                    'hv237','hv252','hv230a',"sh121h","sh121j","sh121k",
                                    "sh121l","sh121m","sh121s","sh121n","sh122h","hv221",
                                    "hv243e","sh122i","sh121g", "sh121i","sh121p","sh121q",
                                    "sh121r","hv210","hv243c","hv212","hv243d","hv247"]).fit_transform(X_final)




        y = data[["hv270"]]
        bestfeatures = SelectKBest(score_func=chi2, k=k)
        # Initialize SelectKBest with the desired scoring function
        fit = bestfeatures.fit(X_final,y)
        dfscores = pd.DataFrame(fit.scores_)
        dfcolumns = pd.DataFrame(X_final.columns)
    #concat two dataframes for better visualization 
        featureScores = pd.concat([dfcolumns,dfscores],axis=1)
        featureScores.columns = ['Specs','Score']  #naming the dataframe columns
        features_best = featureScores.nlargest(10,'Score') 
        return features_best #print 10 best features




    def main():
        st.title('Feature Selection App')

        # File uploader
        uploaded_file = st.file_uploader("Choose a csv file", type="csv")
        
        if uploaded_file is not None:
            # Read the stata file
            data = pd.read_csv(uploaded_file)
            
            # Display the first few rows of the data
            st.write("First few rows of the data:")
            st.write(data.head())

            # Get the list of columns
            columns = data.columns.tolist()

            # Select target column
            target_column = st.selectbox("Select the target column", columns)

            # Select number of features
            k = st.slider("Select number of features to keep", min_value=1, max_value=len(columns)-1, value=10)

            if st.button('Run Feature Selection'):
                # Run the feature selection
                selected_data = select_k_best_features(data, k)
                selected_features = selected_data["Specs"]            
                # Display the selected features
                st.write("Selected Features:")
                st.write(selected_data)

                # Display the data with selected features
                st.write("Data with selected features:")
                st.write(data[selected_features])

    if __name__ == "__main__":
        main()


with tab2:
    st.header("Home")
    st.markdown("<h4 style = 'color: green;'>The aim of this project is to develop and validate a robust statistical multivariable machine learning model to estimate the socio-economic status using trauma registries in Cameroon; including both DHS economic and socio-demographic variables </h4>", unsafe_allow_html=True)
    st.header("Prediction")
    #st.write("Predicting the Socio-economic status!")
    #st.write("In this section, the socio-economic status(poorest, poorer,middle,richer,richest) of a household is assessed with some probability based on the information provided by the user using the selected features.")
    st.markdown("<h2 style='color: blue;'>Predicting the Socio-economic Status!</h2>", unsafe_allow_html=True)
    st.markdown("""
            <p style='font-size:20px;'>
                In this section, the socio-economic status(poorest,poorer,middle,richer,richest) of a household is assessed by attributing a probability of a patient  to belong to each of the aforementioned categories based on the information provided by the user using the selected features presented as follows.
            </p>
        """, unsafe_allow_html=True)
    with open('model.pkl', 'rb') as file:
     model = pickle.load(file)
    mappings = { "Educational level": {"Not educated": 0, "Pre-school": 1, "Primary": 2,"Secondary": 3,"Higher": 4}, 
                "Marital Status": {"Single": 0, "Married": 1,"Widow":2,"Divorced/Separated":3}, 
                "Region of Origin": {"Far-North": 4, "East": 3,"Centre(Without Yaounde)":1,"Yaoundé":11,"adamawa":0,
                                     "South-West":10,"South":9,"West":8,"Douala":2,"Littoral(without Douala)":5,
                                     "North-West":7,"North":6}, 
                "Residential area": {"Urban": 1, "Rural": 0}, 
                "Sex": {"Male": 1, "Female": 0} ,
                "Source of drinking water":{"Tube well or borehole":13,"Pipe to yard or plot":5,"Public tap/ stand pipe":8,"Piped to neighbour":4,
                                            "Unprotected spring":14, "Protected well":7,"Unprotected well":15,"River/dam/lake/ponds/stream/canal/irrigation channel":10,
                                            "Protected spring":6,"Bottled water":0,"Piped into dweeling":3,"Cart with small tank":1,"Sachet water":11,
                                            "Rainwater":9,"Tanker truck":12,"Other":2},
                "Mixer":{"No":0,"Yes":1},
                "Gas stove":{"No":0,"Yes":1},
                "Fan":{"No":0,"Yes":1},
                "Bank account":{"No":0,"Yes":1},
                "Laptop/Computer":{"No":0,"Yes":1},
                "Modem/Router":{"No":0,"Yes":1},
                "Cooker":{"No":0,"Yes":1},
                "CD/DVD Player":{"No":0,"Yes":1},
                "Clock": {"No":0,"Yes":1}
                }
    
    Number_of_Households = st.number_input("Enter the number of household members:",min_value=1, max_value=40,step=1)
    Age_of_head_of_household = st.number_input("Enter the age(in years) of the head of the household:",min_value=18,max_value=80,step=1)
    # Create a dictionary to store the encoded inputs
    encoded_inputs = {}
     # Create a selectbox for each categorical variable
   # Create a selectbox for each categorical variable
    for var, mapping in mappings.items():
        if set(mapping.keys()) == {"Yes", "No"}:
            user_input = st.selectbox(f"Do you have {var}?", list(mapping.keys()))
        else:
            user_input = st.selectbox(f"Select {var}", list(mapping.keys()))
        encoded_inputs[var] = mapping[user_input]
        # Display the encoded inputs 
    #st.write("Encoded inputs:")
    #st.write(encoded_inputs)

    # Combine encoded categorical variables and continuous variables into a single input array
    input_data = np.array([encoded_inputs["Educational level"], encoded_inputs["Marital Status"], encoded_inputs["Region of Origin"], 
                           encoded_inputs["Residential area"], encoded_inputs["Sex"], Number_of_Households, Age_of_head_of_household,
                           encoded_inputs["Source of drinking water"], encoded_inputs["Mixer"], encoded_inputs["Gas stove"],
                           encoded_inputs["Fan"], encoded_inputs["Bank account"], encoded_inputs["Laptop/Computer"], 
                           encoded_inputs["Modem/Router"], encoded_inputs["Cooker"], encoded_inputs["CD/DVD Player"], encoded_inputs["Clock"]
                           ])
    #st.write("Input Data:") 
    #st.write(input_data)
    if st.button("Predict"):
        probabilities = model.predict_proba(input_data)
        # Convert probabilities to percentages
        percentages = probabilities * 100
       # Class names
        class_names = ["Poorest", "Poorer", "Middle", "Richer", "Richest"] 
        # Display class probabilities using circles (progress bars) 
        max_prob_index = np.argmax(probabilities) 
        st.write(f"**Based on the information provided, the household could belong in the '{class_names[max_prob_index]}' category with a chance of {percentages[max_prob_index]:.2f}%.**")
        st.write("Class Probabilities:") 
        for i, percent in enumerate(percentages): 
            st.write(f"{class_names[i]}: {percent:.2f}%") 
            st.progress(int(percent))

    st.write("**Note: The prediction is based on a machine learning model trained on a dataset from the 2018 DHS survey. The model's performance may vary depending on the specific data used.**")
    st.write("**This project is still in its early stages. If you have any questions or need further assistance, please let me know(dongmo20gilles@gmail.com).**")
    st.write("**Thank you for using this app!**")
    st.markdown("""
    <hr>
    <p style='text-align: center; font-size:16px;'>
        This application was designed by Gilles Protais Lekelem Dongmo, and Pr. Dr. Nguefack Tsague Georges.
    </p>
""", unsafe_allow_html=True)
    

#st.markdown("""
   # <hr>
    #<p style='text-align: center; font-size:16px;'>
        #<b>ACKNOWLEDGMENTS</b><br>
        #This research was supported (Award reference number: 2023/P06/D-SINE Africa/D) by the
        #Data Science Center for the Study of Surgery, Injury, and Equity in Africa (D-SINE Africa), 
        #a multi-institutional partnership funded by the National Institutes of Health (NIH, 5U54TW012087) and the Fogarty International Centre (FIC, D43TW009343). 
        #It was implemented by Challenges Initiative Solutions.
        #The authors wish to express their profound appreciation for the support 
        #provided by this organization. The content is solely the responsibility of the authors and
        #does not necessarily represent the official views of D-SINE Africa.
    #</p>
#""", unsafe_allow_html=True)
    

st.markdown("""
    <hr>
    <p style='text-align: center; font-size:16px;'>
        <b>DISCLAIMER</b><br>
         This application is without any warranty. No author accepts responsibility to anyone for the consequences of using it or for whether it serves any particular purpose or works at all
    </p>
""", unsafe_allow_html=True)
    



        