import streamlit as st

import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
label_encoder = LabelEncoder()
ordinal_encoder = OrdinalEncoder()
from encoder import MultiColumnLabelEncoder


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