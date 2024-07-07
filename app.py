import streamlit as st
import streamlit.components.v1 as stc
import pandas as pd
import numpy as np
import joblib
import os

# Memuat encoder
encoder = joblib.load('encoded_df2.pkl')

def main():
    st.subheader('Welcome to ML Section')

    song_duration_ms = st.number_input('Song Durations (ms)',0,500000)
    accousticness = st.number_input('accousticness',format="%.6f" )
    danceability = st.number_input('danceability',format="%.3f" )
    energy = st.number_input('energy',format="%.3f" )
    instrumentals = st.number_input('instrumentals',format="%.6f" )
    key = st.selectbox('Key', [1,2,3,4,5,6,7,8,9,10])
    liveness = st.number_input('liveness',format="%.4f" )
    loudness = st.number_input('loudness',format="%.3f" )
    audio_mode = st.selectbox('audio mode', [0,1])
    speechiness = st.number_input('speechiness',format="%.4f" )
    tempo = st.number_input('tempo',format="%.3f" )
    time_signature = st.selectbox('time signature', [0,1,3,4,5])
    av = st.number_input('audio valence',format="%.3f" )

    

    with st.expander("Your Selected Options"):
        result = {
           'song_duration_ms' : song_duration_ms ,
           'accousticness' : accousticness,
           'danceability' : danceability,
           'instrumentals' : instrumentals,
           'key' : key,
           'liveness' : liveness,
           'loudness' : loudness,
           'audio_mode' : audio_mode,
           'speechiness' : speechiness,
           'tempo' : tempo,
           'time_signature' : time_signature,
           'audio valence' : av,

        }
        st.write(result)


    # Convert input to DataFrame for encoding
    input_df = pd.DataFrame([result])

    # Perform the same encoding as training
    input_encoded = pd.get_dummies(input_df, columns=['audio_mode', 'key', 'time_signature'], drop_first=False)
    
    # Align the input with the encoder
    input_encoded = input_encoded.reindex(columns=encoder.columns, fill_value=0)

    st.subheader('Encoded Input DataFrame:')
    st.write(input_encoded)
    single_array = np.array(input_encoded).reshape(1,-1)


    model = joblib.load(open(os.path.join('model_classifier.pkl'), 'rb'))
    prediction = model.predict(single_array)

    # Display prediction result
    st.subheader('Prediction result:')
    st.write(round(prediction[0]))

    # Load your model and make prediction
    # model = joblib.load('model.pkl')
    # prediction = model.predict(input_encoded)
    # st.subheader('Prediction result:')
    # st.write(prediction)

if __name__ == '__main__':
    main()