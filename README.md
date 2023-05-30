# carcinogencity-prediction

This repository contains the code to check if a chemical is carcinogenic or not.

The training data was scraped from multiple websites and the corresponding properties of each chemical were calculated using the rdkit cheminformatics library.

The features were selected based on the feature importance of random forest classifier. Based on the model a fastapi and corresponding streamlit app was created.
