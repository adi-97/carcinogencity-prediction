import streamlit as st
import cirpy
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import pandas as pd
import pickle
st.set_page_config(page_title="carcinogenicity_prediction", layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 400px;
        margin-left: -400px;
    }
     
    """,
    unsafe_allow_html=True,
)

#header
st.header('Carcinogenicity')
title = st.text_input('Please enter valid Chemical Name')
cap_button = st.button("Check")
with open("model.pkl","rb") as f:
    rfc=pickle.load(f)
if title not in [""] and cap_button:
    che=Chem.MolFromSmiles(cirpy.resolve(title,'smiles'))
    desc_lst=[i[0] for i in Descriptors._descList]
    rf_feat = ['MaxEStateIndex', 'MinEStateIndex', 'MaxAbsEStateIndex', 'MinAbsEStateIndex', 'qed', 'MolWt', 'HeavyAtomMolWt',
            'ExactMolWt', 'NumValenceElectrons', 'MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge',
           'MinAbsPartialCharge', 'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 'BCUT2D_MWHI',
            'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI',
            'BCUT2D_MRLOW', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v',
            'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'PEOE_VSA1',
            'PEOE_VSA3', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9',
           'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA3', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA2',
            'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'TPSA', 'EState_VSA1', 'EState_VSA10', 'EState_VSA2',
            'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9',
           'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6','VSA_EState7',
           'VSA_EState8', 'FractionCSP3', 'NumHDonors', 'NumRotatableBonds', 'MolLogP', 'MolMR', 'fr_NH0', 'fr_nitroso']
    descriptor=MoleculeDescriptors.MolecularDescriptorCalculator(desc_lst)
    des_che = descriptor.CalcDescriptors(che)
    features = pd.DataFrame([des_che],columns = desc_lst)
    features.fillna(0.0,inplace = True)
    prob = rfc.predict_proba(features[rf_feat])
    if prob[0][-1]>0.5:
        st.write("Carcinogen")
    else:
        st.write("Not a Carcinogen")
