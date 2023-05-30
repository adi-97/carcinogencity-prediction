import cirpy

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.ensemble import RandomForestClassifier
import warnings
import pickle
warnings.filterwarnings('ignore')
import pandas as pd
df = pd.read_excel('cd.xlsx')
df.drop(columns=['ID_v5','  CASRN'],inplace=True)
df['SMILES'].replace(['Cbr'],'CBr',inplace=True)
df=df[df['SMILES'].notna()]


potency={'NP':0,'P':1}
df['Carcinogenic Potency Expressed as P or NP']=df['Carcinogenic Potency Expressed as P or NP'].map(potency).astype('int')

mol_lst=[]

for i in df.SMILES:
    mol=Chem.MolFromSmiles(i)
    mol_lst.append(mol) # Calculation of Mol Objects

desc_lst=[i[0] for i in Descriptors._descList]
descriptor=MoleculeDescriptors.MolecularDescriptorCalculator(desc_lst)
descrs = [] #Calculation of descriptors

for i in range(len(mol_lst)):
    descrs.append(descriptor.CalcDescriptors(mol_lst[i]))
molDes=pd.DataFrame(descrs,columns=desc_lst)

molDes.insert(0,'Name',df['Chemical Name'])

rfc = RandomForestClassifier(max_depth=9, min_samples_leaf=3, min_samples_split=5,
                       n_estimators=50)

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


rfc.fit(molDes[rf_feat],df['Carcinogenic Potency Expressed as P or NP'])
with open('model.pkl', 'wb') as f:
    pickle.dump(rfc, f)
