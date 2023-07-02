from rdkit import Chem
from rdkit.Chem import Draw

m = 'COC[C@H](C)N1CCN([C@H](C)C(=O)c2c(C)[nH]c3ccccc23)CC1'
m = Chem.MolFromSmiles(m)
f = 'data/picture/5.png'
Draw.MolToFile(m, f, size=(150, 100))