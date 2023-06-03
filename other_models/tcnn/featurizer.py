import numpy as np
import csv
from functools import reduce

def string2smiles_list(string):
    char_list = []
    i = 1
    while i < len(string):
        c = string[i]
        if c.islower():
            char_list.append(string[i-1:i+1])
            i += 1
        else:
            char_list.append(string[i-1])
        i += 1
    if not string[-1].islower():
        char_list.append(string[-1])
    return char_list

def onehot_encode(char_list, smiles_string, length):
    encode_row = lambda char: map(int, [c == char for c in smiles_string])
    ans = np.array(map(encode_row, char_list))
    if ans.shape[1] < length:
        residual = np.zeros((len(char_list), length - ans.shape[1]), dtype=np.int8)
        ans = np.concatenate((ans, residual), axis=1)
    return ans

def smiles_to_onehot(smiles, c_chars, c_length):
    c_ndarray = np.ndarray(shape=(len(smiles), len(c_chars), c_length), dtype=np.float32)
    for i in range(len(smiles)):
        c_ndarray[i, ...] = onehot_encode(c_chars, smiles[i], c_length)
    return c_ndarray

def charsets(smiles):
    union = lambda x, y: set(x) | set(y)
    c_chars = list(reduce(union, map(string2smiles_list, list(smiles[:, 2]))))
    i_chars = list(reduce(union, map(string2smiles_list, list(smiles[:, 3]))))
    return c_chars, i_chars

def save_drug_smiles_onehot(smiles):
    # we will abandon isomerics smiles from now on
    c_chars, _ = charsets(smiles)
    c_length = max(map(len, map(string2smiles_list, list(smiles[:, 2]))))
    
    count = smiles.shape[0]
    drug_names = smiles[:, 0].astype(str)
    drug_cids = smiles[:, 1].astype(int)
    smiles = [string2smiles_list(smiles[i, 2]) for i in range(count)]
    
    canonical = smiles_to_onehot(smiles, c_chars, c_length)
    
    save_dict = {}
    save_dict["drug_names"] = drug_names
    save_dict["drug_cids"] = drug_cids
    save_dict["canonical"] = canonical
    save_dict["c_chars"] = c_chars

    print("drug onehot smiles data:")
    print(drug_names.shape)
    print(drug_cids.shape)
    print(canonical.shape)

    np.save(folder + "drug_onehot_smiles.npy", save_dict)
    print("finish saving drug onehot smiles data:")
    return drug_names, drug_cids, canonical