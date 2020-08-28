import joblib
import os
from os import path
import re

def load_gridsearch(run_num, gs_dir = 'tuning/'):
    dirname = gs_dir+'run'+str(run_num)+'/'
    info_fname = dirname+'run'+str(run_num)+'_info.txt'
    gs_fname = 'gs_'
    if not path.exists(dirname):
        raise OSError(dirname+" does not exist yet.")

    info_file = open(info_fname, 'r')
    # info_txt = info_file.readlines()
    info_dict = {info[0]: info[1] for info in 
                [line.strip('\n').split(": ") 
                for line in info_file.readlines()]}
    # TODO: Convert values to the right types

    gs_fname = dirname+'gs_'
    if bool(info_dict['TEST']):
        gs_fname = gs_fname+'test_'
    gs_fname = gs_fname+'run'+str(run_num)+'.pkl'
    gs = joblib.load(gs_fname)
    return gs, info_dict
