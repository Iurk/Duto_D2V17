#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 15:51:45 2021

@author: iurk
"""
import re
import yaml
import numpy as np
from os import walk
import utilidades as util
import multiprocessing as mp
import funcoes_graficos as fg
from time import time

def plotting(args):
    idx_file, rho, ux, y = args
    
    fg.grafico(ux[:,256], y, idx_file, pasta_img)
    
if __name__ == '__main__':

    ini = time()
    main = "../bin"
    fileyaml = "../bin/dados.yml"
    velocity = "Velocity"
    
    datafile = open(fileyaml)
    data = yaml.load(datafile, Loader=yaml.FullLoader)
    datafile.close()
    
    Nx = data['domain']['Nx']
    Ny = data['domain']['Ny']
    
    Steps = data['simulation']['NSTEPS']
    Saves = data['simulation']['NSAVE']
    digitos = len(str(Steps))
    
    results = "../bin/Results/"
    pasta_img = util.criar_pasta('Teste_BC', folder=velocity, main_root=main)
    # pasta_stream = util.criar_pasta('Stream', folder=velocity, main_root=main)
    
    rho_files = []
    ux_files = []
    uy_files = []
    dic = {"rho": rho_files, "ux":ux_files, "uy":uy_files}
    
    for var in dic.keys():
        path = results + var
        
        for root, dirs, files in walk(path):
            for file in sorted(files):
                path_full = path + "/%s" % file
                dic[var].append(path_full)
                
    x = np.arange(1, Nx+1, 1)
    y = np.arange(1, Ny+1, 1)
    
    CPU = mp.cpu_count()
    pool = mp.Pool()
    
    idx = []
    rhos = np.empty((CPU, Ny, Nx))
    uxs = np.empty_like(rhos)
    uys = np.empty_like(rhos)
    ys = np.empty((CPU, Ny))
    
    i = 0
    pattern = r'\d{%d}' % len(str(Steps))
    
    print("Reading and plotting data...")
    while(i < len(rho_files)):
        for j in range(CPU):
            idx.append(re.search(pattern, rho_files[i]).group(0))
            rhos[j] = np.fromfile(rho_files[i], dtype='float64').reshape(Ny, Nx)
            uxs[j] = np.fromfile(ux_files[i], dtype='float64').reshape(Ny, Nx)
            uys[j] = np.fromfile(uy_files[i], dtype='float64').reshape(Ny, Nx)
            ys[j] = np.arange(1, Ny+1, 1)
            
            i += 1
            if(i == len(rho_files)):
                break
        
        inputs = zip(idx, rhos, uxs, ys)
        pool.map(plotting, inputs)
        idx = []
        
    print('Animating...')
    fg.animation('Velocidade', './', pasta_img)
    print('Done!')
    fim = time()
    print("Finish in {} s".format(fim - ini))