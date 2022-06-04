import os
from ArcFaceUtilsVF import *
path = 'C:/Users/FPM/Google Drive/Colab Notebooks/arcfacelib/prueba' #Path general
path_img = os.listdir(path) #Lista de imagenes
Dics = get_dictionary(path_img)
Embs = get_embeddings(path, path_img)
ind, com = get_dist_emb(path_img, Dics, Embs)