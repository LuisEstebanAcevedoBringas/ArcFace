import cv2
import matplotlib.pyplot as plt
from arcface import ArcFace
face_rec = ArcFace.ArcFace()
########## Funciones ##########
####### Dicionario #######
def get_dictionary(path_img):
    dic = {}
    for i in range(len(path_img)):
        dic[i] = path_img[i]
    return dic
####### Embeddings #######
def get_embeddings(path, path_img):
    Emb = []
    for i in range(len(path_img)):
        emb = face_rec.calc_emb(path + '/' + path_img[i])
        Emb.append(emb)
    return Emb
####### Comparacion de todas las imagenes#######
def get_dist_emb(path_img, Dics,Embs):
    a = []
    for i in range(len(path_img)):
        a.append(i)
    aa = a
    A = []
    B = []
    for i in range(len(a)-1):
        aa.pop(0)
        A.append(aa[:])
        bb = []
        for j in range(len(aa)):
            bb.append(i)
        B.append(bb)  
    Ind = B
    Com = A
    list_ind = []
    list_com = []
    for i in range(len(A)):
        for j in range(len(A[i])):
            Dist = face_rec.get_distance_embeddings(Embs[B[i][j]], Embs[A[i][j]])
            #print('Distancia ', B[i][j], '-', A[i][j], ': ', Dist)
            #print('Distancia ', Dics[B[i][j]], '-', Dics[A[i][j]], ': ', Dist)
            if Dist == 0.0:
                list_ind.append(Dics[B[i][j]])
                list_com.append(Dics[B[i][j]])
                print('Distancia ', B[i][j], '-', A[i][j], ': ', Dist)
                print('Distancia ', Dics[B[i][j]], '-', Dics[A[i][j]], ': ', Dist)
    return list_ind, list_com
####### Guardar y mostrar imagenes####### imp = get_img_show_save(path, path_img)
def get_img_show_save(path, path_img):
    for i in range(len(path_img)):
        img = cv2.imread(path + '/' + path_img[i])
        path_save = 'C:/Users/FPM/Google Drive/Colab Notebooks/arcfacelib/result/' + str(i) + '.jpg'
        cv2.imwrite(path_save, img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #print('Indice: ', i)
        #plt.xticks([]), plt.yticks([])
        #plt.imshow(img, cmap='gray', interpolation='bicubic')
        #plt.show()
    print('Termino de guardar las imagenes con respecto a su indice')