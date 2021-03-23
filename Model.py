import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os



from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report


#----------------------------------------------PRIPREMA PODATAKA, POGLED NA PCA ANALIZU-----------------------------------


# ucitavam dataset
podaci = pd.read_csv("Dataset_spine.csv")
#podaci.info()

# 13-ti column sadrzi metapodatke o datasetu -  to najjednostavnije vidimo via podaci['Unnamed: 13'][:20] - brisemo ga i dajemo imena columnima
podaci.drop(['Unnamed: 13'],axis=1,inplace=True)
podaci.columns = ['pelvic_incidence','pelvic tilt','lumbar_lordosis_angle','sacral_slope','pelvic_radius','degree_spondylolisthesis','pelvic_slope','direct_tilt','thoracic_slope','cervical_tilt','sacrum_angle','scoliosis_slope','State']
#podaci.info()

#dijelim podatke u x (kutovi kraljeznica) i y(target, tj normal-abnormal value) dimenzije
y = podaci.State.values
scaler = StandardScaler()
x = podaci.drop('State',axis=1)
podaci.info()

# standardiziram (normaliziram) podatke u x
scaler = StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)

#---------------------------------------MOJA NAKODIRANA PCA ANALIZA NAD PODACIMA------------------------------------------

#Naši podaci za PCA analizu se sastoje od 310 redaka normaliziranih podataka, u 12 stupaca
#print("Dimenzije originalnih podataka: %s" % str(x_scaled.shape))
assert x_scaled.shape == (310, 12), "Matrica je neočekivanih dimenzija!"

#transponiramo nase podatke, sada x_transpose_scaled predstavlja "matricu uzoraka" 
x_transpose_scaled=np.transpose(x_scaled)
#print("Transponirani podaci: %s" % str(x_transpose_scaled.shape))
assert x_transpose_scaled.shape == (12, 310), "Matrica je neočekivanih dimenzija!"

#računamo prosjek vrijednosti u svakom stupcu, konstruiramo mean_vector kao vektor srednjih vrijednosti
mean_vector=np.zeros((12,1))
for i in range (0,12):
    mean_vector[i,0]=np.mean(x_transpose_scaled[i,:])
print('Vektor srednjih vrijednosti:\n', mean_vector)
print(20 * '_')

#version1: računamo matricu raspršenja
scatter_matrix = np.zeros((12,12))
for i in range(x_transpose_scaled.shape[1]):
    scatter_matrix += (x_transpose_scaled[:,i].reshape(12,1) - mean_vector).dot((x_transpose_scaled[:,i].reshape(12,1) - mean_vector).T)
print('Matrica raspršenja:\n', scatter_matrix)
print(20 * '_')

#version2: računamo matricu kovarijanci (alternativelni način računanja u odnosu na matricu raspršenja)
cov_mat = np.cov([x_transpose_scaled[0,:],x_transpose_scaled[1,:],x_transpose_scaled[2,:],x_transpose_scaled[3,:],x_transpose_scaled[4,:],x_transpose_scaled[5,:],x_transpose_scaled[6,:],x_transpose_scaled[7,:],x_transpose_scaled[8,:],x_transpose_scaled[9,:],x_transpose_scaled[10,:],x_transpose_scaled[11,:]])
print('Matrica kovarijanci:\n', cov_mat)
print(20 * '_')

#Sada trebamo izračunati eigenvrijednosti i eigenvektore 

#version1: preko matrice raspršenja
eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_matrix)

#version2: preko matrice kovarijanci
eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)

#kontrola sustavne greške - oba načina računanja bi trebala davati isti rezultat, inače baca grešku
#(odgovarajući eigenvrijednosti su jednake do na množenje skalarom, a eigenvektori identični)
#ispisujemo sve eigenvektore i eigenvrijednosti 
for i in range(len(eig_val_sc)):
    eigvec_sc = eig_vec_sc[:,i].reshape(1,12).T
    eigvec_cov = eig_vec_cov[:,i].reshape(1,12).T
    assert eigvec_sc.all() == eigvec_cov.all(), 'Eigenvektori nisu isti'

    print('Eigenvektor {}: \n{}'.format(i+1, eigvec_sc))
    print('Eigenvrijednost {} iz matrice raspršenja: {}'.format(i+1, eig_val_sc[i]))
    print('Eigenvrijednost {} iz matrice kovarijanci: {}'.format(i+1, eig_val_cov[i]))
    print('Faktor skaliranja: ', eig_val_sc[i]/eig_val_cov[i])
    print(20 * '_')

#Sortira eigenvektore po padajućim eigenvrijednostima
for ev in eig_vec_sc:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
    #stavljam almost_equal da ne bi došlo do "lažnih" errora radi zaokruživanja

# napravi listu parova (eigenvrijednost, eigenvektor) i sortira ih
eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:,i]) for i in range(len(eig_val_sc))]
eig_pairs.sort(key=lambda x: x[0], reverse=True)
    
#biramo prvih k vrijednosti za našu PCA analizu, ovaj puta je k=12
matrix_w = np.hstack((eig_pairs[0][1].reshape(12,1), eig_pairs[1][1].reshape(12,1), eig_pairs[2][1].reshape(12,1), eig_pairs[3][1].reshape(12,1), eig_pairs[4][1].reshape(12,1),eig_pairs[5][1].reshape(12,1), eig_pairs[6][1].reshape(12,1), eig_pairs[7][1].reshape(12,1), eig_pairs[8][1].reshape(12,1), eig_pairs[9][1].reshape(12,1), eig_pairs[10][1].reshape(12,1), eig_pairs[11][1].reshape(12,1)))
print('Matrica W:\n', matrix_w)
print(20 * '_')

#transformiramo matricu uzorka u novi prostor odredjen matricom w
transformed = matrix_w.T.dot(x_transpose_scaled)
print(transformed.shape)
assert transformed.shape == (12,310), "Matrica je neočekivanih dimenzija!"

#opet transponiramo rezultat, kako bi dobili jednaki izgled podataka kao pri unosu
pca_hardcoded=np.transpose(transformed)

#kod koji printa novi primjer izgleda skupa podataka prije i nakon sto smo primjenili sklearn PCA
print("SKUPOVI (normaliziranih) PODATAKA PRIJE I NAKON PRIMJENE HARDKODIRANOG PCA \n")
podaci_scaled = pd.DataFrame(x_scaled, columns=['ND'+str(i) for i in range(1,x_scaled.shape[1]+1)])
print(podaci_scaled.head())
podaci_pca_hardcode = pd.DataFrame(pca_hardcoded, columns=['PC'+str(i) for i in range(1,pca_hardcoded.shape[1]+1)])
print(podaci_pca_hardcode.head())
print("\n")


#----------------------------------------PCA ANALIZA IZ SKLEARN-A, ZA USPOREDBU--------------------------------------

#radim analizu glavnih komponenti - PCA nad standardniziranim podacima
#projicira nase podatke u novi prostor dobiven stvaranjem novih atributa iz starih ovisno o korelaciji
pca = PCA() 
pca.fit(x_scaled) 
x_pca = pca.transform(x_scaled) 
#print("Dimenzije originalnih podataka: %s" % str(x_scaled.shape))
#print("Dimenzije projiciranih podataka: %s" % str(x_pca.shape))
#IZ OVOG PRINTA SE VIDI DA JE BROJ DIMENZIJA OSTAO ISTI

#kod koji printa novi primjer izgleda skupa podataka prije i nakon sto smo primjenili sklearn PCA
print("SKUPOVI (normaliziranih) PODATAKA PRIJE I NAKON PRIMJENE PCA IZ SKLEARNa \n")
podaci_scaled = pd.DataFrame(x_scaled, columns=['ND'+str(i) for i in range(1,x_scaled.shape[1]+1)])
print(podaci_scaled.head())
podaci_pca = pd.DataFrame(x_pca, columns=['PC'+str(i) for i in range(1,x_pca.shape[1]+1)])
print(podaci_pca.head())
print("\n")

#------------------------------------------USPOREDBA NAŠE IMPLEMENTACIJE PCA SA ONOM IZ SKLEARNa-----------------

print("\n \n Stupci iz naše implementacije PCA moraju biti identični odgovarajućem stupcu iz PCA iz sklearna, do na predznak jer je predznak eigenvektora nebitan. \n")

flag=True
for i in range (0,12):
    if(not(np.allclose(podaci_pca.iloc[:,i], podaci_pca_hardcode.iloc[:,i]) or np.allclose(podaci_pca.iloc[:,i], podaci_pca_hardcode.iloc[:,i]*(-1)))):
        flag=False
        break

if (flag):
    print("Uspješno smo implementirali PCA! \n")
else:
        print("Pogriješili smo pri implementaciji PCA! \n")

#----------------------------------------definiram pipeline, tj algoritam-------------------------------------------

pipe = Pipeline(steps=[('scaler', StandardScaler()),
                       ('pca', PCA()),
                       ('logistic',LogisticRegression(solver='liblinear'))])

#-----------------------------------------IZRADA FUNKCIJE PRI CEMU TRAIN:TEST=70:30-----------------------------------------

#pripremam podatke, analogno kao gore
podaci_train, podaci_test = train_test_split(podaci, test_size=0.30, random_state=42)

X_train = podaci_train.drop('State',axis=1)
y_train = podaci_train['State']

X_test = podaci_test.drop('State',axis=1)
y_test = podaci_test['State']

#ovdje se mogu igrati tako da gledam koliko ce komponenti uzeti u obzir za LogReg
components_to_evaluate = 12

#krosvalidacija
estimator = GridSearchCV(pipe,
                         dict(pca__n_components=range(1,components_to_evaluate)),
                         cv=10,
                         scoring='accuracy')

model = estimator.fit(X_train, y_train)

#plotting regresije
plt.figure(figsize=(7,5))

results_selected = zip(model.cv_results_['params'],model.cv_results_['mean_test_score'],model.cv_results_['std_test_score'])
for params, mean_scores, std_scores in results_selected:
    plt.errorbar(params['pca__n_components'], mean_scores, yerr=std_scores, fmt='ok')

plt.hlines(accuracy_score(y_test, model.predict(X_test)), 0, components_to_evaluate, 
           color='red', linestyle='--', 
           label='tocnost na test setu (' + str(model.best_params_['pca__n_components']) + ' komponenti)')

plt.ylabel('tocnost')
plt.xlabel('broj glavnih komponenti')
plt.title('Unakrsna validacija train:test=70:30 s PCA i logistickom regresijom')
plt.xlim(0.5,components_to_evaluate-0.5)
plt.ylim(0.6,1.05)
plt.xticks(range(1,components_to_evaluate))
plt.legend(loc='lower right')
plt.show()

#STATISTIKA--

#accuracy
print("\n točnost za 70-30 logisticku regresiju:", accuracy_score(y_test, model.predict(X_test)))

#matrica konfuzije
cnf_matrix = confusion_matrix(y_test, model.predict(X_test))
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Matrica konfuzije za model 70-30', y=1.1)
plt.ylabel('Stvarna vrijednost')
plt.xlabel('Predviđena vrijednost')
plt.show()

#-----------------------------------------IZRADA FUNKCIJE PRI CEMU TRAIN:TEST=80:20-----------------------------------------

#pripremam podatke, analogno kao gore
podaci_train, podaci_test = train_test_split(podaci, test_size=0.20, random_state=42)

X_train2 = podaci_train.drop('State',axis=1)
y_train2 = podaci_train['State']

X_test2 = podaci_test.drop('State',axis=1)
y_test2 = podaci_test['State']

#ovdje se mogu igrati tako da gledam koliko ce komponenti uzeti u obzir za LogReg
components_to_evaluate2 = 12

#krosvalidacija
estimator = GridSearchCV(pipe,
                         dict(pca__n_components=range(1,components_to_evaluate2)),
                         cv=10,
                         scoring='accuracy')

model2 = estimator.fit(X_train2, y_train2)

#plotting regresije
plt.figure(figsize=(7,5))

results_selected = zip(model2.cv_results_['params'],model2.cv_results_['mean_test_score'],model2.cv_results_['std_test_score'])
for params, mean_scores, std_scores in results_selected:
    plt.errorbar(params['pca__n_components'], mean_scores, yerr=std_scores, fmt='ok')

plt.hlines(accuracy_score(y_test2, model2.predict(X_test2)), 0, components_to_evaluate2, 
           color='blue', linestyle='--', 
           label='tocnost na test setu (' + str(model.best_params_['pca__n_components']) + ' komponenti)')

plt.ylabel('tocnost')
plt.xlabel('broj glavnih komponenti')
plt.title('Unakrsna validacija train:test=80:20 s PCA i logistickom regresijom')
plt.xlim(0.5,components_to_evaluate2-0.5)
plt.ylim(0.6,1.05)
plt.xticks(range(1,components_to_evaluate2))
plt.legend(loc='lower right')
plt.show()

#STATISTIKA--

#accuracy
print("\n točnost za 80-20 logisticku regresiju:", model2.score(X_test2,y_test2))

#matrica konfuzije
cnf_matrix2 = confusion_matrix(y_test2, model.predict(X_test2))
p2 = sns.heatmap(pd.DataFrame(cnf_matrix2), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Matrica konfuzije za model 80-20', y=1.1)
plt.ylabel('Stvarna vrijednost')
plt.xlabel('Predviđena vrijednost')
plt.show()

#------------------------------------------------------------FUNKCIJA ZA PROCJENU NOVOG INPUTA-------------------------------

def is_my_lower_back_going_to_hurt():
    print("\n \n Dobrodošli u Vaš procjenitelj predispozicije za bol u kralježici! \n Samo upišite kutove kostiju, a mi ćemo procjeniti boli li vas lumbarni dio kralježnice! \n")

    #user input koji hvata grešku ako se upise nesto nekonvertibilno u float
    while True:
        try:
            val0 = float(input("pelvic_incidence: "))
            val1 = float(input("pelvic_tilt: "))
            val2 = float(input("lumbar_lordosis_angle: "))
            val3 = float(input("sacral_slope: "))
            val4 = float(input("pelvic_radius: "))
            val5 = float(input("degree_spondylolisthesis: "))
            val6 = float(input("pelvic_slope: "))
            val7 = float(input("direct_tilt: "))
            val8 = float(input("thoracic_slope: "))
            val9 = float(input("cervical_tilt: "))
            val10 = float(input("sacrum_angle: "))
            val11 = float(input("scoliosis_slope: "))
            break
        except ValueError:
             print("Ups! Niste upisali broj. Probajte opet... \n")

    #prebacujemo to u numpy array
    podatak=np.array([[val0, val1, val2, val3, val4, val5, val6, val7, val8, val9, val10, val11]], dtype=np.float64)
    podatak.reshape(-1,1)

    #predviđam hoće li Vas boljeti kralježnica
    if (model.predict(podatak)==["Abnormal"]):
            print("Prema modelu 70:30 predviđamo da ćete imati bolove u lumbarnom dijelu krallježnice, žao nam je. \n")
    else:
            print("Prema modelu 70:30 predviđamo da nećete imati bolove u lumbarnom dijelu krallježnice, odlično! \n")
    
    if (model2.predict(podatak)==["Abnormal"]):
            print("Prema modelu 80:20 predviđamo da ćete imati bolove u lumbarnom dijelu krallježnice, žao nam je. \n")
    else:
            print("Prema modelu 80:20 predviđamo da nećete imati bolove u lumbarnom dijelu krallježnice, odlično! \n")
    
    #kraj funkcije
    return

