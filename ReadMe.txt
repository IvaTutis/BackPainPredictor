Članovi projekta

Iva Tutiš - iva.tutis@student.math.hr, ivach14@yahoo.com
Dario Bogović - dario.bogovic@student.math.hr
Fran Borić - fran.boric@student.math.hr, fboric@yahoo.com

Tema - Predviđanje pojave boli u lumbarnom dijelu kralježnice strojnim učenjem

------------------------------------------------------------------------UVOD---------------------------------------------------------------

Ovo je kratka dokumentacija za kod u datoteci Zavrsni_kod_Sarsorama.py, za projekt grupe Sarsorama u sklopu predmeta Strojno Ucenje na PMF-MO 2019/2020.

Primarni cilj projekta je predviđanje potencijala za pojavu boli u lumbarnom dijelu kralježnice (omogućiti korisniku unos tih podataka, te rezultat), na osnovu modela treniranog nadziranim učenjem nad skupom podataka koji se nalazi u datoteci Dataset_spine.csv.

Sekundarni cilj projekta je uspješna (hardcoded) implementacija TZV. Partial Component Analysisa nad danim, prethodno normaliziranim, podacima. 

Program se pokrece pokretanjem Zavrsni_kod_Sarsorama.py datoteke, bilo kroz neki IDE, bilo direktno kroz terminal/konzolu.

-----------------------------------------------------------------------STRUKTURA KODA-----------------------------------------------------
>unos relevantnih biblioteka: numpy, sklearn, pandas, matplotlib.pyplot, seaborn, os
>unos podataka i njihova normalizacija
>implementacija PCA
>usporedba naše implementacije sa PCA funkcijom iz biblioteke sklearn
>definicija algoritma (pipelinea): Normalizacija podataka -> PCA -> Logistička regresija
>definicija broja komponenti koji uzimamo u obzir nakon PCA analize (components_to_evaluate,components_to_evaluate2) za odgovarajuće modele 
>konstrukcija dva modela nadziranog ucenja (model, model2) krosvalidacijom+pipelineom s odgovarajucim brojem komponenti koji se uzima u obzir nakon PCA analize
    pri cemu je jedina razlika u tome sto je omjer skupova train:test za model jednak 70:30, a za model2 80:20
>kod za ispis grafa modela s obzirom na točnost po danom atributu i podatke u test skupu    
>ispisom relevantne statistike za dobivene modele: točnost (accuracy, score) i matrica konfuzije
>implementacija funkcije is_my_lower_back_going_to_hurt() čijim se pozivom u konzoli mogu upisati kutovi kostiju za jednu osobu/1 redak podataka, te ispisuje predviđanja                     pojave boli u leđima s obzirom na oba modela

--------------------------------------------------------------------KORIŠTENJE BIBLIOTEKA-------------------------------------------
>pandas
-unos podataka

>numpy
-manipulacije podacima u obliku np.array i njihove transformacije (npr. u implementaciji PCA)

>sklearn
-metrics za potrebe prosuđivanja kvalitete predikcija modela
-train_test_split za particiju podataka na test i train podskupove 
-StandardScaler za normalizaciju podataka
-PCA za usporedbu s implementiranim PCA 
-LogisticRegression za logističku regresiju
-accuracy_score za procjenu točnosti
-Pipeline za pipeline
-GridSearchCV za krosvalidaciju
-confusion_matrix, classification_report za ispis grafa i matrice konfuzije

>matplotlib.pyplot
za ispis grafa modela s obzirom na točnost po danom atributu i podatke u test skupu 
za ispis matrice konfuzije 

>seaborn 
za prikaz grafova

>os
za dodatne potrebe inputa/outputa




