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


-----------------------------------------------------------------------IZGLED ISPISA-----------------------------------------------------------
>izgled stupaca u datoteci nakon pripreme podataka

---ISPIS VEZAN UZ PCA IMPLEMENTACIJU--
>vektor srednjih vrijednosti
>matrica raspršenja
>matrica kovarijanci
>eigenvektori sa pripadajucim eigenvrijednostima dobiveni racunanjem preko obje metode (preko matrice raspršenja vs matrice kovarijanci)
>matrica W (matrica transformacije za PCA)
>skupovi (normaliziranih) podataka prije i nakon izvršenja implementacije PCA
>skupovi (normaliziranih) podataka prije i nakon izvršenja sklearn PCA
>test jednakosti tih dviju transformacija (tj test valjanosti implementacije)

---ISPIS VEZAN UZ MODELE---
>graf modela s obzirom na točnost po danom atributu i podatke u test skupu
>tocnost i matrica konfuzije za oba modela, krosvalidacijom (nad test podacima)

---(OPCIONALNO) ISPIS NAKON POZIVA FUNKCIJE is_my_lower_back_going_to_hurt()---
>ispis "dobrodošlice" i forma za upis kuteva u (float) obliku 
>ispis prognoze za bol u ledjima ovisno o danom upisu

--------------------------------------------------------------------------PRIMJER  ISPISA-----------------------------------------------------------

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 310 entries, 0 to 309
Data columns (total 13 columns):
 #   Column                    Non-Null Count  Dtype  
---  ------                    --------------  -----  
 0   pelvic_incidence          310 non-null    float64
 1   pelvic tilt               310 non-null    float64
 2   lumbar_lordosis_angle     310 non-null    float64
 3   sacral_slope              310 non-null    float64
 4   pelvic_radius             310 non-null    float64
 5   degree_spondylolisthesis  310 non-null    float64
 6   pelvic_slope              310 non-null    float64
 7   direct_tilt               310 non-null    float64
 8   thoracic_slope            310 non-null    float64
 9   cervical_tilt             310 non-null    float64
 10  sacrum_angle              310 non-null    float64
 11  scoliosis_slope           310 non-null    float64
 12  State                     310 non-null    object 
dtypes: float64(12), object(1)
memory usage: 31.6+ KB
Vektor srednjih vrijednosti:
 [[-8.02225669e-17]
 [ 2.06286601e-16]
 [ 1.60445134e-16]
 [-1.03143300e-16]
 [ 3.78192101e-16]
 [-2.29207334e-17]
 [-2.86509168e-18]
 [ 8.88178420e-17]
 [ 2.52128068e-16]
 [-4.18303385e-16]
 [-9.16829336e-17]
 [-1.14603667e-16]]
____________________
Matrica raspršenja:
 [[ 3.10000000e+02  1.95051620e+02  2.22357533e+02  2.52637597e+02
  -7.67148338e+01  1.98010252e+02  1.34199632e+01 -2.43073454e+01
  -2.78194951e+01  5.19234863e+00  5.96446522e+00 -2.24792151e+00]
 [ 1.95051620e+02  3.10000000e+02  1.34156798e+02  1.93270398e+01
   1.01270220e+01  1.23337307e+02  2.74421613e+00 -2.23569214e+01
  -1.96522324e+01  8.96846219e+00  1.01861576e+01 -1.75306196e+01]
 [ 2.22357533e+02  1.34156798e+02  3.10000000e+02  1.85499936e+02
  -2.49065189e+01  1.65436774e+02  9.14057326e+00 -3.50110221e+01
  -1.97105257e+01  1.98011398e+01  1.77315392e+01 -1.52067031e+01]
 [ 2.52637597e+02  1.93270398e+01  1.85499936e+02  3.10000000e+02
  -1.06059788e+02  1.62302813e+02  1.51863887e+01 -1.45434786e+01
  -2.10700371e+01 -1.94670998e-02  6.40833260e-02  1.01843735e+01]
 [-7.67148338e+01  1.01270220e+01 -2.49065189e+01 -1.06059788e+02
   3.10000000e+02 -8.08015205e+00  4.88782695e+00  1.96641411e+01
   1.87479163e+01 -1.23381156e+01  9.21892930e+00 -9.34480823e+00]
 [ 1.98010252e+02  1.23337307e+02  1.65436774e+02  1.62302813e+02
  -8.08015205e+00  3.10000000e+02  2.66369043e+01 -1.97818493e+01
  -1.77585115e+01  1.75385640e+01  7.13647530e+00 -1.27172067e+01]
 [ 1.34199632e+01  2.74421613e+00  9.14057326e+00  1.51863887e+01
   4.88782695e+00  2.66369043e+01  3.10000000e+02  3.93500124e+00
  -3.64879439e+00  2.72852885e+01  1.88303038e+01 -2.29068410e+01]
 [-2.43073454e+01 -2.23569214e+01 -3.50110221e+01 -1.45434786e+01
   1.96641411e+01 -1.97818493e+01  3.93500124e+00  3.10000000e+02
   3.06502972e+00  2.25488700e+01 -1.16028063e+01 -7.44660118e+00]
 [-2.78194951e+01 -1.96522324e+01 -1.97105257e+01 -2.10700371e+01
   1.87479163e+01 -1.77585115e+01 -3.64879439e+00  3.06502972e+00
   3.10000000e+02  1.62599364e+01  3.51967670e+00  2.95777840e+00]
 [ 5.19234863e+00  8.96846219e+00  1.98011398e+01 -1.94670998e-02
  -1.23381156e+01  1.75385640e+01  2.72852885e+01  2.25488700e+01
   1.62599364e+01  3.10000000e+02  1.78098519e+01  6.58526903e+00]
 [ 5.96446522e+00  1.01861576e+01  1.77315392e+01  6.40833260e-02
   9.21892930e+00  7.13647530e+00  1.88303038e+01 -1.16028063e+01
   3.51967670e+00  1.78098519e+01  3.10000000e+02  4.75458076e+00]
 [-2.24792151e+00 -1.75306196e+01 -1.52067031e+01  1.01843735e+01
  -9.34480823e+00 -1.27172067e+01 -2.29068410e+01 -7.44660118e+00
   2.95777840e+00  6.58526903e+00  4.75458076e+00  3.10000000e+02]]
____________________
Matrica kovarijanci:
 [[ 1.00323625e+00  6.31235016e-01  7.19603666e-01  8.17597400e-01
  -2.48268071e-01  6.40809877e-01  4.34303017e-02 -7.86645481e-02
  -9.00307285e-02  1.68037172e-02  1.93024764e-02 -7.27482689e-03]
 [ 6.31235016e-01  1.00323625e+00  4.34164395e-01  6.25470545e-02
   3.27735339e-02  3.99149860e-01  8.88095835e-03 -7.23524965e-02
  -6.35994575e-02  2.90241495e-02  3.29649112e-02 -5.67333967e-02]
 [ 7.19603666e-01  4.34164395e-01  1.00323625e+00  6.00323418e-01
  -8.06036211e-02  5.35394090e-01  2.95811432e-02 -1.13304279e-01
  -6.37881092e-02  6.40813587e-02  5.73836219e-02 -4.92126313e-02]
 [ 8.17597400e-01  6.25470545e-02  6.00323418e-01  1.00323625e+00
  -3.43235561e-01  5.25251821e-01  4.91468891e-02 -4.70662739e-02
  -6.81878223e-02 -6.30003231e-05  2.07389404e-04  3.29591374e-02]
 [-2.48268071e-01  3.27735339e-02 -8.06036211e-02 -3.43235561e-01
   1.00323625e+00 -2.61493594e-02  1.58182102e-02  6.36379971e-02
   6.06728682e-02 -3.99291768e-02  2.98347227e-02 -3.02420978e-02]
 [ 6.40809877e-01  3.99149860e-01  5.35394090e-01  5.25251821e-01
  -2.61493594e-02  1.00323625e+00  8.62035739e-02 -6.40189298e-02
  -5.74709111e-02  5.67591067e-02  2.30953893e-02 -4.11560088e-02]
 [ 4.34303017e-02  8.88095835e-03  2.95811432e-02  4.91468891e-02
   1.58182102e-02  8.62035739e-02  1.00323625e+00  1.27346319e-02
  -1.18083961e-02  8.83019046e-02  6.09394946e-02 -7.41321715e-02]
 [-7.86645481e-02 -7.23524965e-02 -1.13304279e-01 -4.70662739e-02
   6.36379971e-02 -6.40189298e-02  1.27346319e-02  1.00323625e+00
   9.91919002e-03  7.29736893e-02 -3.75495348e-02 -2.40990329e-02]
 [-9.00307285e-02 -6.35994575e-02 -6.37881092e-02 -6.81878223e-02
   6.06728682e-02 -5.74709111e-02 -1.18083961e-02  9.91919002e-03
   1.00323625e+00  5.26211535e-02  1.13905395e-02  9.57209837e-03]
 [ 1.68037172e-02  2.90241495e-02  6.40813587e-02 -6.30003231e-05
  -3.99291768e-02  5.67591067e-02  8.83019046e-02  7.29736893e-02
   5.26211535e-02  1.00323625e+00  5.76370612e-02  2.13115503e-02]
 [ 1.93024764e-02  3.29649112e-02  5.73836219e-02  2.07389404e-04
   2.98347227e-02  2.30953893e-02  6.09394946e-02 -3.75495348e-02
   1.13905395e-02  5.76370612e-02  1.00323625e+00  1.53869927e-02]
 [-7.27482689e-03 -5.67333967e-02 -4.92126313e-02  3.29591374e-02
  -3.02420978e-02 -4.11560088e-02 -7.41321715e-02 -2.40990329e-02
   9.57209837e-03  2.13115503e-02  1.53869927e-02  1.00323625e+00]]
____________________
Eigenvektor 1: 
[[ 0.53040629]
 [ 0.32227728]
 [ 0.45578068]
 [ 0.44079985]
 [-0.14363085]
 [ 0.42143938]
 [ 0.04407527]
 [-0.07515866]
 [-0.06988346]
 [ 0.03188274]
 [ 0.025719  ]
 [-0.01928425]]
Eigenvrijednost 1 iz matrice raspršenja: 1016.5571071370841
Eigenvrijednost 1 iz matrice kovarijanci: 3.2898288256863597
Faktor skaliranja:  308.9999999999997
____________________
Eigenvektor 2: 
[[ 7.17289820e-01]
 [-4.16492034e-01]
 [-1.69859818e-11]
 [-5.58596187e-01]
 [-4.47897307e-12]
 [ 7.33562542e-12]
 [-3.53314333e-13]
 [ 2.96989411e-12]
 [ 1.20422641e-11]
 [ 8.81908954e-14]
 [ 1.09404460e-11]
 [-2.26267340e-11]]
Eigenvrijednost 2 iz matrice raspršenja: 9.217987499807309e-14
Eigenvrijednost 2 iz matrice kovarijanci: -4.124423012829093e-16
Faktor skaliranja:  -223.49762551354678
____________________
Eigenvektor 3: 
[[-0.41215524]
 [-0.13270763]
 [ 0.67405327]
 [-0.43029848]
 [-0.27762796]
 [ 0.27214718]
 [ 0.02230857]
 [ 0.08207742]
 [ 0.00702032]
 [-0.0975235 ]
 [-0.02614552]
 [ 0.06627744]]
Eigenvrijednost 3 iz matrice raspršenja: 97.04574671339246
Eigenvrijednost 3 iz matrice kovarijanci: 0.3140639052213347
Faktor skaliranja:  309.0000000000001
____________________
Eigenvektor 4: 
[[ 0.10607768]
 [ 0.01552233]
 [ 0.53802317]
 [ 0.12464016]
 [ 0.17094808]
 [-0.80335799]
 [ 0.07673329]
 [ 0.01870134]
 [-0.00563916]
 [ 0.01964528]
 [-0.04706875]
 [ 0.00492237]]
Eigenvrijednost 4 iz matrice raspršenja: 145.07214469983617
Eigenvrijednost 4 iz matrice kovarijanci: 0.46948914142341797
Faktor skaliranja:  309.00000000000006
____________________
Eigenvektor 5: 
[[-0.02531501]
 [ 0.41451647]
 [ 0.08615896]
 [-0.34157234]
 [ 0.62257455]
 [ 0.15735952]
 [ 0.26933278]
 [ 0.09960792]
 [ 0.12118064]
 [ 0.1868549 ]
 [ 0.2445182 ]
 [-0.31500569]]
Eigenvrijednost 5 iz matrice raspršenja: 386.11779438258594
Eigenvrijednost 5 iz matrice kovarijanci: 1.2495721501054597
Faktor skaliranja:  308.9999999999991
____________________
Eigenvektor 6: 
[[-0.12474752]
 [-0.59750974]
 [ 0.18444621]
 [ 0.28531867]
 [ 0.51372137]
 [ 0.23105204]
 [-0.23234572]
 [-0.19583724]
 [-0.17249802]
 [ 0.23387043]
 [-0.04363055]
 [-0.12651877]]
Eigenvrijednost 6 iz matrice raspršenja: 225.2677544831238
Eigenvrijednost 6 iz matrice kovarijanci: 0.7290218591686853
Faktor skaliranja:  309.00000000000006
____________________
Eigenvektor 7: 
[[-0.02902137]
 [-0.27954027]
 [-0.02971153]
 [ 0.17116043]
 [-0.26532868]
 [-0.00790939]
 [ 0.46801256]
 [ 0.30795997]
 [ 0.20285993]
 [ 0.62888099]
 [ 0.25723638]
 [ 0.06070714]]
Eigenvrijednost 7 iz matrice raspršenja: 354.331374122302
Eigenvrijednost 7 iz matrice kovarijanci: 1.1467034761239538
Faktor skaliranja:  309.0000000000002
____________________
Eigenvektor 8: 
[[-0.00806335]
 [ 0.0731055 ]
 [ 0.04807222]
 [-0.06486191]
 [ 0.0761748 ]
 [-0.02071889]
 [-0.23219433]
 [-0.46182065]
 [ 0.33197038]
 [ 0.10916057]
 [ 0.51733055]
 [ 0.56999728]]
Eigenvrijednost 8 iz matrice raspršenja: 327.2191347728565
Eigenvrijednost 8 iz matrice kovarijanci: 1.058961601206654
Faktor skaliranja:  309.00000000000034
____________________
Eigenvektor 9: 
[[ 0.06635884]
 [ 0.11767202]
 [ 0.04103507]
 [-0.00252586]
 [ 0.16753098]
 [ 0.07119817]
 [-0.45243475]
 [ 0.50359849]
 [ 0.42873158]
 [ 0.22712701]
 [-0.42129344]
 [ 0.27518991]]
Eigenvrijednost 9 iz matrice raspršenja: 317.41928913415603
Eigenvrijednost 9 iz matrice kovarijanci: 1.0272468904017995
Faktor skaliranja:  309.0
____________________
Eigenvektor 10: 
[[-0.00638212]
 [-0.1303788 ]
 [ 0.02651101]
 [ 0.08901583]
 [-0.10608369]
 [ 0.00640614]
 [ 0.09677404]
 [-0.3487213 ]
 [ 0.7594564 ]
 [-0.15017592]
 [-0.19390106]
 [-0.44174991]]
Eigenvrijednost 10 iz matrice raspršenja: 299.97465736512055
Eigenvrijednost 10 iz matrice kovarijanci: 0.970791771408157
Faktor skaliranja:  309.00000000000006
____________________
Eigenvektor 11: 
[[-0.01595623]
 [ 0.18580394]
 [ 0.00844775]
 [-0.15902562]
 [-0.32518666]
 [-0.13741979]
 [-0.53451698]
 [-0.15723514]
 [-0.14046182]
 [ 0.50629177]
 [ 0.11080135]
 [-0.46525489]]
Eigenvrijednost 11 iz matrice raspršenja: 274.1543507058345
Eigenvrijednost 11 iz matrice kovarijanci: 0.8872309084331218
Faktor skaliranja:  308.99999999999983
____________________
Eigenvektor 12: 
[[-0.03508816]
 [ 0.16468781]
 [-0.0367651 ]
 [-0.16784852]
 [ 0.01709633]
 [ 0.01753247]
 [ 0.30401683]
 [-0.48055409]
 [-0.1375649 ]
 [ 0.40199306]
 [-0.61121398]
 [ 0.25309833]]
Eigenvrijednost 12 iz matrice raspršenja: 276.84064648370816
Eigenvrijednost 12 iz matrice kovarijanci: 0.895924422277372
Faktor skaliranja:  309.0000000000002
____________________
Matrica W:
 [[ 5.30406294e-01 -2.53150056e-02 -2.90213688e-02 -8.06334975e-03
   6.63588428e-02 -6.38211777e-03 -3.50881637e-02 -1.59562338e-02
  -1.24747517e-01  1.06077684e-01 -4.12155243e-01  7.17289820e-01]
 [ 3.22277278e-01  4.14516474e-01 -2.79540266e-01  7.31055018e-02
   1.17672016e-01 -1.30378797e-01  1.64687815e-01  1.85803940e-01
  -5.97509739e-01  1.55223251e-02 -1.32707626e-01 -4.16492034e-01]
 [ 4.55780681e-01  8.61589633e-02 -2.97115299e-02  4.80722166e-02
   4.10350704e-02  2.65110132e-02 -3.67651016e-02  8.44774900e-03
   1.84446211e-01  5.38023172e-01  6.74053266e-01 -1.69859818e-11]
 [ 4.40799851e-01 -3.41572337e-01  1.71160426e-01 -6.48619140e-02
  -2.52585849e-03  8.90158285e-02 -1.67848524e-01 -1.59025620e-01
   2.85318673e-01  1.24640160e-01 -4.30298481e-01 -5.58596187e-01]
 [-1.43630848e-01  6.22574553e-01 -2.65328675e-01  7.61747956e-02
   1.67530981e-01 -1.06083691e-01  1.70963255e-02 -3.25186661e-01
   5.13721366e-01  1.70948085e-01 -2.77627957e-01 -4.47897307e-12]
 [ 4.21439381e-01  1.57359524e-01 -7.90938807e-03 -2.07188869e-02
   7.11981730e-02  6.40613613e-03  1.75324721e-02 -1.37419788e-01
   2.31052039e-01 -8.03357990e-01  2.72147183e-01  7.33562542e-12]
 [ 4.40752659e-02  2.69332777e-01  4.68012562e-01 -2.32194335e-01
  -4.52434748e-01  9.67740367e-02  3.04016835e-01 -5.34516980e-01
  -2.32345722e-01  7.67332938e-02  2.23085708e-02 -3.53314333e-13]
 [-7.51586584e-02  9.96079154e-02  3.07959972e-01 -4.61820654e-01
   5.03598491e-01 -3.48721303e-01 -4.80554089e-01 -1.57235143e-01
  -1.95837241e-01  1.87013408e-02  8.20774231e-02  2.96989411e-12]
 [-6.98834584e-02  1.21180636e-01  2.02859933e-01  3.31970383e-01
   4.28731578e-01  7.59456403e-01 -1.37564895e-01 -1.40461818e-01
  -1.72498019e-01 -5.63915527e-03  7.02032187e-03  1.20422641e-11]
 [ 3.18827448e-02  1.86854902e-01  6.28880992e-01  1.09160569e-01
   2.27127009e-01 -1.50175917e-01  4.01993056e-01  5.06291766e-01
   2.33870435e-01  1.96452799e-02 -9.75235043e-02  8.81908954e-14]
 [ 2.57189973e-02  2.44518202e-01  2.57236383e-01  5.17330550e-01
  -4.21293441e-01 -1.93901062e-01 -6.11213975e-01  1.10801348e-01
  -4.36305536e-02 -4.70687511e-02 -2.61455211e-02  1.09404460e-11]
 [-1.92842460e-02 -3.15005694e-01  6.07071369e-02  5.69997282e-01
   2.75189907e-01 -4.41749908e-01  2.53098328e-01 -4.65254892e-01
  -1.26518773e-01  4.92236611e-03  6.62774383e-02 -2.26267340e-11]]
____________________
(12, 310)
SKUPOVI (normaliziranih) PODATAKA PRIJE I NAKON PRIMJENE HARDKODIRANOG PCA 

        ND1       ND2       ND3  ...      ND10      ND11      ND12
0  0.147086  0.501369 -0.665177  ...  1.167129 -1.196587  1.712368
1 -1.245864 -0.748769 -1.453001  ...  1.679551 -0.940325 -0.913941
2  0.484370  0.467932 -0.099262  ...  1.635969 -1.227178 -0.615686
3  0.511390  0.711562 -0.411339  ... -0.176157 -1.345020 -0.652989
4 -0.626648 -0.789693 -1.274745  ... -1.059666 -0.190502 -0.069858

[5 rows x 12 columns]
        PC1       PC2       PC3  ...      PC10      PC11          PC12
0 -0.174284 -1.208808  0.984397  ...  0.103074 -0.314809 -3.426243e-11
1 -2.210186  0.075289  0.815071  ... -0.603445 -0.278021  1.920178e-11
2  0.170257 -0.181279  1.289299  ...  0.616817 -0.519912  2.340197e-11
3  0.316767 -0.852278 -0.464238  ...  0.010709 -0.389510 -3.987569e-10
4 -1.573705 -0.770620  0.512878  ... -0.490473 -0.082999  5.600299e-11

[5 rows x 12 columns]


SKUPOVI (normaliziranih) PODATAKA PRIJE I NAKON PRIMJENE PCA IZ SKLEARNa 

        ND1       ND2       ND3  ...      ND10      ND11      ND12
0  0.147086  0.501369 -0.665177  ...  1.167129 -1.196587  1.712368
1 -1.245864 -0.748769 -1.453001  ...  1.679551 -0.940325 -0.913941
2  0.484370  0.467932 -0.099262  ...  1.635969 -1.227178 -0.615686
3  0.511390  0.711562 -0.411339  ... -0.176157 -1.345020 -0.652989
4 -0.626648 -0.789693 -1.274745  ... -1.059666 -0.190502 -0.069858

[5 rows x 12 columns]
        PC1       PC2       PC3  ...      PC10      PC11          PC12
0 -0.174284 -1.208808  0.984397  ... -0.103074 -0.314809 -3.426262e-11
1 -2.210186  0.075289  0.815071  ...  0.603445 -0.278021  1.920128e-11
2  0.170257 -0.181279  1.289299  ... -0.616817 -0.519912  2.340174e-11
3  0.316767 -0.852278 -0.464238  ... -0.010709 -0.389510 -3.987568e-10
4 -1.573705 -0.770620  0.512878  ...  0.490473 -0.082999  5.600287e-11

[5 rows x 12 columns]



 
 Stupci iz naše implementacije PCA moraju biti identični odgovarajućem stupcu iz PCA iz sklearna, do na predznak jer je predznak eigenvektora nebitan. 

Uspješno smo implementirali PCA! 

<plot logističke regresije nastan modelom 70:30 u skočnom prozoru>

 točnost za 70-30 logisticku regresiju: 0.8387096774193549

<matrica konfuzije za model 80:20 u skočnom prozoru>

<plot logističke regresije nastan modelom 70:30 u skočnom prozoru>

 točnost za 80-20 logisticku regresiju: 0.7741935483870968

<matrica konfuzije za model 80:20  u skočnom prozoru>

    UNOS >>> is_my_lower_back_going_to_hurt()

 
 Dobrodošli u Vaš procjenitelj predispozicije za bol u kralježici! 
 Samo upišite kutove kostiju, a mi ćemo procjeniti boli li vas lumbarni dio kralježnice! 

pelvic_incidence: 3
pelvic_tilt: 3
lumbar_lordosis_angle: D
Ups! Niste upisali broj. Probajte opet... 

pelvic_incidence: 1
pelvic_tilt: 55
lumbar_lordosis_angle: 33
sacral_slope: 22
pelvic_radius: 34
degree_spondylolisthesis: 65
pelvic_slope: 23
direct_tilt: 66
thoracic_slope: 23
cervical_tilt: 65
sacrum_angle: 33
scoliosis_slope: 3
Prema modelu 70:30 predviđamo da ćete imati bolove u lumbarnom dijelu krallježnice, žao nam je. 

Prema modelu 80:20 predviđamo da ćete imati bolove u lumbarnom dijelu krallježnice, žao nam je. 


