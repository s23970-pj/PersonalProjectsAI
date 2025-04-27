## 1) Game Bot - Atari Carnival


![0130(1)](https://github.com/user-attachments/assets/1732ffc3-ee5d-47c4-8b2f-ac50c3c5ee95)


![0130(2)](https://github.com/user-attachments/assets/c20ec376-fcd7-4c3b-b08e-b08cd59e09ba)



## 2) GESTURE CONTROLLED SPOTIFY
using Computer vision
<img width="1273" alt="Pasted image 20250111102725" src="https://github.com/user-attachments/assets/4283099d-6161-49c9-937c-baf39fcf367b" />
#Mute gesture open hand horizontally<br>
![0116](https://github.com/user-attachments/assets/62234897-f816-41bd-ac40-85e26fa4099c)<br>
#Pause/Play - pinch sign<br>
![0116(1)](https://github.com/user-attachments/assets/e1a1a116-1533-4786-8ac8-04885e7b63ee)<br>
#Next - Index finger up, Previous - Like sign<br>
![0116(2)](https://github.com/user-attachments/assets/56201b0f-856f-4f69-b9ef-5f55153754fb)<br>




## 3) - Using neural network for classification problems

[cifar10_log.txt](https://github.com/user-attachments/files/18204066/cifar10_log.txt)
Confusion matrices: 
![Confusion Matrix - Logistic Regression](https://github.com/user-attachments/assets/f93ec3c3-bc32-4ff2-ab57-b1f6604c3db1)
![Confusion_Matrix_Animals](https://github.com/user-attachments/assets/316e4350-22d5-41f0-b9d3-8d168007e4a6)
![Confusion Matrix - Neural Network](https://github.com/user-attachments/assets/553e1e03-cd6c-484d-8f1b-b5042772572e)

Run reports:
![Run_Report_IonosphereCNN](https://github.com/user-attachments/assets/440d1575-97eb-4ddd-a06c-67cee84901b1)

![Run_Report_IonosphereCNN](https://github.com/user-attachments/assets/bb22aa79-47f4-4241-96e7-0d49a393ee28)
![Run_Report_FashionMnist](https://github.com/user-attachments/assets/0a48fc5e-cc33-4eef-af71-210a5f4690c8)
![Run_Report_Animals](https://github.com/user-attachments/assets/de6457bc-31db-47b4-ba03-659416583d72)
![Run_Report_IsThatSanta](https://github.com/user-attachments/assets/c30e6580-8825-4c28-bb05-dc83c2498c03)

Some plots and logs:
![stars_plot](https://github.com/user-attachments/assets/6c2d592a-4312-4063-89c4-90e4afce71b0)
:[stars_log.txt](https://github.com/user-attachments/files/18204067/stars_log.txt)
![cifar10_plot](https://github.com/user-attachments/assets/6f22b004-d117-4064-b1a6-16891f1b4ef6)

[UplCIFAR-10:
     0    1    2    3    4    5    6  ...  3066  3067  3068  3069  3070  3071  label
0   59   62   63   43   46   45   50  ...   151   118    84   123    92    72      6
1  154  177  187  126  137  136  105  ...   143   134   142   143   133   144      9
2  255  255  255  253  253  253  253  ...    79    85    83    80    86    84      9
3   28   25   10   37   34   19   38  ...    63    56    37    72    65    46      4
4  170  180  198  168  178  196  177  ...    71    75    78    73    77    80      1

[5 rows x 3073 columns]

Epoch 1/50
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 9s 4ms/step - accuracy: 0.3137 - loss: 2.0463 - val_accuracy: 0.4026 - val_loss: 1.6750
.
.
.
Test accuracy: 0.47

Stars:
          u         g         r         i         z  redshift   class
0  23.87882  22.27530  20.39501  19.16573  18.79371  0.634794  GALAXY
1  24.77759  22.83188  22.58444  21.16812  21.61427  0.779136  GALAXY
2  25.26307  22.66389  20.60976  19.34857  18.94827  0.644195  GALAXY
3  22.13682  23.77656  21.61162  20.50454  19.25010  0.932346  GALAXY
4  19.43718  17.58028  16.49747  15.97711  15.54461  0.116123  GALAXY

Epoch 1/50
4000/4000 ━━━━━━━━━━━━━━━━━━━━ 10s 2ms/step - accuracy: 0.8992 - loss: 0.3227 - val_accuracy: 0.9596 - val_loss: 0.1449
.
.
.
Test accuracy: 0.97
oading cifar10_log.txt…]()


Databases used:
https://www.kaggle.com/datasets/borhanitrash/animal-image-classification-dataset/data
https://www.kaggle.com/datasets/deepcontractor/is-that-santa-image-classification/code
https://machinelearningmastery.com/standard-machine-learning-datasets/ --> IONOSPHERE DATABASE

###2) Data set classification using decision tree and SVM with visualization.
Using two Datasets stars and ionosphere:
-Run instructions at the beginning of code
-Below are effects of RUN: 

![Run_Report_ionosphere](https://github.com/user-attachments/assets/011271a1-19be-4790-99b6-0e029ad2f469)
![Run_Report_stars](https://github.com/user-attachments/assets/85e8fe33-dae6-4ec4-8614-5b85c353c809)



#### STARS decision tree based on 100,000 observations of space taken by the SDSS (Sloan Digital Sky Survey)
![decision_tree_stars](https://github.com/user-attachments/assets/29e3b43d-d944-4c2d-bd7e-6861fb998fe9)

#### Ionosphere 
![decision_tree_ionosphere](https://github.com/user-attachments/assets/adec7c71-d841-4d9d-83a9-9ac99a56d9db)

### 4) MATRIX GAME ALL INSTRUCTIONS IN THE FOLDER


### 5) FUZZY AUTOLAND SIMULATOR
Instructions in Polish 
Na początku kodu dodaj opis problemu; wymień autorów rozwiązania; dodaj instrukcję przygotowania środowisk.

AUTORZY: ADRIAN GOIK, ŁUKASZ SOLDATKE
OPIS PROBLEMU: Symulator automatycznego lądowania 2D przy użyciu logiki rozmytej (fuzzy logic) implementujący zmiany
w czasie rzeczywistym bazujące na wysokości, Ground Speed (prędkości postępowej względem terenu) oraz odległości
strefy przyziemienia.

PRZYGOTOWANIE ŚRODOWISKA:
1. Pobrać kod źródłowy main
2. Zainstalować biblioteki numpy, skfuzzy, pygame oraz packaging
pip install packaging
pip install networkx
(SPRAWDŹ CZY MASZ ZAINSTALOWANE NARZĘDZIE PIP)
3. Uruchomić program

   #### Screenshot of working program
   
![Autoland1](https://github.com/user-attachments/assets/7f31db2d-7b16-4ad2-8cc7-300f2d65b156)
![Autoland2](https://github.com/user-attachments/assets/b823e2fc-a3a9-41fd-98ad-58f36267c58d)


![autolanding_simulation](https://github.com/user-attachments/assets/e3134abc-1d32-4775-bbd9-d3ed7d7a184a)
