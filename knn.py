from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from Pre.preprocessor import SimplePreprocessor
from Pre.datasets import SimpleDatasetLoader
from imutils import paths
import cv2
import glob
import argparse


print("Loading images...")

imagePaths = list(paths.list_images('./NEU-CLS-64/'))


sp = SimplePreprocessor(32,32)

sdl = SimpleDatasetLoader(preprocessors=[sp])

(data,labels)=sdl.load(imagePaths,verbose = 500)



data = data.reshape((data.shape[0],3072))

print("[INFO] FEATURES MATRIX : {:.1f}MB".format(data.nbytes/(1024*1000.0)))

le= LabelEncoder()
labels= le.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)


f= open("Report.txt","w+")


## Choice of k is very critical – A small value of k means that noise will have a higher influence on the result.
# A large value make it computationally expensive and kinda defeats the basic philosophy behind KNN (that points
#  that are near might have similar densities or classes )

for i in range(3,10):


    print("[INFO] evaluating k-NN classifier")
    model = KNeighborsClassifier(n_neighbors=i,n_jobs=-1)
    model.fit(trainX,trainY)

    print(classification_report(testY , model.predict(testX),target_names = le.classes_))
    f.write("k = %d\n" % i);
    f.write(classification_report(testY , model.predict(testX),target_names = le.classes_))
    f.write("\n");


f.close
