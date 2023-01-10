from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D,Dense,MaxPooling2D,BatchNormalization,Flatten,Dropout
from PIL import Image
from keras.models import model_from_json
import cv2
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import json
import tensorflow as tf
#p=1;

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        MainWindow.setStyleSheet("background-color:black;")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        #label.setStyleSheet(“background-color: cyan”)
        self.BrowseImage = QtWidgets.QPushButton(self.centralwidget)
        self.BrowseImage.setGeometry(QtCore.QRect(160, 370, 151, 51))
        self.BrowseImage.setObjectName("BrowseImage")
        font_1 = QtGui.QFont()
        font_1.setBold(True)
        font_1.setPointSize(8)
        #font_1.setFamily("Courier New")
        self.BrowseImage.setFont(font_1)
        self.BrowseImage.setStyleSheet("background-color: lightgreen")
        self.imageLbl = QtWidgets.QLabel(self.centralwidget)
        self.imageLbl.setGeometry(QtCore.QRect(200, 80, 361, 261))
        self.imageLbl.setFrameShape(QtWidgets.QFrame.Box)
        self.imageLbl.setText("")
        self.imageLbl.setObjectName("imageLbl")
        self.imageLbl.setStyleSheet("background-color: lightgreen")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        #self.hoge_label.setStyleSheet("QLabel{color : #ff0000}")
        self.label_2.setGeometry(QtCore.QRect(110, 20, 621, 20))
        font = QtGui.QFont()
        font.setFamily("Courier New")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setStyleSheet("background-color: lightgreen")
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.Classify = QtWidgets.QPushButton(self.centralwidget)
        self.Classify.setGeometry(QtCore.QRect(160, 450, 151, 51))
        self.Classify.setObjectName("Classify")
        font_1 = QtGui.QFont()
        font_1.setBold(True)
        font_1.setPointSize(8)
        #font_1.setFamily("Courier New")
        self.Classify.setFont(font_1)
        self.Classify.setStyleSheet("background-color: lightgreen")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(430, 370, 111, 16))
        self.label.setObjectName("label")
        self.label.setStyleSheet("background-color: lightgreen")
        font_1 = QtGui.QFont()
        font_1.setBold(True)
        font_1.setPointSize(8)
        #font_1.setFamily("Courier New")
        self.label.setFont(font_1)
        self.label.setGeometry(QtCore.QRect(400, 370, 300, 51))
        #self.label.setGeometry(QtCore.QRect(300, 420, 151, 51))
        # self.Training = QtWidgets.QPushButton(self.centralwidget)
        # self.Training.setGeometry(QtCore.QRect(400, 450, 151, 51))
        # self.Training.setObjectName("Training")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(400, 450, 300, 51))
        self.textEdit.setObjectName("textEdit")
        self.textEdit.setStyleSheet("background-color: lightgreen")
        #self.textEdit.setStyleSheet("background-color: lightgreen")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        #self.menubar.setStyleSheet("background-color: lightgreen")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.BrowseImage.clicked.connect(self.loadImage)
        self.Classify.clicked.connect(self.classifyFunction)
        #self.Training.clicked.connect(self.trainingFunction)
        
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "PD Detection Window"))
        self.BrowseImage.setText(_translate("MainWindow", "Browse Image"))
        self.label_2.setText(_translate("MainWindow", "             PARKINSON'S DISEASE DETECTION"))
        self.Classify.setText(_translate("MainWindow", "Classify"))
        self.label.setText(_translate("MainWindow", "                         Recognized Class"))
        #self.Training.setText(_translate("MainWindow", "Training"))

    def loadImage(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "D:/NIT Trichy/PD_detection/spiral/testing", "Image Files (*.png *.jpg *jpeg *.bmp);;All Files ()")# Ask for file
        if fileName: # If the user gives a file
            print(fileName)
            self.file=fileName
            pixmap = QtGui.QPixmap(fileName) # Setup pixmap with the provided image
            pixmap = pixmap.scaled(self.imageLbl.width(), self.imageLbl.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
            self.imageLbl.setPixmap(pixmap) # Set the pixmap onto the label
            self.imageLbl.setAlignment(QtCore.Qt.AlignCenter) # Align the label to center

    def classifyFunction(self):
        json_file = open('SpiralModel.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("SpiralModel.h5py")
        print("Loaded model from disk");
        label=["Healthy", "Parkinson's disease" ]
        path2=self.file
        print(path2)

        pixmap = QtGui.QPixmap(path2) # Setup pixmap with the provided image
        pixmap = pixmap.scaled(self.imageLbl.width(), self.imageLbl.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
        self.imageLbl.setPixmap(pixmap) # Set the pixmap onto the label
        self.imageLbl.setAlignment(QtCore.Qt.AlignCenter) # Align the label to center

        image_healthy = cv2.imread(path2)
        image_healthy = cv2.resize(image_healthy, (128, 128))
        image_healthy = cv2.cvtColor(image_healthy, cv2.COLOR_BGR2GRAY)
        image_healthy = np.array(image_healthy)
        image_healthy = np.expand_dims(image_healthy, axis=0)
        image_healthy = np.expand_dims(image_healthy, axis=-1)
        
        result = loaded_model.predict(image_healthy)
        label2=label[np.argmax(result[0], axis = 0)]
        self.textEdit.setText("                           "+label2)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())