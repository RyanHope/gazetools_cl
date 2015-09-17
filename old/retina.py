#!/usr/bin/env python

from PySide.QtCore import *
from PySide.QtGui import *
import numpy as np
import cv2
import sys

from gazetools import *
GT = gazetools()

class Image(QLabel):
    mousePress = Signal(QMouseEvent)
    mouseRelease = Signal(QMouseEvent)
    def mousePressEvent(self, event):
        self.mousePress.emit(event)

class MainApp(QWidget):

    EZ = 700.0
    SW = 473.76
    SH = 296.1

    def __init__(self):
        QWidget.__init__(self)
        self.reduce_factor = 3
        self.levels = 6
        self.setup_camera()
        self.setup_codec()
        self.setup_ui()

    def setup_codec(self):
        _, frame = self.capture.read()
        self.RX = frame.shape[1]
        self.RY = frame.shape[0]
        self.video_size = QSize(self.RX/self.reduce_factor, self.RY/self.reduce_factor)
        self.focus = (self.RX/self.reduce_factor/2, self.RY/self.reduce_factor/2)
        self.decay_constant = 0.106
        self.halfres_eccentricity = 2.3
        self.contrast_sensitivity = 1.0/64.0
        self.critical_eccentricity = [0.0]
        horizontal_degree = GT.subtended_angle([0],[self.RX/2],[self.RX],[self.RY/2],self.RX,self.RY,self.SW,self.SH,[self.EZ],[0],[0])[0]
        #viewing_distance = (self.RX/2)/np.tan(np.pi*horizontal_degree/360);
        freq = 0.5/(horizontal_degree/self.RX)
        for l in xrange(self.levels):
            ecc = self.halfres_eccentricity * ( (np.log(1/self.contrast_sensitivity)*(1<<l)/(self.decay_constant*freq))-1 )
            if ecc > 90.0: ecc = 90.0
            self.critical_eccentricity.append(ecc)
        self.critical_eccentricity.append(90.0)
        # self.fovea_threshold = np.tan(self.critical_eccentricity[1]*np.pi/180 )*viewing_distance

        w = self.RX*2
        h = self.RY*2
        n = len(range(0,w)*h)
        self.ecc = GT.subtended_angle(np.tile(np.arange(w),h), np.repeat(np.arange(h), w), [w/2]*n, [h/2]*n, self.RX, self.RY, self.SW, self.SH, [self.EZ]*n,[0]*n,[0]*n)
        self.resmap = np.copy(self.ecc)
        self.resmap, self.blendmap, self.lmap = GT.resmap(self.resmap, self.critical_eccentricity)
        self.ecc = np.reshape(self.ecc,(h,w))
        self.resmap = np.reshape(self.resmap,(h,w))
        self.blendmap = np.reshape(self.blendmap,(h,w))
        self.lmap = np.reshape(self.lmap,(h,w))

    def makeImage(self):
        image = Image()
        image.setFixedSize(self.video_size)
        return image

    def addFocus(self, event):
        self.focus = (event.pos().x(),event.pos().y())
        print self.focus

    def setup_ui(self):
        """Initialize widgets.
        """

        self.image_labels = [self.makeImage() for _ in xrange(2)]
        self.image_labels[0].mousePress.connect(self.addFocus)
        self.image_labels1 = [self.makeImage() for _ in xrange(self.levels)]
        self.image_labels2 = [self.makeImage() for _ in xrange(self.levels)]
        self.image_labels3 = [self.makeImage() for _ in xrange(3)]
        self.image_labels3[0].setFixedSize(self.video_size.width()*2,self.video_size.height()*2)
        self.image_labels3[1].setFixedSize(self.video_size.width()*2,self.video_size.height()*2)
        self.image_labels3[2].setFixedSize(self.video_size.width()*2,self.video_size.height()*2)

        self.layers_layout = QHBoxLayout()
        for i in xrange(2):
            self.layers_layout.addWidget(self.image_labels[i])
        self.layers = QWidget()
        self.layers.setLayout(self.layers_layout)
        self.layers1_layout = QHBoxLayout()
        self.layers2_layout = QHBoxLayout()
        self.layers3_layout = QHBoxLayout()
        for i in xrange(self.levels):
            self.layers1_layout.addWidget(self.image_labels1[i])
            self.layers2_layout.addWidget(self.image_labels2[i])
        for l in self.image_labels3:
            self.layers3_layout.addWidget(l)
        self.layers1 = QWidget()
        self.layers1.setLayout(self.layers1_layout)
        self.layers2 = QWidget()
        self.layers2.setLayout(self.layers2_layout)
        self.layers3 = QWidget()
        self.layers3.setLayout(self.layers3_layout)

        self.quit_button = QPushButton("Quit")
        self.quit_button.clicked.connect(self.close)

        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.layers1)
        self.main_layout.addWidget(self.layers2)
        self.main_layout.addWidget(self.layers3)
        self.main_layout.addWidget(self.layers)
        self.main_layout.addWidget(self.quit_button)

        self.setLayout(self.main_layout)

    def setup_camera(self):
        """Initialize camera.
        """
        self.capture = cv2.VideoCapture("http://wpc.c1a9.edgecastcdn.net/hls-live/20C1A9/cnn/ls_satlink/b_828.m3u8")
        #self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_size.width())
        #self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_size.height())

        self.timer = QTimer()
        self.timer.timeout.connect(self.display_video_stream)
        self.timer.start(30)

    def display_video_stream(self):
        """Read frame from camera and repaint QLabel widget.
        """
        _, frame = self.capture.read()
        #frame = cv2.resize(frame, (self.video_size.width(), self.video_size.height()))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #frame = cv2.flip(frame, 1)
        yuv = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)

        o = int(np.log(frame.shape[1])/np.log(2))
        fs = 1<<o
        if fs < frame.shape[1]: fs *= 2

        G = np.zeros((fs,fs), dtype=np.uint8)
        G[0:yuv[:,:,0].shape[0],0:yuv[:,:,0].shape[1]] = yuv[:,:,0]
        gpA = [G]
        for i in xrange(self.levels):
            G = cv2.pyrDown(G)
            gpA.append(G)

        gpB = []
        for i in xrange(self.levels):
            G = gpA[i].copy()
            for _ in xrange(i):
                G = cv2.pyrUp(G)
            gpB.append(G[0:frame.shape[0],0:frame.shape[1]])
        for i in xrange(self.levels):
            gpA[i] = gpA[i][0:frame.shape[0],0:frame.shape[1]]

        orig = np.reshape(gpB[0], (frame.shape[0],frame.shape[1]))
        luma = cv2.merge((orig,orig,orig))
        luma = cv2.resize(luma, (luma.shape[1]/self.reduce_factor, luma.shape[0]/self.reduce_factor))
        cv2.circle(luma, self.focus, 3, (255, 0, 0))
        image = QImage(luma, luma.shape[1], luma.shape[0], luma.strides[0], QImage.Format_RGB888)
        self.image_labels[0].setPixmap(QPixmap.fromImage(image))

        z = GT.blend(np.vstack(gpB), frame.shape[1], frame.shape[0], self.blendmap, self.lmap, self.focus[0]*self.reduce_factor, self.focus[1]*self.reduce_factor)
        luma =  np.reshape(z, (frame.shape[0],frame.shape[1]))

        luma = cv2.merge((luma,luma,luma))
        luma = cv2.resize(luma, (luma.shape[1]/self.reduce_factor, luma.shape[0]/self.reduce_factor))
        image = QImage(luma, luma.shape[1], luma.shape[0], luma.strides[0], QImage.Format_RGB888)
        self.image_labels[1].setPixmap(QPixmap.fromImage(image))

        for l in xrange(self.levels):
            luma = cv2.merge((gpA[l],gpA[l],gpA[l]))
            luma = cv2.resize(luma, (luma.shape[1]/self.reduce_factor, luma.shape[0]/self.reduce_factor))
            image = QImage(luma, luma.shape[1], luma.shape[0],
                            luma.strides[0], QImage.Format_RGB888)
            self.image_labels1[l].setPixmap(QPixmap.fromImage(image))
            luma = cv2.merge((gpB[l],gpB[l],gpB[l]))
            luma = cv2.resize(luma, (luma.shape[1]/self.reduce_factor, luma.shape[0]/self.reduce_factor))
            image = QImage(luma, luma.shape[1], luma.shape[0],
                            luma.strides[0], QImage.Format_RGB888)
            self.image_labels2[l].setPixmap(QPixmap.fromImage(image))

        ecc = np.uint8(255-self.ecc/90*255)
        luma = cv2.merge((ecc,ecc,ecc))
        luma = cv2.resize(luma, (luma.shape[1]/self.reduce_factor, luma.shape[0]/self.reduce_factor))
        image = QImage(luma, luma.shape[1], luma.shape[0],
                        luma.strides[0], QImage.Format_RGB888)
        self.image_labels3[0].setPixmap(QPixmap.fromImage(image))

        resmap = np.uint8(self.resmap*255)
        luma = cv2.merge((resmap,resmap,resmap))
        luma = cv2.resize(luma, (luma.shape[1]/self.reduce_factor, luma.shape[0]/self.reduce_factor))
        image = QImage(luma, luma.shape[1], luma.shape[0],
                        luma.strides[0], QImage.Format_RGB888)
        self.image_labels3[1].setPixmap(QPixmap.fromImage(image))

        blendmap = np.uint8(self.blendmap*255)
        luma = cv2.merge((blendmap,blendmap,blendmap))
        luma = cv2.resize(luma, (luma.shape[1]/self.reduce_factor, luma.shape[0]/self.reduce_factor))
        image = QImage(luma, luma.shape[1], luma.shape[0],
                        luma.strides[0], QImage.Format_RGB888)
        self.image_labels3[2].setPixmap(QPixmap.fromImage(image))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainApp()
    win.show()
    win.activateWindow()
    win.raise_()
    sys.exit(app.exec_())
