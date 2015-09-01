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

    def __init__(self):
        QWidget.__init__(self)
        self.video_size = QSize(320, 180)
        self.levels = 5
        self.focus = None
        self.setup_ui()
        self.setup_camera()
        N = 320*180*4
        self.resmap = np.array(np.reshape(255-GT.subtended_angle(np.tile(np.arange(320*2),180*2),
                                                 np.repeat(np.arange(180*2), 320*2),
                                                 [320]*N,
                                                 [180]*N,
                                                 320*2,180*2,473.76,296.1,
                                                 [700]*N,[0]*N,[0]*N)/40*255,(180*2,320*2)),dtype=np.uint8)

    def makeImage(self):
        image = Image()
        image.setFixedSize(self.video_size)
        return image

    def addFocus(self, event):
        self.focus = event.pos()

    def removeFocus(self):
        self.focus = None

    def setup_ui(self):
        """Initialize widgets.
        """

        self.image_labels1 = [self.makeImage() for _ in xrange(self.levels)]
        self.image_labels2 = [self.makeImage() for _ in xrange(self.levels)]
        self.image_labels3 = [self.makeImage()]
        self.image_labels1[0].mousePress.connect(self.addFocus)
        self.image_labels3[0].setFixedSize(self.video_size.width()*2,self.video_size.height()*2)

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

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.removeFocus)
        self.quit_button = QPushButton("Quit")
        self.quit_button.clicked.connect(self.close)

        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.layers1)
        self.main_layout.addWidget(self.layers2)
        self.main_layout.addWidget(self.layers3)
        self.main_layout.addWidget(self.reset_button)
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
        frame = cv2.resize(frame, (self.video_size.width(), self.video_size.height()))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #frame = cv2.flip(frame, 1)
        yuv = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)

        G = yuv.copy()
        gpA = [G]
        for i in xrange(self.levels):
            G = cv2.pyrDown(G)
            gpA.append(G)

        gpB = []
        for i in xrange(self.levels):
            G = gpA[i].copy()
            for _ in xrange(i):
                G = cv2.pyrUp(G)
            gpB.append(G)

        for l in xrange(self.levels):
            luma = cv2.merge((gpA[l][:,:,0],gpA[l][:,:,0],gpA[l][:,:,0]))
            if l == 0 and self.focus != None:
                cv2.circle(luma, (self.focus.x(),self.focus.y()), 3, (255, 0, 0))
            image = QImage(luma, luma.shape[1], luma.shape[0],
                            luma.strides[0], QImage.Format_RGB888)
            self.image_labels1[l].setPixmap(QPixmap.fromImage(image))
            luma = cv2.merge((gpB[l][:,:,0],gpB[l][:,:,0],gpB[l][:,:,0]))
            image = QImage(luma, luma.shape[1], luma.shape[0],
                            luma.strides[0], QImage.Format_RGB888)
            self.image_labels2[l].setPixmap(QPixmap.fromImage(image))

        luma = cv2.merge((self.resmap,self.resmap,self.resmap))
        image = QImage(luma, luma.shape[1], luma.shape[0],
                        luma.strides[0], QImage.Format_RGB888)
        self.image_labels3[0].setPixmap(QPixmap.fromImage(image))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainApp()
    win.show()
    win.activateWindow()
    win.raise_()
    sys.exit(app.exec_())
