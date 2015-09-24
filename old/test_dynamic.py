#!/usr/bin/env python

import sys,os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../python"))

import cProfile

from PySide.QtCore import *
from PySide.QtGui import *
import numpy as np
import cv2
import sys

import pyopencl as cl
from gazetools import *

class Image(QLabel):
    mousePress = Signal(QMouseEvent)
    mouseRelease = Signal(QMouseEvent)
    def mousePressEvent(self, event):
        self.mousePress.emit(event)

class MainApp(QWidget):

    def __init__(self):
        QWidget.__init__(self)
        self.setup_camera()
        self.setup_codec()
        self.setup_ui()

    def setup_codec(self):
        _, frame = self.capture.read()
        self.video_size = QSize(frame.shape[1], frame.shape[0])
        self.focus = (frame.shape[1]/2,frame.shape[0]/2)
        ctx = cl.create_some_context(answers=[0,1])
        convolve2d.build(ctx)
        vs_pd = 3.546
        vs_sw = 473.76
        vs_rx = 1680
        ez = 700
        self.rf = RetinaFilter(ctx,frame.shape[1],frame.shape[0],vs_rx,vs_sw,vs_pd,ez)

    def makeImage(self):
        image = Image()
        image.setFixedSize(self.video_size)
        return image

    def addFocus(self, event):
        self.focus = (event.pos().x(),event.pos().y())

    def setup_ui(self):
        """Initialize widgets.
        """

        self.video = self.makeImage()
        self.video.mousePress.connect(self.addFocus)
        self.layers_layout = QHBoxLayout()
        self.layers_layout.addWidget(self.video)
        self.layers = QWidget()
        self.layers.setLayout(self.layers_layout)

        self.quit_button = QPushButton("Quit")
        self.quit_button.clicked.connect(self.close)

        self.main_layout = QVBoxLayout()
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
        self.timer.start(15)

    def display_video_stream(self):
        """Read frame from camera and repaint QLabel widget.
        """
        _, frame = self.capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pyramid = np.array(self.rf.makePyramid(frame))
        frame = self.rf.blend(pyramid,self.focus[0],self.focus[1])
        image = QImage(frame, frame.shape[1], frame.shape[0],
                        frame.strides[0], QImage.Format_RGB888)
        self.video.setPixmap(QPixmap.fromImage(image))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainApp()
    win.show()
    win.activateWindow()
    win.raise_()
    sys.exit(app.exec_())
