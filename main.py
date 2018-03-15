#!/usr/bin/env python3

import time
import sys
import threading
import copy
import random
from optparse import OptionParser

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QInputDialog
from PyQt5.QtWidgets import QLabel, QTextEdit, QFrame, QCheckBox
from PyQt5.QtWidgets import QPushButton, QSlider, QHBoxLayout, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor

import gym
import gym_minigrid
from gym_minigrid import minigrid

from model.training import selectAction


class ImgWidget(QLabel):
    """
    Widget to intercept clicks on the full image view
    """

    def __init__(self, window):
        super().__init__()
        self.window = window

    def mousePressEvent(self, event):
        self.window.imageClick(event.x(), event.y())


class AIGameWindow(QMainWindow):
    """Application window for the baby AI game"""

    def __init__(self, env):
        super().__init__()
        self.initUI()

        # By default, manual stepping only
        self.fpsLimit = 0

        self.env = env
        self.lastObs = None

        self.resetEnv()

        self.stepTimer = QTimer()
        self.stepTimer.setInterval(0)
        self.stepTimer.setSingleShot(False)
        self.stepTimer.timeout.connect(self.stepClicked)

        # Pointing and naming data
        self.pointingData = []

        # Demonstration data
        self.curDemo = None
        self.demos = []

    def initUI(self):
        """Create and connect the UI elements"""

        self.resize(512, 512)
        self.setWindowTitle('Baby AI Game')

        # Full render view (large view)
        self.imgLabel = ImgWidget(self)
        self.imgLabel.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        leftBox = QVBoxLayout()
        leftBox.addStretch(1)
        leftBox.addWidget(self.imgLabel)
        leftBox.addStretch(1)

        # Area on the right of the large view
        rightBox = self.createRightArea()

        # Arrange widgets horizontally
        hbox = QHBoxLayout()
        hbox.addLayout(leftBox)
        hbox.addLayout(rightBox)

        # Create a main widget for the window
        mainWidget = QWidget(self)
        self.setCentralWidget(mainWidget)
        mainWidget.setLayout(hbox)

        # Show the application window
        self.show()
        self.setFocus()

    def createRightArea(self):
        # Agent render view (partially observable)
        self.obsImgLabel = QLabel()
        self.obsImgLabel.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        miniViewBox = QHBoxLayout()
        miniViewBox.addStretch(1)
        miniViewBox.addWidget(self.obsImgLabel)
        miniViewBox.addStretch(1)

        self.stepsLabel = QLabel()
        self.stepsLabel.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.stepsLabel.setAlignment(Qt.AlignCenter)
        self.stepsLabel.setMinimumSize(60, 10)
        resetBtn = QPushButton("Reset")
        resetBtn.clicked.connect(self.resetEnv)
        stepsBox = QHBoxLayout()
        stepsBox.addStretch(1)
        stepsBox.addWidget(QLabel("Steps remaining"))
        stepsBox.addWidget(self.stepsLabel)
        stepsBox.addWidget(resetBtn)
        stepsBox.addStretch(1)

        hline1 = QFrame()
        hline1.setFrameShape(QFrame.HLine)
        hline1.setFrameShadow(QFrame.Sunken)

        teachBox = QHBoxLayout()
        self.testSetBox = QCheckBox('Keep in test set', self)
        startDemoBtn = QPushButton("Start demo")
        startDemoBtn.clicked.connect(self.startDemo)
        endDemoBtn = QPushButton("End demo")
        endDemoBtn.clicked.connect(self.endDemo)
        teachBox.addWidget(self.testSetBox)
        teachBox.addWidget(startDemoBtn)
        teachBox.addWidget(endDemoBtn)

        hline2 = QFrame()
        hline2.setFrameShape(QFrame.HLine)
        hline2.setFrameShadow(QFrame.Sunken)

        self.missionBox = QTextEdit()
        self.missionBox.setMinimumSize(500, 100)
        self.missionBox.textChanged.connect(self.missionEdit)

        buttonBox = self.createButtons()

        # Stack everything up in a vetical layout
        vbox = QVBoxLayout()
        vbox.addLayout(miniViewBox)
        vbox.addLayout(stepsBox)
        vbox.addWidget(hline1)
        vbox.addLayout(teachBox)
        vbox.addWidget(hline2)
        vbox.addWidget(QLabel("Mission"))
        vbox.addWidget(self.missionBox)
        vbox.addLayout(buttonBox)

        return vbox

    def createButtons(self):
        """Create the row of UI buttons"""

        stepButton = QPushButton("Step")
        stepButton.clicked.connect(self.stepClicked)

        minusButton = QPushButton("- Reward")
        minusButton.clicked.connect(self.minusReward)

        plusButton = QPushButton("+ Reward")
        plusButton.clicked.connect(self.plusReward)

        slider = QSlider(Qt.Horizontal, self)
        slider.setFocusPolicy(Qt.NoFocus)
        slider.setMinimum(0)
        slider.setMaximum(100)
        slider.setValue(0)
        slider.valueChanged.connect(self.setFrameRate)

        self.fpsLabel = QLabel("Manual")
        self.fpsLabel.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.fpsLabel.setAlignment(Qt.AlignCenter)
        self.fpsLabel.setMinimumSize(80, 10)

        # Assemble the buttons into a horizontal layout
        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(stepButton)
        hbox.addWidget(slider)
        hbox.addWidget(self.fpsLabel)
        hbox.addStretch(1)
        hbox.addWidget(minusButton)
        hbox.addWidget(plusButton)
        hbox.addStretch(1)

        return hbox

    def keyPressEvent(self, e):
        # Manual agent control
        actions = self.env.unwrapped.actions
        if e.key() == Qt.Key_Left:
            self.stepEnv(actions.left)
        elif e.key() == Qt.Key_Right:
            self.stepEnv(actions.right)
        elif e.key() == Qt.Key_Up:
            self.stepEnv(actions.forward)
        elif e.key() == Qt.Key_Space:
            self.stepEnv(actions.toggle)

        elif e.key() == Qt.Key_Backspace:
            self.resetEnv()
        elif e.key() == Qt.Key_Escape:
            self.close()

    def mousePressEvent(self, event):
        """
        Clear the focus of the text boxes and buttons if somewhere
        else on the window is clicked
        """

        # Set the focus on the full render image
        self.imgLabel.setFocus()

        QMainWindow.mousePressEvent(self, event)

    def imageClick(self, x, y):
        """
        Pointing and naming logic
        """

        # Set the focus on the full render image
        self.imgLabel.setFocus()

        env = self.env.unwrapped
        imgW = self.imgLabel.size().width()
        imgH = self.imgLabel.size().height()

        i = (env.grid.width * x) // imgW
        j = (env.grid.height * y) // imgH
        assert i < env.grid.width
        assert j < env.grid.height

        print('grid clicked: i=%d, j=%d' % (i, j))

        desc, ok = QInputDialog.getText(
            self, 'Pointing & Naming', 'Enter Description:')
        desc = str(desc)

        if not ok or len(desc) == 0:
            return

        pointObj = env.grid.get(i, j)

        if pointObj is None:
            return

        print('description: "%s"' % desc)
        print('object: %s %s' % (pointObj.color, pointObj.type))

        viewSz = minigrid.AGENT_VIEW_SIZE

        NUM_TARGET = 50
        numItrs = 0
        numPos = 0
        numNeg = 0

        while (numPos < NUM_TARGET or numNeg < NUM_TARGET) and numItrs < 300:
            env2 = copy.deepcopy(env)

            # Randomly place the agent around the selected point
            x, y = i, j
            x += random.randint(-viewSz, viewSz)
            y += random.randint(-viewSz, viewSz)
            x = max(0, min(x, env2.grid.width - 1))
            y = max(0, min(y, env2.grid.height - 1))
            env2.agentPos = (x, y)
            env2.agentDir = random.randint(0, 3)

            # Don't want to place the agent on top of something
            if env2.grid.get(*env2.agentPos) != None:
                continue

            agentSees = env2.agentSees(i, j)

            obs, _, _, _ = env2.step(env2.actions.wait)
            img = obs['image'] if isinstance(obs, dict) else obs
            obsGrid = minigrid.Grid.decode(img)

            datum = {
                'desc': desc,
                'img': img,
                'pos': (i, j),
                'present': agentSees
            }

            if agentSees and numPos < NUM_TARGET:
                self.pointingData.append(datum)
                numPos += 1

            if not agentSees and numNeg < NUM_TARGET:
                # Don't want identical object in mismatch examples
                if (pointObj.color, pointObj.type) not in obsGrid:
                    self.pointingData.append(datum)
                    numNeg += 1

            numItrs += 1

        print('positive examples: %d' % numPos)
        print('negative examples: %d' % numNeg)
        print('total examples: %d' % len(self.pointingData))

    def missionEdit(self):
        # The agent will get the mission as an observation
        # before performing the next action
        text = self.missionBox.toPlainText()
        self.lastObs['mission'] = text

    def plusReward(self):
        print('+reward')
        self.env.setReward(1)

    def minusReward(self):
        print('-reward')
        self.env.setReward(-1)

    def stepClicked(self):
        self.stepEnv(action=None)

    def setFrameRate(self, value):
        """Set the frame rate limit. Zero for manual stepping."""

        print('Set frame rate: %s' % value)

        self.fpsLimit = int(value)

        if value == 0:
            self.fpsLabel.setText("Manual")
            self.stepTimer.stop()

        elif value == 100:
            self.fpsLabel.setText("Fastest")
            self.stepTimer.setInterval(0)
            self.stepTimer.start()

        else:
            self.fpsLabel.setText("%s FPS" % value)
            self.stepTimer.setInterval(int(1000 / self.fpsLimit))
            self.stepTimer.start()

    def resetEnv(self):
        obs = self.env.reset()
        self.lastObs = obs
        self.showEnv(obs)

    def stepEnv(self, action=None):
        # If no manual action was specified by the user
        if action is None:
            action = selectAction(self.lastObs)

        obs, reward, done, info = self.env.step(action)

        self.showEnv(obs)
        self.lastObs = obs

        if done:
            self.resetEnv()

        if self.curDemo:
            self.curDemo['numSteps'] += 1
            self.curDemo['actions'].append(action)
            img = obs['image'] if isinstance(obs, dict) else obs
            self.curDemo['nextObservations'].append(img)

    def startDemo(self):
        assert self.curDemo is None

        mission = self.missionBox.toPlainText()
        assert len(mission) > 0

        env = self.env.unwrapped

        env.mission = mission

        obs = self.env._genObs()
        img = obs['image'] if isinstance(obs, dict) else obs

        self.curDemo = {
            'mission': mission,
            'startPos': env.agentPos,
            'startGrid': env.grid.copy(),
            'startObservation': img,
            'numSteps': 0,
            'actions': [],
            'nextObservations': []
        }

        # Set the focus on the full render image
        self.imgLabel.setFocus()

    def endDemo(self):
        import pickle

        assert self.curDemo is not None

        env = self.env.unwrapped

        self.curDemo['endGrid'] = env.grid.copy()
        self.curDemo['endPos'] = env.agentPos
        self.curDemo['testSet'] = self.testSetBox.isChecked()

        self.demos.append(self.curDemo)
        print('new demo with length %d: "%s" (%d total demos)' % (
            self.curDemo['numSteps'],
            self.curDemo['mission'],
            len(self.demos)
        ))
        self.curDemo = None

        # Clear the mission text
        env.mission = ''
        self.missionBox.setPlainText('')

        # Clear the test set box
        self.testSetBox.setCheckState(False)

        pickle.dump(self.demos, open('demos.p', 'wb'))

    def showEnv(self, obs):
        unwrapped = self.env.unwrapped

        # Render and display the environment
        pixmap = self.env.render(mode='pixmap')
        self.imgLabel.setPixmap(pixmap)

        # Render and display the agent's view
        image = obs['image']
        obsPixmap = unwrapped.getObsRender(image)
        self.obsImgLabel.setPixmap(obsPixmap)

        # Update the mission text
        mission = obs['mission']
        self.missionBox.setPlainText(mission)

        # Set the steps remaining
        stepsRem = unwrapped.getStepsRemaining()
        self.stepsLabel.setText(str(stepsRem))


def main(argv):
    parser = OptionParser()
    parser.add_option(
        "--env-name",
        help="gym environment to load",
        default='MiniGrid-Playground-v0'
    )
    (options, args) = parser.parse_args()

    # Load the gym environment
    env = gym.make(options.env_name)

    # Create the application window
    app = QApplication(sys.argv)
    window = AIGameWindow(env)

    # Run the application
    sys.exit(app.exec_())

if __name__ == '__main__':
    main(sys.argv)
