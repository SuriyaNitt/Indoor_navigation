from numpy import dot, sum, tile, linalg
from numpy.linalg import inv, pinv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class Kalman():
    def __init__(self, X, P, A, Q, R, B, U, H):
        self.X = X
        self.P = P
        self.A = A
        self.Q = Q
        self.R = R
        self.B = B
        self.H = H

    def __call__(self):
        pass

    # def predict(X, P, A, Q, B, U):
    def predict(self, U):
        '''
            X - state vector
            P - state covariance matrix
            A - transition matrix
            Q - process covariance matrix
            B - input effect matrix
            U - control input
        '''

        self.X = dot(self.A, self.X) + dot(self.B, U)
        self.P = dot(self.A, dot(self.P, self.A.T)) + self.Q
        return self.X

    # def update(X, P, Y, H, R):
    def update(self, Y):
        '''
            X - state vector 
            P - state covariance matrix
            Y - measurement
            H - measurement matrix
            R - measurement covariance matrix
        '''

        Y_pred = dot(self.H, self.X)
        S = dot(self.H, dot(self.P, self.H.T)) + self.R

        if type(S) == np.float64:
            K = dot(self.P, dot(self.H.T, 1/S))
        else:
            # print S
            K = dot(self.P, dot(self.H.T, pinv(S)))
        # K - Kalman gain
        self.X = self.X + dot(K, (Y - Y_pred))
        self.P = self.P - dot(K, dot(self.H, self.P))
        return self.X
    
def process_datafile(fileName):
    fileData = pd.read_csv(fileName)
    fileData = fileData.dropna()
    fileData = fileData.reset_index(drop=True)
    print fileData.dtypes

    accelerometerData = fileData[['motionUserAccelerationX', 'motionUserAccelerationY', 'motionUserAccelerationZ']]
    gyroscopeData = fileData[['motionRotationRateX', 'motionRotationRateY', 'motionRotationRateZ']]
    deviceOrientation = fileData[['motionYaw', 'motionRoll', 'motionPitch']]
    headingData = fileData[['HeadingX', 'HeadingY', 'HeadingZ', 'TrueHeading', 'MagneticHeading']]

    return (accelerometerData, gyroscopeData, deviceOrientation, headingData)

def dict_to_list(myDict):
    myList = []
    for key, value in myDict.iteritems():
        myList.append(value)

    return myList

if __name__ == '__main__':
    aData, gData, dData, hData = process_datafile('input.csv')
    aX = aData[['motionUserAccelerationX']].to_dict()
    aY = aData[['motionUserAccelerationY']].to_dict()
    aZ = aData[['motionUserAccelerationZ']].to_dict()

    rX = gData[['motionRotationRateX']].to_dict()
    rY = gData[['motionRotationRateY']].to_dict()
    rZ = gData[['motionRotationRateZ']].to_dict()

    yaw = dData[['motionYaw']].to_dict()
    roll = dData[['motionRoll']].to_dict()
    pitch = dData[['motionPitch']].to_dict()

    hX = hData[['HeadingX']].to_dict()
    hY = hData[['HeadingY']].to_dict()
    hZ = hData[['HeadingZ']].to_dict()
    trueHeading = hData[['TrueHeading']].to_dict()

    aXlist = dict_to_list(aX['motionUserAccelerationX'])
    aYlist = dict_to_list(aY['motionUserAccelerationY'])
    aZlist = dict_to_list(aZ['motionUserAccelerationZ'])

    rXlist = dict_to_list(rX['motionRotationRateX'])
    rYlist = dict_to_list(rY['motionRotationRateY'])
    rZlist = dict_to_list(rZ['motionRotationRateZ'])

    yawlist = dict_to_list(yaw['motionYaw'])
    rolllist = dict_to_list(roll['motionRoll'])
    pitchlist = dict_to_list(pitch['motionPitch'])

    hXlist = dict_to_list(hX['HeadingX'])
    hYlist = dict_to_list(hY['HeadingY'])
    hZlist = dict_to_list(hZ['HeadingZ'])
    trueHeadinglist = dict_to_list(trueHeading['TrueHeading'])

    timestamp = range(len(aXlist))

    # plt.figure()
    # plt.plot(aXlist, 'k+', label='accelerationX')
    # # plt.plot(timestamp, 'b-', 'timestamp')
    # plt.legend()
    # plt.title('accelerationX over time')
    # plt.xlabel('timestamp')
    # plt.ylabel('accelerationX')

    # plt.show()

    timepertimestamp = 0.05 # unit is in seconds
    noIters = len(aXlist)

    distance = 0
    movementX = 0
    movementY = 0
    movementZ = 0

    pathX = []
    pathY = []
    pathZ = []
    steps = []

    for i in range(noIters):
        if i == 0:
            X = np.zeros((3,1))
            P = np.eye(3)
            A = np.eye(3)
            Q = np.array([1e-5, 1e-5, 1e-5])
            R = np.array([5e-1, 5e-1, 1e-2])
            B = np.zeros((3,3))
            U = np.zeros((3,1))
            H = np.eye(3)

            accKalman = Kalman(np.array(X), \
                                  np.array(P), \
                                  np.array(A), \
                                  np.array(Q), \
                                  np.array(R), \
                                  np.array(B), \
                                  np.array(U), \
                                  np.array(H))

            distanceX = aXlist[i] * (timepertimestamp ** 2 ) * 0.5
            distanceY = aYlist[i] * (timepertimestamp ** 2 ) * 0.5
            distanceZ = aZlist[i] * (timepertimestamp ** 2 ) * 0.5

        else:

            accKalman.predict(np.zeros((3,1)))
            oldAcc = np.array([[aXlist[i]], [aYlist[i]], [aZlist[i]]])
            newAcc = accKalman.update(oldAcc)

            distanceX = newAcc[0][0] * (timepertimestamp ** 2 ) * 0.5
            distanceY = newAcc[1][0] * (timepertimestamp ** 2 ) * 0.5  
            distanceZ = oldAcc[2][0] * (timepertimestamp ** 2 ) * 0.5           

        distance += np.sqrt(distanceX ** 2 + distanceY ** 2)
        steps.append(np.sqrt(distanceX ** 2 + distanceY ** 2))

        movementX += distanceX
        movementY += distanceY
        movementZ += -1*distanceZ

        pathX.append(movementX)
        pathY.append(movementY)
        pathZ.append(movementZ)

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot(pathX, pathY, pathZ, label='movement curve')
    # ax.legend()

    angleX = 0
    angleY = 0
    angleZ = 0

    poseX = []
    poseY = []
    poseZ = []

    rotations = []

    for i in range(noIters):
        angleX += rXlist[i] * timepertimestamp
        angleY += rYlist[i] * timepertimestamp

        if i == 0:
            X = 0
            P = 1
            A = 1
            Q = 1e-5
            R = 3e-1
            B = 0
            U = 0
            H = 1

            angleZKalman = Kalman(np.array(X), \
                                  np.array(P), \
                                  np.array(A), \
                                  np.array(Q), \
                                  np.array(R), \
                                  np.array(B), \
                                  np.array(U), \
                                  np.array(H))
            angleZ += rZlist[i] * timepertimestamp
            rotations.append(angleZ)

        else:
            
            angleZKalman.predict(np.array(0))
            rotationRate = angleZKalman.update(np.array(rZlist[i]))
            rotation = rotationRate * timepertimestamp 
            angleZ += rotation
            rotations.append(rotation)

        poseX.append(angleX)
        poseY.append(angleY)
        poseZ.append(angleZ)

    # fig2 = plt.figure()
    # ax = fig2.gca()
    # ax.plot(timestamp, poseX, label='poseX')
    # ax.legend()

    # fig3 = plt.figure()
    # ax = fig3.gca()
    # ax.plot(timestamp, poseY, label='poseY')
    # ax.legend()

    # fig4 = plt.figure()
    # ax = fig4.gca()
    # ax.plot(timestamp, poseZ, label='poseZ')
    # ax.legend()

    # fig5 = plt.figure()
    # ax = fig5.gca()
    # ax.plot(timestamp, yawlist, label='yaw')
    # ax.legend()

    # fig6 = plt.figure()
    # ax = fig6.gca()
    # ax.plot(timestamp, rolllist, label='roll')
    # ax.legend()

    # fig7 = plt.figure()
    # ax = fig7.gca()
    # ax.plot(timestamp, pitchlist, label='pitch')
    # ax.legend()

    # fig8 = plt.figure()
    # ax = fig8.gca()
    # ax.plot(timestamp, hXlist, label='HeadingX')
    # ax.legend()

    # fig9 = plt.figure()
    # ax = fig9.gca()
    # ax.plot(timestamp, hYlist, label='HeadingY')
    # ax.legend()

    # fig10 = plt.figure()
    # ax = fig10.gca()
    # ax.plot(timestamp, hZlist, label='HeadingZ')
    # ax.legend()

    tH = []

    for i in range(len(trueHeadinglist)):
        if i == 0:
            X = 0
            P = 1
            A = 1
            Q = 1e-5
            R = 5e-1
            B = 0
            U = 0
            H = 1

            thKalman = Kalman(np.array(X), \
                              np.array(P), \
                              np.array(A), \
                              np.array(Q), \
                              np.array(R), \
                              np.array(B), \
                              np.array(U), \
                              np.array(H))

            tH.append(trueHeadinglist[i])

        else:
            thKalman.predict(np.array(0))
            newTH = thKalman.update(np.array(trueHeadinglist[i]))
            # print newTH
            tH.append(newTH)


    # fig11 = plt.figure()
    # ax = fig11.gca()
    # ax.plot(timestamp, tH, label='TrueHeading')
    # ax.legend()

    x = [0, steps[1]]
    y = [0, 0]

    for i in range(1, len(rotations)-1):
        angle = rotations[i+1]
        radius = steps[i+1]

        # print angle

        slope = (y[i] - y[i-1]) / (x[i] - x[i-1])

        x1 = x[i] + radius * np.sqrt(1/(1+slope**2))
        x2 = x[i] - radius * np.sqrt(1/(1+slope**2))

        y1 = y[i] + slope * radius * np.sqrt(1/(1+slope**2))
        y2 = y[i] - slope * radius * np.sqrt(1/(1+slope**2))

        valid = [0, 0, 0, 0]
        if (abs((y1-y[i])/(x1-x[i]) - slope) < 1e-6 ):
            valid[0] = 1
        if (abs((y1-y[i])/(x2-x[i]) - slope) < 1e-6 ):
            valid[1] = 1
        if (abs((y2-y[i])/(x1-x[i]) - slope) < 1e-6 ):
            valid[2] = 1
        if (abs((y2-y[i])/(x2-x[i]) - slope) < 1e-6 ):
            valid[3] = 1

        for j, v in enumerate(valid):
            if v == 1:
                if j == 0 or j == 2:
                    if (x[i-1] < x[i] and x1 < x[i]) or (x[i-1] > x[i] and x1 > x[i]):
                        valid[0] = 0
                        valid[2] = 0
                if j == 1 or j == 3:
                    if (x[i-1] < x[i] and x2 < x[i]) or (x[i-1] > x[i] and x2 > x[i]):
                        valid[1] = 0
                        valid[3] = 0

        endX = 0
        endY = 0

        for j, v in enumerate(valid):
            if v == 1:
                if j == 0 or j == 2:
                    endX = x1
                    endY = y1
                    break
                if j == 1 or j == 3:
                    endX = x2
                    endY = y2
                    break

        rotatedCoords = dot(np.array([ [np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)] ]), \
                            np.array([ [endX - x[i]], [endY - y[i]] ]) ) \
                        + np.array([[x[i]], [y[i]]])

        x.append(rotatedCoords[0][0])
        y.append(rotatedCoords[1][0])


    fig12 = plt.figure()
    ax = fig12.gca()
    ax.plot(x[:], y[:], label='Path - top view')
    ax.legend()

    fig13 = plt.figure()
    ax = fig13.gca(projection='3d')
    print len(x), len(y), len(pathZ)
    ax.plot(x, y, pathZ[:], label='Path - 3D')
    ax.legend()

    plt.show()





