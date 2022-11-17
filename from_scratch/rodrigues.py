import numpy as np
import math

def R_to_rvec(R): # TODO Fix this function
    R_ = (R-R.T)/2
    rx_ = R_[2][1]
    ry_ = R_[0][2]
    rz_ = R_[1][0]
    R_ = R_.flatten()
    norm = np.linalg.norm([rx_,ry_,rz_])
    theta = math.asin(norm)
    # if norm and not np.allclose(R_-np.sin(theta)*R_/norm,0.001):
    #     theta = np.pi - theta
    #     print("theta", theta)
    rvec_ = np.array([rx_,ry_,rz_])
    if theta == 0:
        return np.zeros((3,1))
    rvec = theta * rvec_/ math.sin(theta)
    return rvec.reshape((3,1))

def rvec_to_R(rvec):
    theta = np.linalg.norm(rvec)
    if theta == 0:
        return np.eye(3)
    r = rvec/theta
    rx = r[0][0]
    ry = r[1][0]
    rz = r[2][0]
    R = math.cos(theta)*np.eye(3) + (1-math.cos(theta))*r @ r.T + math.sin(theta) * np.array([[0, -rz, ry],[rz,0,-rx],[-ry, rx, 0]])
    return R
