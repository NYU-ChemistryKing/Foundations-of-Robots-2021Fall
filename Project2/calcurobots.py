# Here are some other basic functions writen before.
import numpy as np
import math
from scipy.linalg import expm
from scipy.linalg import logm

# 1.from a → [a] ( 3 x 3 )
def GetMatrixForm(w):

    bnw = np.array([
        [0      , -w[2,0] , w[1,0] ],
        [w[2,0] , 0       , -w[0,0]],
        [-w[1,0], w[0,0]  , 0      ]
        ])
    
    return bnw

# 2.from [a] → a ( 3 x 1 )   
def GetVector(mw):

    w=([
    [mw[2,1]],
    [mw[0,2]],
    [mw[1,0]]
    ])

    return w

# 3.Rot( axis, theta )
# Counterclockwise means positive angle
def Rot( axis, theta ):

    if axis == 'x':

        R = np.array([
            [1, 0 ,0],
            [0, math.cos(theta), -math.sin(theta)],
            [0, math.sin(theta), math.cos(theta)]
        ])

        return R
    
    elif axis == 'y':

        R = np.array([
            [math.cos(theta), 0 ,math.sin(theta)],
            [0, 1, 0],
            [-math.sin(theta), 0, math.cos(theta)]
        ])   

        return R
    
    elif axis == 'z':

        R = np.array([
        [math.cos(theta), -math.sin(theta), 0],
        [math.sin(theta), math.cos(theta), 0],
        [0, 0 ,1]
    ])

        return R
    
    else:  

        return print("Error!")

# 4.EulerAngles α β γ Rotation Matrix
def EulerRotationMatrix(x,y,z):

    R = np.array([

        [math.cos(x) * math.cos(y), math.cos(x) * math.sin(y) * math.sin(z) - math.sin(x) * math.cos(z), math.cos(x) * math.sin(y) * math.cos(z) + math.sin(x) * math.sin(z)],
        [math.sin(x) * math.cos(y), math.sin(x) * math.sin(y) * math.sin(z) + math.cos(x) * math.cos(z), math.sin(x) * math.sin(y) * math.cos(z) - math.cos(x) * math.sin(z)],
        [-math.sin(y)             , math.cos(y) * math.sin(z)                                          , math.cos(y) * math.cos(z)                                          ]

        ])

    return R    

# 5.From R → EulerAngles α β γ
# Positive angle means clockwise (shun shi zhen) around the axis
def getEulerAngles(R):
    sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    
    return np.array([x,y,z])

# 5.1 Judge R is or not a Real Rotation Matrix
# print(isRotationMatrix(R))
def isRotationMatrix(R):
    # square matrix test
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        return False
    should_be_identity = np.allclose(R.dot(R.T), np.identity(R.shape[0], np.float))
    should_be_one = np.allclose(np.linalg.det(R), 1)
    return should_be_identity and should_be_one

# 6.from R,p → T (Transforms)
#   R,p = getRpfromT(T)
def getTfromRp(R,p):

    M = np.hstack(     (R, p)    )
    T = np.vstack(     (M, np.array([0,0,0,1]))   )

    return T

# 7.from T → R,p
def getRpfromT(T):

    def getRfromT(T):
        
        R = T[0:3, 0:3]
        
        return R

    def getpfromT(T):

        p = np.array([

            [T[0,3]],
            [T[1,3]],
            [T[2,3]]

        ])

        return p

    R = getRfromT(T)
    p = getpfromT(T)

    return R,p

# 8.from T → T^(-1)
def getInversedT(T):

    def getRfromT(T):
        
        R = T[0:3, 0:3]
        
        return R

    def getpfromT(T):

        p = np.array([

            [T[0,3]],
            [T[1,3]],
            [T[2,3]]

        ])

        return p
    
    R = getRfromT(T)
    p = getpfromT(T)
    TR = np.transpose(R)
    Tp = -TR @ p
    M = np.hstack( (TR, Tp) )  
    N = np.array([0,0,0,1])  
    T = np.vstack(( M, N ))

    return T   

# 9.from T,point or T,vector → Transfromed point or Transformed vector
def PointsTransform(T,p):

    pbar = np.vstack( (p,1) )
    pt = T @ pbar
    p = pt[0:3]

    return p

def VectorTransform(T,v):

    vbar = np.vstack( (v,0) )
    vt = T @ vbar
    v = vt[0:3]

    return v
# 10. From w,R → dr
def getdiffR(w,R):

    mw = GetMatrixForm(w)
    dR = mw @ R

    return dR

# 10.1 From w,t ➡️ R
def getRfromwt(w,t):

    mw = GetMatrixForm(w)
    R = expm( mw * t)

    return R
# 10.2 w0 = R01 @ w1
# 10.3 w01 = - w10

# How to get w10_in0 from w01_in1?
#------------------------------------

# w01_in0 = R01 @ w01_in1
# w10_in0 = -1 * w01 in 0

#------------------------------------

# 10.4 get w and theta from R
def getwthetafromR(R):

    theta = math.acos(0.5*(np.trace(R)-1))
    bdw = (R-np.transpose(R))/(2*math.sin(theta))
    w = GetVector(bdw)

    return w,theta
# 11. From Twist → w,v
#     w,v = getwvfromTwist(V)
def getwvfromTwist(V):

    w = np.array([
        [V[0,0]],
        [V[1,0]],
        [V[2,0]]
    ])

    v = np.array([
        [V[3,0]],
        [V[4,0]],
        [V[5,0]]
    ])

    return w,v

# 12. From w,v → Twist
def getTwistfromwv(w,v):

    V = np.array([

        [w[0,0]],
        [w[1,0]],
        [w[2,0]],
        [v[0,0]],
        [v[1,0]],
        [v[2,0]]

    ])
        
    return V    
# 12.1 From w,p → Twist
def getTwistfromwp(w,p):

    Mw = -GetMatrixForm(w)
    v = Mw @ p

    V = np.array([

        [w[0,0]],
        [w[1,0]],
        [w[2,0]],
        [v[0,0]],
        [v[1,0]],
        [v[2,0]]

    ])
        
    return V 

# 13. From V → [V]
def getMatrixTwist(V):
    
    MV = np.array([

        [0, -V[2,0], V[1,0], V[3,0]],
        [V[2,0], 0, -V[0,0], V[4,0]],
        [-V[1,0], V[0,0], 0, V[5,0]],
        [0, 0, 0, 0]

    ])

    return MV

# 14. From [V] → V
def getVectorV(MV):

    V = np.array([

        [MV[2,1]],
        [MV[0,2]],
        [MV[1,0]],
        [MV[0,3]],
        [MV[1,3]],
        [MV[2,3]]

    ])
        
    return V

# 15. from T → V
def getTwistfromTranform(T):

    MV = logm(T)
    V = getVectorV(MV)

    return V

# 16. from V → T
def getTransformfromTwist(V):

    MV = getMatrixTwist(V)
    T = expm(MV)

    return T

# (if the input is M V t)
def GetTransformationFromV(M,V,t):

    ExpV = expm( getMatrixTwist(V) * t )
    T = ExpV @ M

    return T

# 17. Adjoint
def getAdT(T):
    
    R,p = getRpfromT(T)
    mp = GetMatrixForm(p)
    M = np.dot(mp, R)
    O = np.dot(0, np.eye(3))
    
    H1 = np.hstack((R,O))
    H2 = np.hstack((M,R))

    AdT = np.vstack((H1,H2))
    
    return AdT
# 17.1 Get a velocity of a point
def GetpointvelocityfromTwist(V,p):

    w,v = getwvfromTwist(V)
    mw = GetMatrixForm(w)
    dp = v + mw @ p

    return dp
# 18. exp( [S] * theta )
#     T = exp_MS1 @ exp_MS2 @ ... @ exp_MSn @ M
#     S & M is read from picture
def expS(S,theta):
    
    MS = getMatrixTwist(S)
    exp_MS = expm( MS * theta)

    return exp_MS

def vec_to_skew(w):
    
    skew_w = np.array([
        [0      , -w[2,0] , w[1,0] ],
        [w[2,0] , 0       , -w[0,0]],
        [-w[1,0], w[0,0]  , 0      ]
        ])
    
    return skew_w

def twist_to_skew(V):
    
    skew_V = np.array([

        [0, -V[2,0], V[1,0], V[3,0]],
        [V[2,0], 0, -V[0,0], V[4,0]],
        [-V[1,0], V[0,0], 0, V[5,0]],
        [0, 0, 0, 0]

    ])

    return skew_V

def exp_twist_bracket(V):

    MV = twist_to_skew(V)
    T = expm(MV)

    return T

def inverseT(T):

    def getRfromT(T):
        
        R = T[0:3, 0:3]
        
        return R

    def getpfromT(T):

        p = np.array([

            [T[0,3]],
            [T[1,3]],
            [T[2,3]]

        ])

        return p
    
    R = getRfromT(T)
    p = getpfromT(T)
    TR = np.transpose(R)
    Tp = -TR @ p
    M = np.hstack( (TR, Tp) )  
    N = np.array([0,0,0,1])  
    T = np.vstack(( M, N ))

    return T

def getAdjoint(T):
    
    R,p = getRpfromT(T)
    mp = vec_to_skew(p)
    M = np.dot(mp, R)
    O = np.dot(0, np.eye(3))
    
    H1 = np.hstack((R,O))
    H2 = np.hstack((M,R))

    AdT = np.vstack((H1,H2))
    
    return AdT    



