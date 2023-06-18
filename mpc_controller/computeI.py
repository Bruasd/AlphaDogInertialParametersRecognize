import pybullet as p
import numpy as np
import Ei
import math




def computeY(J, sv,sa,parent_num,E,S1_0):
    #Jacobian:关节雅克比矩阵
    #sv:世界坐标系下的关节速度
    #sa:世界坐标系下的关节加速度
    #parent_num:父关节数目
    #E:系数矩阵
    

    #计算y_i,令I_i=e_
    Y=np.zeros((3,30))
    I=[[1,0,0],[0,1,0], [0,0,1]]
    num=0
    # print(sv[0])
    # print(sa[0])

    for i in range(3):
    #分别计算10个Y
        for j in range(10):
            I1=I[i][0]
            I2=I[i][1]
            I3=I[i][2]
            I1=I1*E[j]
            I2=I2*E[j]
            I3=I3*E[j]
            #求和
            tempI=[]
            tempI.append(I1)
            tempI.append(I2)
            tempI.append(I3)

            a=np.dot(tempI[i],sa[i]).reshape(6,1)
            # if  i==0:
            #     print(a)
            b=np.dot(get_screw_v_star(sv[i]),tempI[i])
            b=np.dot(b,sv[i]).reshape(6,1)
            a=np.dot(np.transpose(J[i]),a+b).reshape(3,)
            
           
            Y[:,num]=a
            num=num+1
            # print(Y)
        
    
    return np.array(Y)

def get_screw_v_star(sv):
    #计算sv*
    sv_star=np.zeros((6,6))
    sv_star[0:3,0:3]=get_skew_matrix(sv[0:3])
    sv_star[3:6,3:6]=get_skew_matrix(sv[0:3])
    sv_star[0:3,3:6]=get_skew_matrix(sv[3:6])
    return sv_star
    

def get_skew_matrix(v):
    #计算反对称矩阵
    skew_matrix=np.zeros((3,3))
    skew_matrix[0,1]=-v[2]
    skew_matrix[0,2]=v[1]
    skew_matrix[1,0]=v[2]
    skew_matrix[1,2]=-v[0]
    skew_matrix[2,0]=-v[1]
    skew_matrix[2,1]=v[0]
    return skew_matrix
def get_screw_v(sv):
    #计算sv
    w=sv[0:3]
    v=sv[3:6]
    skew_matrix1=get_skew_matrix(w)
    skew_matrix2=get_skew_matrix(v)
    sv=np.zeros((6,6))
    sv[0:3,0:3]=skew_matrix1
    sv[3:6,3:6]=skew_matrix1
    sv[3:6,0:3]=skew_matrix2
    return sv

#计算joint的速度在自己坐标系下的表示
def computeV(X, V_p, S, theta_dot): 
    # X 变换矩阵
    # V_p 父的速度
    # S 轴
    # theta_dot 角速度
    #输出：关节坐标系下的速度
    S=S.reshape(6,)
    V = V_p + np.dot(X, S*theta_dot)
    
    return V

#计算joint的加速度在自己坐标系下的表示
def computeA(X ,V, A_p, S, theta_dot, theta_ddot): 
    # X 变换矩阵
    # V_p 父的速度
    # A_p 父的加速度
    # S 轴
    # theta_dot 角速度
    # theta_ddot 角加速度
    #输出：关节坐标系下的加速度
    S=S.reshape(6,)
    
    A = A_p+ np.dot(X,(S*theta_ddot).reshape(6,)) + np.dot(get_screw_v(V), np.dot(X,S*theta_dot))
    return A

#三个joint的速度拼接
def compute_sv(X1_0, S1, theta1_dot, X2_0, S2, theta2_dot, X3_0, S3, theta3_dot):   
    sv=[]
    V1_1 = computeV(np.linalg.inv(X1_0), np.array([0, 0, 0, 0, 0, 0]), S1, theta1_dot)
    V2_2 = computeV(np.linalg.inv(X2_0), V1_1, S2, theta2_dot)
    V3_3 = computeV(np.linalg.inv(X3_0), V2_2, S3, theta3_dot)
    sv.append(V1_1)
    sv.append(V2_2)
    sv.append(V3_3)
    return np.array(sv)

#三个joint的加速度拼接
def compute_sa(X1_0, S1, theta1_dot, theta1_ddot, X2_0, S2, theta2_dot, theta2_ddot, X3_0, S3, theta3_dot, theta3_ddot):
    V1_1 = computeV(np.linalg.inv(X1_0), np.array([0, 0, 0, 0, 0, 0]), S1, theta1_dot)
    A1_1 = computeA(np.linalg.inv(X1_0), V1_1, np.array([0, 0, 0, 0, 0, 0]), 
                    S1, theta1_dot, theta1_ddot)
    V2_2 = computeV(np.linalg.inv(X2_0), V1_1, S2, theta2_dot)
    A2_2 = computeA(np.linalg.inv(X2_0), V2_2, A1_1, S2, theta2_dot, theta2_ddot)
    V3_3 = computeV(np.linalg.inv(X3_0), V2_2, S3, theta3_dot)
    A3_3 = computeA(np.linalg.inv(X3_0), V3_3, A2_2, S3, theta3_dot, theta3_ddot)
    sa = []
    sa.append(A1_1)
    sa.append(A2_2)
    sa.append(A3_3)
    return np.array(sa)


# #测试用例
# S1_1 = np.array([0, 0, 1, 0, 0, 0])
# S2_2 = np.array([0, 0, 1, 0, 0, 0])
# theta_1=0.89
# theta_1_dot=-1.26
# theta_1_ddot=-268
# theta_2=-1.8
# theta_2_dot=5.5
# theta_2_ddot=1102

# X0_1 = np.array(
#     [
#         [math.cos(theta_1), -math.sin(theta_1), 0, 0,0,0],
#         [math.sin(theta_1), math.cos(theta_1), 0, 0,0,0],
#         [0, 0, 1, 0,0,0],
#         [0, 0, 0, math.cos(theta_1), -math.sin(theta_1), 0],
#         [0, 0, 0, math.sin(theta_1), math.cos(theta_1), 0],
#         [0, 0, 0, 0, 0, 1]
#     ]
# )
# X1_2 = np.array(
#     [
#         [math.sin(theta_2), math.cos(theta_2), 0, 0,0,0],
#         [0,0,-1,0,0,0],
#         [-math.cos(theta_2), math.sin(theta_2), 0, 0,0,0],
#         [0, 0, 0, math.sin(theta_2), math.cos(theta_2), 0],
#         [math.cos(theta_2), -math.sin(theta_2), 0, 0, 0, -1],
#         [0, 0, -1, -math.cos(theta_2), math.sin(theta_2), 0]
#     ]
# )
# J=[]
# Jacbian1=np.array(

#    [ [0,0],
#     [0,0],
#     [1,0],
#     [0,0],
#     [0,0],
#     [0,0]
#    ]
# )
# J.append(Jacbian1)
# Jacbian2=np.array(
#     [ [-math.cos(theta_2),0],
#     [math.sin(theta_2),0],
#     [0,0],
#     [0,0],
#     [0,0],
#     [0,0]
     
#      ]
# )
# J.append(Jacbian2)
# J=np.array(J)
# sv=compute_sv(X0_1, S1_1, theta_1_dot, X1_2, S2_2, theta_2_dot)
# sa=compute_sa(X0_1, S1_1, theta_1_dot, theta_1_ddot, X1_2, S2_2, theta_2_dot, theta_2_ddot)
# y=computeY(J,sv,sa,1,Ei.E)
# print(y)
# print(sa)