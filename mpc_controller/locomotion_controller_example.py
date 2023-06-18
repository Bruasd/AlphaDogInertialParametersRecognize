
from __future__ import absolute_import
from __future__ import division
#from __future__ import google_type_annotations
from __future__ import print_function
from sklearn.linear_model import *
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

from absl import app
from absl import flags
import scipy.interpolate
import numpy as np
import pybullet_data as pd
from pybullet_utils import bullet_client

import time
import pybullet
import random
import modern_robotics as mr
from mpc_controller import com_velocity_estimator
from mpc_controller import gait_generator as gait_generator_lib
from mpc_controller import locomotion_controller
from mpc_controller import openloop_gait_generator
from mpc_controller import raibert_swing_leg_controller
from mpc_controller import torque_stance_leg_controller
import computeI

from scipy.optimize import minimize

#uncomment the robot of choice
#from mpc_controller import laikago_sim as robot_sim
from mpc_controller import a1_sim as robot_sim

FLAGS = flags.FLAGS


_NUM_SIMULATION_ITERATION_STEPS = 300


_STANCE_DURATION_SECONDS = [
    0.3
] * 4  # For faster trotting (v > 1.5 ms reduce this to 0.13s).


# Trotting
_DUTY_FACTOR = [0.6] * 4
_INIT_PHASE_FULL_CYCLE = [0.9, 0, 0, 0.9]
_MAX_TIME_SECONDS = 5

_INIT_LEG_STATE = (
    gait_generator_lib.LegState.SWING,
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.SWING,
)



def _generate_example_linear_angular_speed(t):
  """Creates an example speed profile based on time for demo purpose."""
  vx = 0 * robot_sim.MPC_VELOCITY_MULTIPLIER
  vy = 0* robot_sim.MPC_VELOCITY_MULTIPLIER
  wz = 0* robot_sim.MPC_VELOCITY_MULTIPLIER
  
  time_points = (0, 5, 10, 15, 20, 25,30)
  speed_points = ((0, 0, 0, 0), (0, 0, 0, wz), (vx, 0, 0, 0), (0, 0, 0, -wz), (0, -vy, 0, 0),
                  (0, 0, 0, 0), (0, 0, 0, wz))


  speed = scipy.interpolate.interp1d(
      time_points,
      speed_points,
      kind="previous",
      fill_value="extrapolate",
      axis=0)(
          t)

  return speed[0:3], speed[3]


def _setup_controller(robot):
  """Demonstrates how to create a locomotion controller."""
  desired_speed = (0, 0)
  desired_twisting_speed = 0

  gait_generator = openloop_gait_generator.OpenloopGaitGenerator(
      robot,
      stance_duration=_STANCE_DURATION_SECONDS,
      duty_factor=_DUTY_FACTOR,
      initial_leg_phase=_INIT_PHASE_FULL_CYCLE,
      initial_leg_state=_INIT_LEG_STATE)
  state_estimator = com_velocity_estimator.COMVelocityEstimator(robot,
                                                                window_size=20)
  sw_controller = raibert_swing_leg_controller.RaibertSwingLegController(
      robot,
      gait_generator,
      state_estimator,
      desired_speed=desired_speed,
      desired_twisting_speed=desired_twisting_speed,
      desired_height=robot_sim.MPC_BODY_HEIGHT,
      foot_clearance=0.01)

  st_controller = torque_stance_leg_controller.TorqueStanceLegController(
      robot,
      gait_generator,
      state_estimator,
      desired_speed=desired_speed,
      desired_twisting_speed=desired_twisting_speed,
      desired_body_height=robot_sim.MPC_BODY_HEIGHT,
      body_mass=robot_sim.MPC_BODY_MASS,
      body_inertia=robot_sim.MPC_BODY_INERTIA)

  controller = locomotion_controller.LocomotionController(
      robot=robot,
      gait_generator=gait_generator,
      state_estimator=state_estimator,
      swing_leg_controller=sw_controller,
      stance_leg_controller=st_controller,
      clock=robot.GetTimeSinceReset)
  return controller



def _update_controller_params(controller, lin_speed, ang_speed):
  controller.swing_leg_controller.desired_speed = lin_speed
  controller.swing_leg_controller.desired_twisting_speed = ang_speed
  controller.stance_leg_controller.desired_speed = lin_speed
  controller.stance_leg_controller.desired_twisting_speed = ang_speed


def _run_example(max_time=_MAX_TIME_SECONDS):
  """Runs the locomotion controller example."""
  
  #recording video requires ffmpeg in the path
  record_video = False
  if record_video:
    p = pybullet
    p.connect(p.GUI, options="--widjoint_namesth=1280 --height=720 --mp4=\"test.mp4\" --mp4fps=100")
    p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
  else:
     p = bullet_client.BulletClient(connection_mode=pybullet.GUI)    
         



  p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
  p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
  
  p.setAdditionalSearchPath(pd.getDataPath())
  
  num_bullet_solver_iterations = 30

  p.setPhysicsEngineParameter(numSolverIterations=num_bullet_solver_iterations)
  
  
  p.setPhysicsEngineParameter(enableConeFriction=0)
  p.setPhysicsEngineParameter(numSolverIterations=30)
  simulation_time_step = 0.001

  p.setTimeStep(simulation_time_step)
 
  p.setGravity(0, 0, 0)
  p.setPhysicsEngineParameter(enableConeFriction=0)
  p.setAdditionalSearchPath(pd.getDataPath())
  
  #random.seed(10)
  #p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
  heightPerturbationRange = 0.06
  
  plane = True
  if plane:
    p.loadURDF("plane.urdf")
    #planeShape = p.createCollisionShape(shapeType = p.GEOM_PLANE)
    #ground_id  = p.createMultiBody(0, planeShape)
  else:
    numHeightfieldRows = 256
    numHeightfieldColumns = 256
    heightfieldData = [0]*numHeightfieldRows*numHeightfieldColumns 
    for j in range (int(numHeightfieldColumns/2)):
      for i in range (int(numHeightfieldRows/2) ):
        height = random.uniform(0,heightPerturbationRange)
        heightfieldData[2*i+2*j*numHeightfieldRows]=height
        heightfieldData[2*i+1+2*j*numHeightfieldRows]=height
        heightfieldData[2*i+(2*j+1)*numHeightfieldRows]=height
        heightfieldData[2*i+1+(2*j+1)*numHeightfieldRows]=heigfile = open('myfile.txt', 'w')
    
    terrainShape = p.createCollisionShape(shapeType = p.GEOM_HEIGHTFIELD, meshScale=[.05,.05,1], heightfieldTextureScaling=(numHeightfieldRows-1)/2, heightfieldData=heightfieldData, numHeightfieldRows=numHeightfieldRows, numHeightfieldColumns=numHeightfieldColumns)
    ground_id  = p.createMultiBody(0, terrainShape)

  #p.resetBasePositionAndOrientation(ground_id,[0,0,0], [0,0,0,1])
  
  #p.changeDynamics(ground_id, -1, lateralFriction=1.0)
  
  robot_uid = p.loadURDF(robot_sim.URDF_NAME, robot_sim.START_POS)


  # Suspend the robot in mid-air
  basePosition, baseOrientation = p.getBasePositionAndOrientation(robot_uid)
  childFramePosition = [0, 0, 3]
  fixtorso = p.createConstraint(robot_uid, -1, -1, -1, p.JOINT_FIXED, basePosition, baseOrientation, childFramePosition)
  
  #启动关节力传感器
  p.enableJointForceTorqueSensor(robot_uid, 1, 1)
  p.enableJointForceTorqueSensor(robot_uid, 3, 1)
  p.enableJointForceTorqueSensor(robot_uid, 4, 1)

  robot = robot_sim.SimpleRobot(p, robot_uid, simulation_time_step=simulation_time_step)
  
  controller = _setup_controller(robot)
  controller.reset()
  
  p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
  #while p.isConnected():
  #  pos,orn = p.getBasePositionAndOrientation(robot_uid)
  #  print("pos=",pos)
  #  p.stepSimulation()
  #  time.sleep(1./240)  
  current_time = robot.GetTimeSinceReset()
  #logId = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "mpc.json")
  theta_1=0
  theta_1_dot=0
  theta_1_ddot=0

  theta_2=0
  theta_2_dot=0
  theta_2_ddot=0

  theta_3=0
  theta_3_dot=0
  theta_3_ddot=0

  test_torche=[]
  test_y=[]
  S1_1=np.zeros((6,1))
  S1_1[0:3,0]=p.getJointInfo(robot_uid, 1)[13]
  S2_2=np.zeros((6,1))
  S2_2[0:3,0]=p.getJointInfo(robot_uid, 3)[13]
  S3_3=np.zeros((6,1))
  S3_3[0:3,0]=p.getJointInfo(robot_uid, 4)[13]

  
  oldv1=0
  oldv2=0
  oldv3=0
  Sbar_3_0 = mr.ScrewToAxis(np.array([0.183, 0.0335, 0.2]), np.array([0, -1, 0]), 0)
  Sbar_2_0 = mr.ScrewToAxis(np.array([0.183, 0.0335, 0]), np.array([0, -1, 0]), 0)
  Sbar_1_0 = mr.ScrewToAxis(np.array([0.183, -0.047, 0]), np.array([1, 0, 0]), 0)
  # Sbar_3_0 = mr.ScrewToAxis(np.array([0.183, -0.01275, -2.2]), np.array([0, -1, 0]), 0)
  # Sbar_2_0 = mr.ScrewToAxis(np.array([0.183, -0.01275, 0]), np.array([0, -1, 0]), 0)
  # Sbar_1_0 = mr.ScrewToAxis(np.array([0.183, -0.047, 0]), np.array([1, 0, 0]), 0)
  while current_time < max_time:
    #pos,orn = p.getBasePositionAndOrientation(robot_uid)
    #print("pos=",pos, " orn=",orn)
    p.submitProfileTiming("loop")
    temp_torche=np.zeros((3,1))
    # Updates the controller behavior parameters.
    lin_speed, ang_speed = _generate_example_linear_angular_speed(current_time)
    # lin_speed, ang_speed = (0., 0., 0.), 0.
    _update_controller_params(controller, lin_speed, ang_speed)

    # Needed before every call to get_action().
    controller.update()
    hybrid_action, info = controller.get_action()
    
    wrench = []
    num_joints = p.getNumJoints(robot_uid)
    for i in range(num_joints):
      joint_info = p.getJointState(robot_uid, i)
      if(i == 1):
        
        theta_1=joint_info[0]
        theta_1_dot=joint_info[1]
        hip_joint_force = joint_info[2]
        temp=np.zeros((6,1))
        temp[0:3]=np.array(hip_joint_force[3:6]).reshape(3,1)
        temp[3:6]=np.array(hip_joint_force[0:3]).reshape(3,1)
        
        wrench.append(temp)
        theta_1_ddot=(joint_info[1]-oldv1)/0.005
        oldv1=joint_info[1]

      if(i == 3):
        
        theta_2=joint_info[0]
        theta_2_dot=joint_info[1]
        upper_joint_force = joint_info[2]
        temp=np.zeros((6,1))
        temp[0:3]=np.array(upper_joint_force[3:6]).reshape(3,1)
        temp[3:6]=np.array(upper_joint_force[0:3]).reshape(3,1)
        
        wrench.append(temp)
        theta_2_ddot=(joint_info[1]-oldv2)/0.005
        oldv2=joint_info[1]

      if(i == 4):
        
        theta_3=joint_info[0]
        theta_3_dot=joint_info[1]
        lower_joint_force = joint_info[2]
        temp=np.zeros((6,1))
        temp[0:3]=np.array(lower_joint_force[3:6]).reshape(3,1)
        temp[3:6]=np.array(lower_joint_force[0:3]).reshape(3,1)
        
        
        wrench.append(temp)
        theta_3_ddot=(joint_info[1]-oldv3)/0.005
        oldv3=joint_info[1]

    if(abs(theta_1_ddot)<700 and abs(theta_2_ddot)<700 and abs(theta_3_ddot)<700):
      
      
      M1=getM(np.eye(3),np.array([0.0813,-0.047,0]).reshape(3,))
      RotationMatix=np.array(
        [[1,0,0],
         [0,-1,0],
         [0,0,-1]])
      M2=getM(RotationMatix,np.array([0.0813,0.0335,0]).reshape(3,))
      M3=getM(RotationMatix,np.array([0.0813,0.0335,0.2]).reshape(3,))

      X1_0=myPoETOX(M1,theta_1,0,0,Sbar_1_0,Sbar_2_0,Sbar_3_0)
      X2_0=myPoETOX(M2,theta_1,theta_2,0,Sbar_1_0,Sbar_2_0,Sbar_3_0)
      X3_0=myPoETOX(M3,theta_1,theta_2,theta_3,Sbar_1_0,Sbar_2_0,Sbar_3_0)

      X1_0_start=np.transpose(np.linalg.inv(X1_0))
      X2_0_start=np.transpose(np.linalg.inv(X2_0))
      X3_0_start=np.transpose(np.linalg.inv(X3_0))
      wrench[0]=np.dot(X1_0_start,wrench[0])
      wrench[1]=np.dot(X2_0_start,wrench[1])
      wrench[2]=np.dot(X3_0_start,wrench[2])
      temp_torche[0,0] = np.dot(np.transpose(np.dot(X1_0,S1_1)),wrench[0])
      temp_torche[1,0] = np.dot(np.transpose(np.dot(X2_0,S2_2)),wrench[1])
      temp_torche[2,0] = np.dot(np.transpose(np.dot(X3_0,S3_3)),wrench[2])
      test_torche.append(temp_torche)

      J=[]
      j1=np.zeros((6,3))
      j1[:,0]=np.dot(X1_0,S1_1).reshape(6,)
      

      j2=np.zeros((6,3))
      
      j2[:,0]=np.dot(X1_0,S1_1).reshape(6,)
      j2[0:,1]=np.dot(X2_0,S2_2).reshape(6,)
      

      j3=np.zeros((6,3))
      j3[:,0]=np.dot(X1_0,S1_1).reshape(6,)
      j3[:,1]=np.dot(X2_0,S2_2).reshape(6,)
      j3[:,2]=np.dot(X3_0,S3_3).reshape(6,)


      J.append(j1)
      J.append(j2)
      J.append(j3)
      J=np.array(J)
      
      sv=computeI.compute_sv(X1_0, S1_1, theta_1_dot, X2_0, S2_2, theta_2_dot, X3_0, S3_3, theta_3_dot)
      
      sa=computeI.compute_sa(X1_0, S1_1, theta_1_dot, theta_1_ddot, X2_0, S2_2, theta_2_dot, theta_2_ddot, X3_0, S3_3, theta_3_dot, theta_3_ddot)
      y=computeI.computeY(J,sv,sa,1,computeI.Ei.E,Sbar_1_0)
      
      test_y.append(np.transpose(y))
    robot.Step(hybrid_action)
    if record_video:
      p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)

    #time.sleep(0.003)
    
    current_time = robot.GetTimeSinceReset()
  # 将A和y向量化
  test_y=np.array(test_y)
  test_torche=np.array(test_torche)
  A_vec = np.concatenate([A.flatten() for A in test_y]).reshape(-1, 30) 
  y_vec = np.concatenate([y.flatten() for y in test_torche])
  
  X=np.linalg.lstsq(A_vec,y_vec,rcond=None)[0]
  print(X)

  # 创建逻辑回归模型
  logreg = Ridge()
  # 在数据集上拟合模型
  logreg.fit(A_vec,y_vec)
  # 查看相关系数
  print(logreg.coef_)


def getM(R,P):
  M=np.zeros((4,4))
  M[0:3,0:3]=R
  M[0:3,3]=P
  M[3,3]=1
  return M


def myPoETOX(M,theta1, theta2, theta3, Sbar_1_0, Sbar_2_0, Sbar_3_0):
    SbarMatrix_3_0 = mr.VecTose3(Sbar_3_0)
    SbarMatrix_2_0 = mr.VecTose3(Sbar_2_0)
    SbarMatrix_1_0 = mr.VecTose3(Sbar_1_0)
    T=mr.MatrixExp6(SbarMatrix_1_0*theta1)@mr.MatrixExp6(SbarMatrix_2_0*theta2)@mr.MatrixExp6(SbarMatrix_3_0*theta3)@M

    AdT1=np.zeros((6,6))
    psb=T[0:3,3]
    
    relative_rotation_matrix=T[0:3,0:3]
    AdT1[0:3,0:3]=relative_rotation_matrix
    AdT1[3:6,3:6]=relative_rotation_matrix
    AdT1[3:6,0:3]=np.dot(psb,relative_rotation_matrix)
    
    return AdT1

def main(argv):
  del argv
  _run_example()


if __name__ == "__main__":
  
  app.run(main)
