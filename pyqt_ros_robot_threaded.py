import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename")
parser.add_argument("-p", "--part")
parser.add_argument("-u", "--user")

args = parser.parse_args()

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import numpy as np
import cvxpy as cp

from itertools import permutations

from cvx_ELM import *
from utils import *
from moo_cvx_ELM import *
from downsampling import *
from param_estimation import ema_debiased

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('qtagg')

import sys
import copy
from math import pi
import h5py
import time


import rospy
import roslib
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from tf.msg import tfMessage
from moveit_commander.conversions import pose_to_list

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from myviz import *

from std_msgs.msg import Int32

joint_pos = None
tf_pos = None
tf_rot = None
gr_data = None

#How the robot is understood and controlled
class MoveGroupPythonInterface(object):
  def __init__(self):
    super(MoveGroupPythonInterface, self).__init__()
    #the moveit_commander is what is responsible for sending info the moveit controllers
    print(sys.argv)
    moveit_commander.roscpp_initialize(sys.argv)
    #initialize node
    #rospy.init_node('demo_xyz_playback', anonymous=True)
    #Instantiate a `RobotCommander`_ object. Provides information such as the robot's kinematic model and the robot's current joint states
    robot = moveit_commander.RobotCommander()
    #Instantiate a `PlanningSceneInterface`_ object. This provides a remote interface for getting, setting, and updating the robot's internal understanding of the surrounding world:
    scene = moveit_commander.PlanningSceneInterface()
    #Instantiate a `MoveGroupCommander`_ object.  This object is an interface to a planning group (group of joints), which in our moveit setup is named 'manipulator'
    group_name = "manipulator"
    move_group = moveit_commander.MoveGroupCommander(group_name)
    #Create a `DisplayTrajectory`_ ROS publisher which is used to display trajectories in Rviz:
    display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory, queue_size=20)
    self.gripper_publisher = rospy.Publisher('/gripper_sends/position', Int32, queue_size=20)
 
    #Get all the info which is carried with the interface object
    #We can get the name of the reference frame for this robot:
    planning_frame = move_group.get_planning_frame()
    #print "Planning frame: %s" % planning_frame

    # We can also print the name of the end-effector link for this group:
    eef_link = move_group.get_end_effector_link()
    #print  End effector link: %s" % eef_link

    # We can get a list of all the groups in the robot:
    group_names = robot.get_group_names()
    #print  Available Planning Groups:", robot.get_group_names()
    
    global joint_pos, tf_pos, tf_rot, gr_data
    self.jp = joint_pos
    self.tp = tf_pos
    self.tr = tf_rot
    self.gr = gr_data

    # Misc variables
    self.box_name1 = ''
    self.box_name2 = ''
    self.robot = robot
    self.scene = scene
    self.move_group = move_group
    self.display_trajectory_publisher = display_trajectory_publisher
    self.planning_frame = planning_frame
    self.eef_link = eef_link
    self.group_names = group_names
    
    self.demo_name = None
    self.repro_traj = None
    
  def set_demo(self, demo_name):
    self.demo_name = demo_name
    
  def set_repro(self, repro):
    self.repro_traj = repro

  def goto_joint_state(self, js_array, i=0):
    # To start the playback of the demo, go to the initial demo position, which can be interpreted as the 0th set of joint states
    # I use joint states instead of cartesians because cartesians will fail if the current state is too far away from the goal state, whereas joint states will simply execute
    joint_goal = self.move_group.get_current_joint_values()
    print(joint_goal)
    joint_goal[0] = js_array[i][2]
    joint_goal[1] = js_array[i][1]
    joint_goal[2] = js_array[i][0]
    joint_goal[3] = js_array[i][3]
    joint_goal[4] = js_array[i][4]
    joint_goal[5] = js_array[i][5]
    # go to the initial position
    # The go command can be called with joint values, poses, or without any parameters if you have already set the pose or joint target for the group
    self.move_group.go(joint_goal, wait=True)
    # Calling ``stop()`` ensures that there is no residual movement
    self.move_group.stop()
  
  def goto_xyz(self, pos_data, rot_data, i=0):
    wpose = self.move_group.get_current_pose().pose
    wpose.position.x = -pos_data[i][0]
    wpose.position.y = -pos_data[i][1]
    wpose.position.z = pos_data[i][2]
    wpose.orientation.x = -rot_data[i][1]
    wpose.orientation.y = rot_data[i][0]
    wpose.orientation.z = rot_data[i][3]
    wpose.orientation.w = -rot_data[i][2]
    waypoints = []
    waypoints.append(copy.deepcopy(wpose))
    (start_plan, start_fraction) = self.move_group.compute_cartesian_path(waypoints, 0.001, 0.0)
    self.move_group.execute(start_plan, wait=True)
  
  def goto_xyz2(self, pos_data, rot_data):
    wpose = self.move_group.get_current_pose().pose
    wpose.position.x = -pos_data[0]
    wpose.position.y = -pos_data[1]
    wpose.position.z = pos_data[2]
    wpose.orientation.x = -rot_data[1]
    wpose.orientation.y = rot_data[0]
    wpose.orientation.z = rot_data[3]
    wpose.orientation.w = -rot_data[2]
    waypoints = []
    waypoints.append(copy.deepcopy(wpose))
    (start_plan, start_fraction) = self.move_group.compute_cartesian_path(waypoints, 0.001, 0.0)
    self.move_group.execute(start_plan, wait=True)
    
  def execute_joint_path(self, scale=1):
    #start planning demo playback by reading data from the demo.h5 file
    
    js_data = self.jp

    #### move to starting position ####
    self.goto_joint_state(js_data, 0)
    print("Press 'Enter' to start")
    input()
    
    for i in range(1, len(js_data)):
    	self.goto_joint_state(js_data, i)
 
    return 
    
  def exec_cartesian_path(self, scale=1):
    #start planning demo playback by reading data from the demo.h5 file
    
    js_data = self.jp
    rot_data = self.tr
    
    
    #### move to starting position ####
    self.goto_joint_state(js_data, 0)
    
    print('start')
    print(self.repro_traj[0, :])
    print('end')
    print(self.repro_traj[-1, :])
    (n_pts, n_dims) = np.shape(self.repro_traj)
    
    self.goto_xyz(repro_traj, rot_data)

    (n_pts_og, _) = np.shape(rot_data)
    og_moments = [0.0, 0.25, 0.5, 0.75, 1.0]
    moments = [0.0, 0.25, 0.5, 0.75, 1.0]
    true_inds = [int(moments[i] * (n_pts_og - 1)) for i in range(len(og_moments))]
    key_inds = [int(moments[i] * (n_pts - 1)) for i in range(len(moments))]
    key_rots = R.from_quat([ [rot_data[ind][0], rot_data[ind][1], rot_data[ind][2], rot_data[ind][3]] for ind in true_inds])
    #print(key_rots.as_quat())
    #print(true_inds)
    #print(key_inds)
    
    slerp = Slerp(key_inds, key_rots)

    print('Press enter to execute')
    input()
    print('Executing')
    waypoints = []
    wpose = self.move_group.get_current_pose().pose
    for i in range(1, n_pts):
      
      cur_R = slerp(np.array([i]))
      cur_quats = cur_R.as_quat()
      self.goto_xyz2(repro_traj[i, :], cur_quats[0])
    return
    
  def goto_start_position(self):
    #start planning demo playback by reading data from the demo.h5 file
    
    js_data = self.jp
    
    
    #### move to starting position ####
    print('Going to Joint State: ' + str(js_data[0]))
    self.goto_joint_state(js_data, 0)
    
  def plan_cartesian_path(self, scale=1):
    #start planning demo playback by reading data from the demo.h5 file
    
    js_data = self.jp
    rot_data = self.tr
    
    
    #### move to starting position ####
    print('Going to Joint State: ' + str(js_data[0]))
    self.goto_joint_state(js_data, 0)
    
    print('start')
    print(self.repro_traj[0, :])
    print('end')
    print(self.repro_traj[-1, :])
    (n_pts, n_dims) = np.shape(self.repro_traj)
    
    
    self.goto_xyz(self.repro_traj, rot_data)

    
    (n_pts_og, _) = np.shape(rot_data)
    og_moments = [0.0, 0.25, 0.5, 0.75, 1.0]
    moments = [0.0, 0.25, 0.5, 0.75, 1.0]
    true_inds = [int(og_moments[i] * (n_pts_og - 1)) for i in range(len(og_moments))]
    key_inds = [int(moments[i] * (n_pts - 1)) for i in range(len(moments))]
    key_rots = R.from_quat([ [rot_data[ind][0], rot_data[ind][1], rot_data[ind][2], rot_data[ind][3]] for ind in true_inds])
    print(key_rots.as_quat())
    print(true_inds)
    print(key_inds)
    
    slerp = Slerp(key_inds, key_rots)
    
    #print('Press enter to continue')
    #input()
    print('Planning')
    waypoints = []
    wpose = self.move_group.get_current_pose().pose
    for i in range(1, n_pts):
    
      wpose.position.x = -self.repro_traj[i][0] #/tf and rviz have x and y opposite signs
      wpose.position.y = -self.repro_traj[i][1] 
      wpose.position.z = self.repro_traj[i][2]
      
      cur_R = slerp(np.array([i]))
      cur_quats = cur_R.as_quat()
      #print(cur_quats)
      wpose.orientation.x = -cur_quats[0][1]
      wpose.orientation.y = cur_quats[0][0]
      wpose.orientation.z = cur_quats[0][3]
      wpose.orientation.w = -cur_quats[0][2]
      
      
      waypoints.append(copy.deepcopy(wpose))
    (plan, fraction) = self.move_group.compute_cartesian_path(
                                       waypoints,   # waypoints to follow
                                       0.01,       # eef_step
                                       0.0)       # jump_threshold
    print("Planning for " + str(fraction * 100) + "% of trajectory acheived!")
    if fraction < 0.4:
    	print("WARNING: Not enough planned!")
    	#exit()
    return plan

  def display_trajectory(self, plan):
    #ask rviz to display the trajectory
    display_trajectory = moveit_msgs.msg.DisplayTrajectory()
    display_trajectory.trajectory_start = self.robot.get_current_state()
    display_trajectory.trajectory.append(plan)
    # Publish
    self.display_trajectory_publisher.publish(display_trajectory);

  def execute_plan(self, plan):
    #execute given plan
    self.move_group.execute(plan, wait=True)
    
  def execute_plan_gripper(self, plan):
    #execute given plan
    js_data = self.jp
    rot_data = self.tr
    gr_points = self.gr
    
    #print(gr_points)
    #print(plan)
    end_time = plan.joint_trajectory.points[-1].time_from_start.secs
    #print(end_time)
    self.move_group.execute(plan, wait=False)
    for i in range(len(gr_points)):
        self.gripper_publisher.publish(int(gr_points[i][0] * 1.25))
        rospy.sleep(end_time / len(gr_points))
    
  def execute_plan_gripper2(self, plan):
    #execute given plan
    traj_data = self.repro_traj
    gr_points = self.gr
    
    self.move_group.execute(plan, wait=False)
    i = 0
    while i < len(gr_points):
        pos_msg = rospy.wait_for_message("/tf", tfMessage, timeout=10)
        if pos_msg.transforms[0].child_frame_id == 'tool0_controller':
            pos = np.array([pos_msg.transforms[0].transform.translation.x, pos_msg.transforms[0].transform.translation.y, pos_msg.transforms[0].transform.translation.z])
            #print(pos, traj_data[i], np.linalg.norm(pos - traj_data[i]))
            if np.linalg.norm(pos - traj_data[i]) < 0.01:
                self.gripper_publisher.publish(int(gr_points[i][0] * 1.25))
                i = i + 1

  def wait_for_state_update(self, box_name, box_is_known=False, box_is_attached=False, timeout=4):
    #either this times out and returns false or the object is found within the planning scene and returns true
    start = rospy.get_time()
    seconds = rospy.get_time()
    while (seconds - start < timeout) and not rospy.is_shutdown():
      attached_objects = self.scene.get_attached_objects([box_name])
      is_attached = len(attached_objects.keys()) > 0
      is_known = box_name in self.scene.get_known_object_names()
      if (box_is_attached == is_attached) and (box_is_known == is_known):
        return True
      rospy.sleep(0.1)
      seconds = rospy.get_time()
    return False


  def add_table(self, timeout=4):
    #define a box for the table below the robot
    box_pose = geometry_msgs.msg.PoseStamped()
    box_pose.header.frame_id = self.robot.get_planning_frame()
    #box origin (default = {0, 0, 0, 0, 0, 0, 0})
    box_pose.pose.orientation.w = 1.0
    box_pose.pose.position.z = -0.15
    self.box_name1 = "table"
    #add box to planning scene and specify dimensions
    self.scene.add_box(self.box_name1, box_pose, size=(10, 10, 0.1))
    #wait for the box to be added in or to timeout
    return self.wait_for_state_update(self.box_name1, box_is_known=True, timeout=timeout)

  def add_wall(self, timeout=4):
    #Same as above with different dimensions
    box_pose = geometry_msgs.msg.PoseStamped()
    box_pose.header.frame_id = self.robot.get_planning_frame()
    box_pose.pose.orientation.w = 1.0
    box_pose.pose.position.y = -0.3 # next to the robot
    self.box_name2 = "wall"
    self.scene.add_box(self.box_name2, box_pose, size=(10, 0.02, 10))
    return self.wait_for_state_update(self.box_name2, box_is_known=True, timeout=timeout)
 
  def remove_workspace(self, timeout=4):
    #remove each object from the planning scene, waiting for scene to update before moving on
    self.scene.remove_world_object(self.box_name1)
    self.wait_for_state_update(self.box_name1, box_is_attached=False, box_is_known=False, timeout=timeout)
    self.scene.remove_world_object(self.box_name2)
    return self.wait_for_state_update(self.box_name2, box_is_attached=False, box_is_known=False, timeout=timeout) 

    
def read_data(filename):
    global joint_pos, tf_pos, tf_rot, gr_data
    hf = h5py.File(filename, 'r')
    print(list(hf.keys()))
    js_info = hf.get('joint_state_info')
    print(list(js_info.keys()))
    joint_pos = np.array(js_info.get('joint_positions'))
    tf_info = hf.get('transform_info')
    print(list(tf_info.keys()))
    tf_pos = np.array(tf_info.get('transform_positions'))
    tf_rot = np.array(tf_info.get('transform_orientations'))
    gr_info = hf.get('gripper_info')
    gr_data = gr_info.get('gripper_position')
    gr_data = np.array(gr_data)
    return joint_pos, tf_pos, tf_rot, gr_data
    
def read_data_ds(filename, n=100):
    global joint_pos, tf_pos, tf_rot, gr_data
    hf = h5py.File(filename, 'r')
    print(list(hf.keys()))
    js_info = hf.get('joint_state_info')
    print(list(js_info.keys()))
    joint_pos = np.array(js_info.get('joint_positions'))
    tf_info = hf.get('transform_info')
    print(list(tf_info.keys()))
    tf_pos = np.array(tf_info.get('transform_positions'))
    tf_rot = np.array(tf_info.get('transform_orientations'))
    gr_info = hf.get('gripper_info')
    gr_data = gr_info.get('gripper_position')
    gr_data = np.array(gr_data)
    tf_pos, ds_inds = db_downsample_inds(tf_pos, n)
    joint_pos = joint_pos[ds_inds, :]
    tf_rot = tf_rot[ds_inds, :]
    gr_data = gr_data[ds_inds, :]
    return joint_pos, tf_pos, tf_rot, gr_data

def random_voting(F_normalized, num_out=3, pref=None):
    num_can, f_dim = np.shape(F_normalized)
    cands = []
    if pref is None:
        cands.append(np.argmin(np.linalg.norm(F_normalized, axis=1)))
    else:
        cands.append(np.argmin(np.linalg.norm(F_normalized - pref, axis=1)))
    while len(cands) < num_out:
        idx = np.random.randint(0, num_can)
        if not idx in cands:
            cands.append(idx)
    return cands

#maximize distance
def first_round_voting(F_normalized, num_out=3, pref=None):
    num_can, f_dim = np.shape(F_normalized)
    cands = []
    if pref is None:
        cands.append(np.argmin(np.linalg.norm(F_normalized, axis=1)))
    else:
        cands.append(np.argmin(np.linalg.norm(F_normalized - pref, axis=1)))
    while len(cands) < num_out:
        cand_dist = np.zeros((num_can))
        for i in range(num_can):
            if not i in cands:
                #cand_dist[i] = np.linalg.norm(F_normalized[i] - F_normalized[cands[0]])
                #for cand in cands:
                    #cand_dist[i] = min(cand_dist[i], np.linalg.norm(F_normalized[i] - F_normalized[cand]))
                cand_dist[i] = np.linalg.norm([F_normalized[i] - F_normalized[cand] for cand in cands])
        #print(cand_dist)
        #print(cands)
        cands.append(np.argmax(cand_dist))
        #print(cands)
    return cands
    
def softmax_stable(X):
    Xp = X - max(X)
    return np.exp(Xp) / np.sum(np.exp(Xp))
    
#probabilistically maximize distance
def first_round_voting_prob(F_normalized, num_out=3, pref=None):
    num_can, f_dim = np.shape(F_normalized)
    cands = []
    if pref is None:
        cand_probs = softmax_stable(np.linalg.norm(F_normalized, axis=1))
        cands.append(np.random.choice(len(cand_probs), p=cand_probs))
    else:
        cands.append(np.argmin(np.linalg.norm(F_normalized - pref, axis=1)))
    while len(cands) < num_out:
        cand_dist = np.zeros((num_can))
        for i in range(num_can):
            if not i in cands:
                #cand_dist[i] = np.linalg.norm(F_normalized[i] - F_normalized[cands[0]])
                #for cand in cands:
                    #cand_dist[i] = min(cand_dist[i], np.linalg.norm(F_normalized[i] - F_normalized[cand]))
                cand_dist[i] = np.linalg.norm([F_normalized[i] - F_normalized[cand] for cand in cands])
        #print(cand_dist)
        #print(cands)
        cand_probs = softmax_stable(cand_dist)
        cands.append(np.random.choice(len(cand_probs), p=cand_probs))
        #print(cands)
    return cands

#minimize distance
def second_round_voting(F_normalized, num_out=3, pref=None):
    num_can, f_dim = np.shape(F_normalized)
    cands = []
    if pref is None:
        cands.append(np.argmin(np.linalg.norm(F_normalized, axis=1)))
    else:
        cands.append(np.argmin(np.linalg.norm(F_normalized - pref, axis=1)))
    while len(cands) < num_out:
        cand_dist = np.ones((num_can))
        for i in range(num_can):
            if not i in cands:
                cand_dist[i] = np.linalg.norm([F_normalized[i] - F_normalized[cand] for cand in cands])
        cands.append(np.argmin(cand_dist))
    return cands


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Trajectory Selection")
        self.setGeometry(100, 100, 1200, 400)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)


class PlotCanvas(QWidget):
    def __init__(self, fig):
        super().__init__()

        self.fig = fig
        self.canvas = self.fig.canvas

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.layout.addWidget(self.canvas)

class Optimization_UI(object):

    def __init__(self, demo, inds=[], csts=[], objective_names=[], rviz=False, demo_name='', part=0):
        self.demo = demo
        self.n_pts, self.n_dims = np.shape(self.demo)
        self.inds = inds
        self.csts = csts
        self.n_csts = len(inds)
        self.objective_names = objective_names
        self.F_normalized = []
        self.est_prefs = []
        self.cur_pref = None
        self.beta = 0.9
        self.selections_rnd1 = []
        self.selections_rnd2 = []
        
        self.part = int(part)
        
        self.command_set1 = [self.sel_idx1, self.sel_idx2, self.sel_idx3]
        self.command_set2 = [self.final_sel_idx1, self.final_sel_idx2, self.final_sel_idx3]
        
        self.app = QApplication(sys.argv)
        self.rviz = rviz
        if self.rviz:
            print("MAKE SURE TERMINAL IS SOURCED")
            rospy.init_node('hri_interface', anonymous=True)
            self.move_group = MoveGroupPythonInterface()
            #self.move_group.add_table()
            #self.move_group.add_wall()
            self.move_group.set_demo(demo_name)
            self.mv = MyViz()
        self.app.setStyleSheet('.QLabel { font-size: 14pt;}')
        self.window = MainWindow()
        self.window.setFixedWidth(1280)
        self.window.setFixedHeight(980)
        
        if self.part == 2:
            print("Loading Previous Data")
            self.selections_rnd1 = np.loadtxt('rnd1_sels' + str(args.user) + '.txt').tolist()
            self.selections_rnd2 = np.loadtxt('rnd2_sels' + str(args.user) + '.txt').tolist()
            self.est_prefs = np.loadtxt('est_prefs' + str(args.user) + '.txt').tolist()
            self.cur_pref = np.loadtxt('cur_pref' + str(args.user) + '.txt').tolist()
        
    def set_problem(self, demo, inds=[], csts=[]):
        self.demo = demo
        self.n_pts, self.n_dims = np.shape(self.demo)
        self.inds = inds
        self.csts = csts
        self.n_csts = len(inds)
        
    def solve_opt(self):
        problem = meta_opt(self.demo)
        problem.set_constraints(self.inds, self.csts)
        algorithm = NSGA2()
        termination = get_termination("n_eval", 1000)
        
        res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               verbose=True)
               
        self.X = res.X
        self.F = res.F
        print("Shape X", np.shape(self.X))
        print("Shape F", np.shape(self.F))
        num_can, f_dim = np.shape(self.F)
        self.F_normalized = np.zeros((num_can, f_dim))
        for i in range(f_dim):
            self.F_normalized[:, i] = (self.F[:, i] - min(self.F[:, i])) / max(self.F[:, i])
        
    def get_sol(self, x_idx):
        params = self.X[x_idx]
        print(params)
        PA = ElMap_Perturbation_Analysis(self.demo, spatial=params[0], shape=params[1], tangent=params[2], stretch=params[3], bend=params[4])
        x_prob = PA.setup_problem()
        constraints = []
        for i in range(self.n_csts):
            for j in range(self.n_dims):
                constraints.append( cp.abs(x_prob[self.inds[i] + (PA.n_pts * j)] - self.csts[i, j]) <= 0 )
        sol = PA.solve_problem(constraints, disp=False)
        return sol
    
    def get_traj_str(self, idx):
        obj_results = self.F_normalized[idx]
        out_str = "Metric Evaluation:\n"
        for i in range(len(obj_results)):
            out_str = out_str + self.objective_names[i] + ": "
            if obj_results[i] > (2/3):
                out_str = out_str + "bad (" + str(round((1-obj_results[i]) * 100)) + "%)\n"
            elif obj_results[i] < (1/3):
                out_str = out_str + "good (" + str(round((1-obj_results[i]) * 100)) + "%)\n"
            else:
                out_str = out_str + "OK (" + str(round((1-obj_results[i]) * 100)) + "%)\n"
        return out_str
    
    def disconnect_buttons(self):
        for i in range(len(self.buttons)):
            self.buttons[i].disconnect()
    
    def dummy_cmd1(self):
        print('Button1 Selected')
        
    def dummy_cmd2(self):
        print('Button2 Selected')
        
    def dummy_cmd3(self):
        print('Button3 Selected')
        
    def next_trial(self):
        self.move_group.goto_start_position()
        print("Part", self.part)
        print("Num Trials", len(self.selections_rnd2))
        if int(self.part) == 1:
            np.savetxt('rnd1_sels' + str(args.user) + '.txt', self.selections_rnd1)
            np.savetxt('rnd2_sels' + str(args.user) + '.txt', self.selections_rnd2)
            np.savetxt('est_prefs' + str(args.user) + '.txt', self.est_prefs)
            np.savetxt('cur_pref' + str(args.user) + '.txt', self.cur_pref)
            if len(self.selections_rnd2) > 4:
                self.cancel()
        if int(self.part) == 2:
            np.savetxt('rnd1_sels' + str(args.user) + 'p2.txt', self.selections_rnd1)
            np.savetxt('rnd2_sels' + str(args.user) + 'p2.txt', self.selections_rnd2)
            np.savetxt('est_prefs' + str(args.user) + 'p2.txt', self.est_prefs)
            np.savetxt('cur_pref' + str(args.user) + 'p2.txt', self.cur_pref)
            if len(self.selections_rnd2) > 9:
                self.cancel()
        #cands = first_round_voting(self.F_normalized, pref=self.cur_pref)
        cands = first_round_voting_prob(self.F_normalized, pref=self.cur_pref)
        #cands = random_voting(UI.F_normalized, pref=UI.cur_pref)
        print("Candidates", cands)
        self.create_UI(cands, ['Option 1: Based on Previous Selection', 'Option 2: Minimizes Objectives', 'Option 3: Minimizes Objectives'], commands=self.command_set1)
        
    def final_sel_idx1(self):
        self.window.setStyleSheet("background-color: darkRed;")
        self.title_label.setText("Please wait while the robot executes...") 
        self.disconnect_buttons()
        self.app.processEvents()
        
        self.selections_rnd2.append(1)
        new_ind = self.sel_inds2[0]
        
        traj = self.cur_sel_trajs[0]
        self.move_group.set_repro(traj)
        plan = self.move_group.plan_cartesian_path()
        #self.move_group.display_trajectory(plan)
        self.move_group.execute_plan_gripper2(plan)
        
        self.est_prefs.append(self.F_normalized[new_ind])
        self.calculate_pref()
        self.next_trial()
        
    def final_sel_idx2(self):
        self.window.setStyleSheet("background-color: darkRed;")
        self.title_label.setText("Please wait while the robot executes...") 
        self.disconnect_buttons()
        self.app.processEvents()
        
        self.selections_rnd2.append(2)
        new_ind = self.sel_inds2[1]
        
        traj = self.cur_sel_trajs[1]
        self.move_group.set_repro(traj)
        plan = self.move_group.plan_cartesian_path()
        #self.move_group.display_trajectory(plan)
        self.move_group.execute_plan_gripper2(plan)
        
        self.est_prefs.append(self.F_normalized[new_ind])
        self.calculate_pref()
        self.next_trial()
        
    def final_sel_idx3(self):
        self.window.setStyleSheet("background-color: darkRed;")
        self.title_label.setText("Please wait while the robot executes...") 
        self.disconnect_buttons()
        self.app.processEvents()
        
        self.selections_rnd2.append(3)
        new_ind = self.sel_inds2[2]
        
        traj = self.cur_sel_trajs[2]
        self.move_group.set_repro(traj)
        plan = self.move_group.plan_cartesian_path()
        #self.move_group.display_trajectory(plan)
        self.move_group.execute_plan_gripper2(plan)
        
        self.est_prefs.append(self.F_normalized[new_ind])
        self.calculate_pref()
        self.next_trial()
        
    def calculate_pref(self):
        est_prefs_arr = np.array(self.est_prefs)
        print('Preference estimation')
        print(est_prefs_arr)
        self.cur_pref = []
        for i in range(np.shape(self.F_normalized)[1]):
            self.cur_pref.append(ema_debiased(est_prefs_arr[:, i], self.beta)[-1])
        print('Current Preference')
        print(self.cur_pref)
        
    def sel_idx1(self):
        self.disconnect_buttons()
        self.selections_rnd1.append(1)
        new_ind = self.sel_inds[0]
        #self.window.close()
        #self.mv.close()
        #self.mv = MyViz()
        #plt.close('all')
        new_cands = second_round_voting(self.F_normalized, pref=self.F_normalized[new_ind])
        print("New Candidates", new_cands)
        self.create_UI2(new_cands, ['Option 1: Previous Selection', 'Option 2: Similar to Previous Selection', 'Option 3: Similar to Previous Selection'], commands=self.command_set2, main_title='Are You Sure? Select Your Preferred Option\n')
        
    def sel_idx2(self):
        self.disconnect_buttons()
        self.selections_rnd1.append(2)
        new_ind = self.sel_inds[1]
        #self.window.close()
        #self.mv.close()
        #self.mv = MyViz()
        #plt.close('all')
        new_cands = second_round_voting(self.F_normalized, pref=self.F_normalized[new_ind])
        print("New Candidates", new_cands)
        self.create_UI2(new_cands, ['Option 1: Previous Selection', 'Option 2: Similar to Previous Selection', 'Option 3: Similar to Previous Selection'], commands=self.command_set2, main_title='Are You Sure? Select Your Preferred Option\n')
        
    def sel_idx3(self):
        self.disconnect_buttons()
        self.selections_rnd1.append(3)
        new_ind = self.sel_inds[2]
        #self.window.close()
        #self.mv.close()
        #self.mv = MyViz()
        #plt.close('all')
        new_cands = second_round_voting(self.F_normalized, pref=self.F_normalized[new_ind])
        print("New Candidates", new_cands)
        self.create_UI2(new_cands, ['Option 1: Previous Selection', 'Option 2: Similar to Previous Selection', 'Option 3: Similar to Previous Selection'], commands=self.command_set2, main_title='Are You Sure? Select Your Preferred Option\n')
        
    def prev_idx1(self):
        new_ind = self.sel_inds[0]
        traj = self.cur_sel_trajs[0]
        self.move_group.set_repro(traj)
        plan = self.move_group.plan_cartesian_path()
        self.move_group.display_trajectory(plan)
        
    def prev_idx2(self):
        new_ind = self.sel_inds[1]
        traj = self.cur_sel_trajs[1]
        self.move_group.set_repro(traj)
        plan = self.move_group.plan_cartesian_path()
        self.move_group.display_trajectory(plan)
        
    def prev_idx3(self):
        new_ind = self.sel_inds[2]
        traj = self.cur_sel_trajs[2]
        self.move_group.set_repro(traj)
        plan = self.move_group.plan_cartesian_path()
        self.move_group.display_trajectory(plan)
    
    def cancel(self):
        if self.rviz:
            self.move_group.remove_workspace()
        self.window.close()
        
    def back(self):
        print("BACK")
        self.disconnect_buttons()
        self.selections_rnd1.pop()
        self.create_UI(self.sel_inds, self.title_strs, commands=self.command_set1)
    
    def base_UI(self, n=3):
        commands = [self.dummy_cmd1, self.dummy_cmd2, self.dummy_cmd3]
        prev_commands = [self.prev_idx1, self.prev_idx2, self.prev_idx3]
        self.title_labels = []
        self.figures = []
        self.axs = []
        self.canvases = []
        self.desc_labels = []
        self.buttons = []
        self.prev_buttons = []
        for i in range(n):
            j = i + 1
            self.title_labels.append(QLabel())
            self.title_labels[i].setText('Title ' + str(j))
            self.title_labels[i].setAlignment(Qt.AlignCenter)
            
            self.figures.append(plt.figure(figsize=(4, 3), dpi=100))
            self.canvases.append(PlotCanvas(self.figures[i]))
            if self.n_dims == 2:
                self.axs.append(self.figures[i].add_subplot(111))
            elif self.n_dims == 3:
                self.axs.append(self.figures[i].add_subplot(111, projection='3d'))
            #self.axs[i].plot(np.arange(10), j*np.arange(10))
            
            self.desc_labels.append(QLabel())
            self.desc_labels[i].setText('Description ' + str(j))
            self.desc_labels[i].setAlignment(Qt.AlignLeft)
            
            self.buttons.append(QPushButton("Select Option " + str(j)))
            self.buttons[i].clicked.connect(commands[i])
            
            self.prev_buttons.append(QPushButton("Preview Option " + str(j)))
            self.prev_buttons[i].clicked.connect(prev_commands[i])
        
        self.outer_outer_layout = QVBoxLayout() 
        self.title_label = QLabel()
        self.title_label.setText('Main Title') 
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 32pt;")
        self.outer_outer_layout.addWidget(self.title_label)
        
        self.outer_layout = QHBoxLayout()    
        
        
        for i in range(n):
            border_frame = QFrame()
            border_frame.setFrameShape(QFrame.Box)  # Set the frame shape to Box
            border_frame.setLineWidth(2)            # Set the border width
            border_frame.setLayout(QVBoxLayout())   # Set layout for the frame
            #border_frame.addWidget(border_frame)
            border_frame.layout().addWidget(self.title_labels[i])
            border_frame.layout().addWidget(self.canvases[i])
            border_frame.layout().addWidget(self.desc_labels[i])
            border_frame.layout().addWidget(self.buttons[i])
            if self.rviz:
                border_frame.layout().addWidget(self.prev_buttons[i])
            #inner_layout.setStyleSheet("border: 2px solid black;")
            inner_layout = QVBoxLayout()
            inner_layout.addWidget(border_frame)
            self.outer_layout.addLayout(inner_layout)
            
        self.outer_outer_layout.addLayout(self.outer_layout)
        if self.rviz:
            self.outer_outer_layout.addWidget(self.mv)
        self.cancel_back_button = QPushButton("Cancel")
        self.cancel_back_button.clicked.connect(self.cancel)
        self.outer_outer_layout.addWidget(self.cancel_back_button)
        
        self.window.central_widget.setLayout(self.outer_outer_layout)
        self.window.show()
    
    def create_UI(self, sol_inds, title_strs, commands, main_title='Select a Preferred Reproduction\n'):
        self.window.setStyleSheet("background-color: #d9dbdb;")
        self.cancel_back_button.clicked.disconnect()
        self.cancel_back_button.clicked.connect(self.cancel)
        self.cancel_back_button.setText("Cancel")
        print("Solution Inds", sol_inds)
        self.sel_inds = sol_inds
        self.title_strs = title_strs
        
        self.move_group.goto_start_position()
        
        self.cur_sel_trajs = []
        
        self.title_label.setText(main_title) 
        
        # Create three subframes to organize the plots and buttons
        for i in range(len(sol_inds)):
            
            title_str = title_strs[i]
            self.title_labels[i].setText(title_str)
            
            self.figures[i].clear()
            if self.n_dims == 2:
                self.axs[i] = self.figures[i].add_subplot(111)
                self.axs[i].plot(self.demo[:, 0], self.demo[:, 1], 'k--', label="Demo")
                traj = self.get_sol(sol_inds[i])
                self.cur_sel_trajs.append(traj)
                self.axs[i].plot(traj[:, 0], traj[:, 1], 'r', label="Repro")
                if self.n_csts > 0:
                    self.axs[i].plot(self.csts[0, 0], self.csts[0, 1], 'k.', ms=10, label="Consts")
                    for j in range(self.n_csts):
                        self.axs[i].plot(self.csts[j, 0], self.csts[j, 1], 'k.', ms=10)
            if self.n_dims == 3:
                self.axs[i] = self.figures[i].add_subplot(111, projection='3d')
                self.axs[i].plot(self.demo[:, 0], self.demo[:, 1], self.demo[:, 2], 'k--', label="Demo")
                traj = self.get_sol(sol_inds[i])
                self.cur_sel_trajs.append(traj)
                np.savetxt("traj" + str(sol_inds[i]) + ".txt", traj)
                self.axs[i].plot(traj[:, 0], traj[:, 1], traj[:, 2], 'r', label="Repro")
                if self.n_csts > 0:
                    self.axs[i].plot(self.csts[0, 0], self.csts[0, 1], self.csts[0, 2], 'k.', ms=10, label="Consts")
                    for j in range(self.n_csts):
                        self.axs[i].plot(self.csts[j, 0], self.csts[j, 1], self.csts[j, 2], 'k.', ms=10)
            self.axs[i].legend()
            self.canvases[i].canvas.draw()
            
            plot_str = self.get_traj_str(sol_inds[i])
            self.desc_labels[i].setText(plot_str)
            
            self.buttons[i].clicked.connect(commands[i])
    
    def create_UI2(self, sol_inds, title_strs, commands, main_title='Select a Preferred Reproduction\n'):
        self.window.setStyleSheet("background-color: #97f0e5;")
        self.cancel_back_button.clicked.disconnect()
        self.cancel_back_button.clicked.connect(self.back)
        self.cancel_back_button.setText("Back")
        print("Solution Inds", sol_inds)
        self.sel_inds2 = sol_inds
        self.title_strs2 = title_strs
        
        self.cur_sel_trajs = []
        
        self.title_label.setText(main_title) 
        
        # Create three subframes to organize the plots and buttons
        for i in range(len(sol_inds)):
            
            title_str = title_strs[i]
            self.title_labels[i].setText(title_str)
            
            self.figures[i].clear()
            if self.n_dims == 2:
                self.axs[i] = self.figures[i].add_subplot(111)
                self.axs[i].plot(self.demo[:, 0], self.demo[:, 1], 'k--', label="Demo")
                traj = self.get_sol(sol_inds[i])
                self.cur_sel_trajs.append(traj)
                self.axs[i].plot(traj[:, 0], traj[:, 1], 'r', label="Repro")
                if self.n_csts > 0:
                    self.axs[i].plot(self.csts[0, 0], self.csts[0, 1], 'k.', ms=10, label="Consts")
                    for j in range(self.n_csts):
                        self.axs[i].plot(self.csts[j, 0], self.csts[j, 1], 'k.', ms=10)
            if self.n_dims == 3:
                self.axs[i] = self.figures[i].add_subplot(111, projection='3d')
                self.axs[i].plot(self.demo[:, 0], self.demo[:, 1], self.demo[:, 2], 'k--', label="Demo")
                traj = self.get_sol(sol_inds[i])
                self.cur_sel_trajs.append(traj)
                np.savetxt("traj" + str(sol_inds[i]) + ".txt", traj)
                self.axs[i].plot(traj[:, 0], traj[:, 1], traj[:, 2], 'r', label="Repro")
                if self.n_csts > 0:
                    self.axs[i].plot(self.csts[0, 0], self.csts[0, 1], self.csts[0, 2], 'k.', ms=10, label="Consts")
                    for j in range(self.n_csts):
                        self.axs[i].plot(self.csts[j, 0], self.csts[j, 1], self.csts[j, 2], 'k.', ms=10)
            self.axs[i].legend()
            self.canvases[i].canvas.draw()
            
            plot_str = self.get_traj_str(sol_inds[i])
            self.desc_labels[i].setText(plot_str)
            
            self.buttons[i].clicked.connect(commands[i])    
    
    
    
def get_pos_csts(tf_pos):
    inds = []
    n_pts, n_dims = np.shape(tf_pos)
    for d in range(n_dims):
        inds.append(np.argmin(tf_pos[:, d]))
        inds.append(np.argmax(tf_pos[:, d]))
    return inds
    
def get_gripper_csts(gripper_data):
    inds = []
    for i in range(1, len(gripper_data)-1):
        #check that of the 3 data points, 2 are the same but the third is different
        if gripper_data[i-1] != gripper_data[i+1]:
            if gripper_data[i-1] == gripper_data[i] or gripper_data[i+1] == gripper_data[i]:
                inds.append(i)
    return inds
    
def main_exp():
    global joint_pos, tf_pos, tf_rot, gr_data
    # demonstration
    #traj = read_RAIL_demo('PUSHING', 1, 1)
    filename = args.filename
    joint_pos, tf_pos, tf_rot, gr_data = read_data_ds(filename, n=100)
    
    x_demo = tf_pos
    added_csts = get_gripper_csts(gr_data)
    pos_csts = get_pos_csts(tf_pos)
    print(np.shape(x_demo))
    n_pts, n_dims = np.shape(x_demo)
    #added_csts = [added_csts[0]]
    inds = [0] + added_csts + pos_csts + [n_pts-1]
    inds.sort()
    inds = list(set(inds))
    csts = np.array([x_demo[i] for i in inds])
    print("Constraints", inds, csts)
    UI = Optimization_UI(x_demo, inds, csts, objective_names=["Spatial Similarity", "Angular Similarity", "Smoothness"], rviz=True, demo_name=filename, part=args.part)
    UI.solve_opt()
    #cands = first_round_voting(UI.F_normalized, pref=UI.cur_pref)
    cands = first_round_voting_prob(UI.F_normalized, pref=UI.cur_pref)
    #cands = random_voting(UI.F_normalized, pref=UI.cur_pref)
    print(cands)
    UI.base_UI()
    if int(args.part) == 1:
        UI.create_UI(cands, ['Option 1: Minimizes Objectives', 'Option 2: Minimizes Objectives', 'Option 3: Minimizes Objectives'], commands=UI.command_set1)
    else:
        UI.create_UI(cands, ['Option 1: Based on Previous Selection', 'Option 2: Minimizes Objectives', 'Option 3: Minimizes Objectives'], commands=UI.command_set1)
    UI.app.exec_()
    print("round 1 selections")
    print(UI.selections_rnd1)
    print("round 2 selections")
    print(UI.selections_rnd2)
    
if __name__ == '__main__':
    main_exp()
