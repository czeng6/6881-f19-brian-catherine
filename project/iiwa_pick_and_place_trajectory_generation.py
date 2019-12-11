import numpy as np
from pydrake.multibody import inverse_kinematics
from pydrake.trajectories import PiecewisePolynomial

from pydrake.math import RollPitchYaw, RotationMatrix, RigidTransform
from pydrake.examples.manipulation_station import ManipulationStation
import pydrake.solvers.mathematicalprogram as mp
from manip_station_sim.plan_utils import ConnectPointsWithCubicPolynomial

# starting position of point Q in world frame.
p_WQ_home = np.array([0.5, 0, 0.4])

# position of point Q in EE frame.  Point Q is fixed in the EE frame.
p_L7Q = np.array([0., 0., 0.21])

# robot home position
q_home = np.array([0, 0, 0, -1.5, 0, 1.0, 0])


def GenerateIiwaTrajectoriesAndGripperSetPoints(X_WO, is_printing=True):
    """
    :param X_WO: the pose of the foam brick in world frame.
    :param is_printing: set to True to print IK solution results.
    :return: 1. a list of joint space trajectories.
        2. a list of gripper set points.
        The two lists must have the same length.
    """
    station = ManipulationStation()
    station.SetupManipulationClassStation()
    station.Finalize()
    plant = station.get_controller_plant()

    # Go to p_WQ_home
    ik_iiwa = inverse_kinematics.InverseKinematics(plant)
    world_frame = plant.world_frame()
    l7_frame = plant.GetFrameByName("iiwa_link_7")

    theta_bound = 0.005 * np.pi
    R_WL7 = RollPitchYaw(0, np.pi * 5 / 6, 0).ToRotationMatrix()

    ik_iiwa.AddOrientationConstraint(
        frameAbar=world_frame, R_AbarA=R_WL7,   # rotate end effector to R_WL7 angle 
        frameBbar=l7_frame, R_BbarB=RotationMatrix(),
        theta_bound=theta_bound)

    ik_iiwa.AddPositionConstraint(
        frameB=l7_frame, p_BQ=p_L7Q,
        frameA=world_frame,
        p_AQ_lower=p_WQ_home - 0.01,  #p_WQ_home corresponds to q_val_0's pos 
        p_AQ_upper=p_WQ_home + 0.01)

    prog = ik_iiwa.prog()
    prog.SetInitialGuess(ik_iiwa.q(), np.zeros(7))
    result = mp.Solve(prog)
    if is_printing:
        print(result.get_solution_result())
    q_val_0 = result.GetSolution(ik_iiwa.q())

    qtraj_leave_home = ConnectPointsWithCubicPolynomial(q_home, q_val_0, 2.0) 

    # close fingers
    q_knots = np.zeros((2, 7))
    q_knots[0] = qtraj_leave_home.value(qtraj_leave_home.end_time()).squeeze()
    q_knots[1] = q_knots[0]
    qtraj_open_gripper = PiecewisePolynomial.ZeroOrderHold([0, 1], q_knots.T)

    # Complete your pick and place trajectories.
    def MoveEndEffectorAlongStraightLine(
        p_WQ_start, p_WQ_end, duration, get_orientation, q_initial_guess, num_knots):
        """
        p_WQ_start is the starting position of point Q (the point Q we defined earlier!)
        p_WQ_end is the end position of point Q.
        duration is the duration of the trajectory measured in seconds. 
        get_orientation(i, n) is a function that returns the interpolated orientation R_WL7 at the i-th knot point out of n knot points. It can simply return a constant orientation, but changing the orientation of link 7 should be helpful for placing the object on the shelf.
        num_knots is the number of knot points in the trajectory. Generally speaking,the larger num_knots is, the smoother the trajectory, but solving for more knot points also takes longer. You can start by setting num_knots to 10, which should be sufficient for the task. 
        """
        t_knots = np.linspace(0, duration, num_knots+1)
        q_knots = np.zeros((num_knots, plant.num_positions()))
        q_knots[0] = q_initial_guess
        n = len(q_initial_guess)

        for i in range(num_knots):
            ik = inverse_kinematics.InverseKinematics(plant)
            prog = ik.prog()
            p_AQ = p_WQ_start + (p_WQ_end - p_WQ_start) * i / (num_knots - 1)
            ik.AddPositionConstraint(frameB=l7_frame, p_BQ=p_L7Q, frameA=world_frame, p_AQ_lower=p_AQ - 0.01, p_AQ_upper=p_AQ + 0.01) # interpolate between p_WQ_start and p_WQ_end
            ik.AddOrientationConstraint(frameAbar=world_frame, R_AbarA=get_orientation(i, num_knots), frameBbar=l7_frame, R_BbarB=RotationMatrix(), theta_bound=theta_bound) # call get_orientation(i, num_knots).
            if i == 0:
                prog.SetInitialGuess(ik.q(), q_initial_guess)
            else:
                # This is very important for the smoothness of the whole trajectory.
                prog.SetInitialGuess(ik.q(), q_knots[i - 1])
            result = mp.Solve(ik.prog())												 
            assert result.get_solution_result() == mp.SolutionResult.kSolutionFound
            q_knots[i] = result.GetSolution(ik.q())

        return PiecewisePolynomial.Cubic(t_knots[1:], q_knots.T, np.zeros(n), np.zeros(n)) # return a cubic spline that connects all q_knots. 

    q_curr = q_knots[1]
    p_mid = np.array([0.53, 0.02, 0.38])
    a_box = 0.14
    tilt_link_7 = lambda i, n: RollPitchYaw(a_box*i/(n-1), 5/6*np.pi, 0).ToRotationMatrix()
    RPY = RollPitchYaw(a_box, np.pi * 5 / 6, 0).ToRotationMatrix()
    qtraj_down = MoveEndEffectorAlongStraightLine(p_WQ_home, p_mid, 3, tilt_link_7, q_curr, 10)

    q_curr = qtraj_down.value(qtraj_down.end_time()).squeeze()
    p_box = np.array([0.52, 0.033, 0.18])
    qtraj_down2 = MoveEndEffectorAlongStraightLine(p_mid, p_box, 4, lambda i, n: RPY, q_curr, 10)

    q_curr = qtraj_down2.value(qtraj_down2.end_time()).squeeze()
    p_up = np.array([0.53, 0.033, 0.38])
    qtraj_up = MoveEndEffectorAlongStraightLine(p_box, p_up, 4, lambda i, n: RPY, q_curr, 10)

    q_curr = qtraj_up.value(qtraj_up.end_time()).squeeze()
    p_down = np.array([0.52, 0.033, 0.21])
    qtraj_backdown = MoveEndEffectorAlongStraightLine(p_up, p_down, 4, lambda i, n: RPY, q_curr, 10)

    q_curr = qtraj_backdown.value(qtraj_backdown.end_time()).squeeze()
    qtraj_return = ConnectPointsWithCubicPolynomial(q_curr, q_home, 3.0)


    q_traj_list = [qtraj_leave_home,
                   qtraj_open_gripper,
                   qtraj_down,
                   qtraj_down2,
                   qtraj_up,
                   qtraj_backdown,
                   qtraj_return
                   ]
    gripper_setpoint_list = [0.01, 0.1, 0.1, 0.1, 0.029, 0.029, 0.1]

    return q_traj_list, gripper_setpoint_list

