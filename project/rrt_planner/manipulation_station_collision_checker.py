import numpy as np

from pydrake.examples.manipulation_station import ManipulationStation
from pydrake.multibody.plant import MultibodyPlant
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.meshcat_visualizer import MeshcatVisualizer
from pydrake.examples.manipulation_station import IiwaCollisionModel

from manip_station_sim.plan_utils import RenderSystemWithGraphviz


class ManipulationStationCollisionChecker:
    def __init__(self, is_visualizing=False):
        self.station = ManipulationStation()
        self.station.SetupManipulationClassStation(
            IiwaCollisionModel.kBoxCollision)
        self.station.Finalize()
        self.plant = self.station.get_mutable_multibody_plant()
        self.scene_graph = self.station.get_mutable_scene_graph()
        self.is_visualizing = is_visualizing

        # scene graph query output port.
        self.query_output_port = self.scene_graph.GetOutputPort("query")

        builder = DiagramBuilder()
        builder.AddSystem(self.station)
        # meshcat visualizer
        if is_visualizing:
            self.viz = MeshcatVisualizer(self.scene_graph,
                                         zmq_url="tcp://127.0.0.1:6000")
            # connect meshcat to manipulation station
            builder.AddSystem(self.viz)
            builder.Connect(self.station.GetOutputPort("pose_bundle"),
                            self.viz.GetInputPort("lcm_visualization"))

        self.diagram = builder.Build()

        # contexts
        self.context_diagram = self.diagram.CreateDefaultContext()
        self.context_station = self.diagram.GetSubsystemContext(
            self.station, self.context_diagram)
        self.context_scene_graph = self.station.GetSubsystemContext(
            self.scene_graph, self.context_station)
        self.context_plant = self.station.GetMutableSubsystemContext(
            self.plant, self.context_station)

        if is_visualizing:
            self.viz.load()
            self.context_meshcat = self.diagram.GetSubsystemContext(
                self.viz, self.context_diagram)

    def SetStationConfiguration(self, q_iiwa, gripper_setpoint, left_door_angle,
                                right_door_angle):
        """
        :param q_iiwa: (7,) numpy array, joint angle of robots in radian.
        :param gripper_setpoint: float, gripper opening distance in meters.
        :param left_door_angle: float, left door hinge angle, \in [0, pi/2].
        :param right_door_angle: float, right door hinge angle, \in [0, pi/2].
        :return:
        """
        self.station.SetIiwaPosition(self.context_station, q_iiwa)
        self.station.SetWsgPosition(self.context_station, gripper_setpoint)

        # cabinet doors
        if left_door_angle > 0:
            left_door_angle *= -1
        left_hinge_joint = self.plant.GetJointByName("left_door_hinge")
        left_hinge_joint.set_angle(context=self.context_plant,
                                   angle=left_door_angle)

        right_hinge_joint = self.plant.GetJointByName("right_door_hinge")
        right_hinge_joint.set_angle(context=self.context_plant,
                                    angle=right_door_angle)

    def DrawStation(self, q_iiwa, gripper_setpoint, q_door_left,
                    q_door_right):
        if not self.is_visualizing:
            print("collision checker is not initialized with visualization.")
            return
        self.SetStationConfiguration(
            q_iiwa, gripper_setpoint, q_door_left, q_door_right)
        self.viz.DoPublish(self.context_meshcat, [])

    def ExistsCollision(self, q_iiwa, gripper_setpoint, q_door_left,
                        q_door_right):

        self.SetStationConfiguration(
            q_iiwa, gripper_setpoint, q_door_left, q_door_right)
        query_object = self.query_output_port.Eval(self.context_scene_graph)
        collision_paris = query_object.ComputePointPairPenetration()

        return len(collision_paris) > 0
