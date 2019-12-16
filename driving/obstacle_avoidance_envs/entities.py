import math
from typing import Text, Union
import numpy as np
from obstacle_avoidance_envs.geometry import Point, Rectangle, Circle
import copy


#def get_entity_dynamics(friction, min_speed, max_speed, min_acc, max_acc, xnp=np):
def get_entity_dynamics(goal, xnp=np):
    # xnp: Either numpy or jax.numpy.
    # TODO: check if distance to is the min distance function
    # TODO: add repulsive dynamics for world boundaries
    # TODO: check if agent.cetner is updated

    def attractive_pf(agent, K):
        return -K * (np.array([agent.center.x, agent.center.y]) - goal)

    def repulsive_pf(agent, dynamic_agents, static_agents, K, r):
        repulsive = np.array([0.0, 0.0])
        for da in dynamic_agents:
            if da != agent:
                if da == goal: continue
                else:
                    min_dist = agent.distanceTo(da)
                    min_dist = max([r / 100, min_dist])
                    if min_dist <= r:
                        norm_diff = (agent.center - da.center) / min_dist
                        #repulse = ((1 / min_dist) - (1.0 / r)) * (1.0 / min_dist ** 2) * norm_diff * 50000
                        repulse = ((1 / min_dist) - (1.0 / r)) * (1.0 / min_dist ** 2) * norm_diff
                        repulsive += repulse
        for sa in static_agents:
            if sa == goal: continue
            else:
                min_dist = float(agent.distanceTo(sa))
                min_dist = max([r / 100, min_dist])
                if min_dist <= r:
                    norm_diff = (agent.center - sa.center) / min_dist
                    #repulse = ((1 / min_dist) - (1.0 / r)) * (1.0 / min_dist ** 2) * norm_diff * 80000
                    repulse = ((1 / min_dist) - (1.0 / r)) * (1.0 / min_dist ** 2) * norm_diff
                    #repulsive += repulse * 1e3
                    repulsive += repulse
        return repulsive


    def entity_dynamics(agent, dynamic_agents, static_agents):
        # x: (x, y, xp, yp, theta, ang_vel, acceleration)
        # u: (steering angle, acceleration)
        #center = x[:2]
        #velocity = x[2:4]
        #speed = xnp.linalg.norm(velocity, ord=2)
        #heading = x[4]
        #angular_velocity = x[5]
        #old_acceleration = x[6]
        #steering_angle = u[0]
        #acceleration = xnp.clip(u[1], min_acc, max_acc)

        #new_angular_velocity = speed * steering_angle
        #new_acceleration = acceleration - friction * speed

        #new_heading = heading + (angular_velocity + new_angular_velocity) * dt / 2.0
        #new_speed = xnp.clip(
        #    speed + (old_acceleration + new_acceleration) * dt / 2.0, min_speed, max_speed
        #)

        #next_speed = (speed + new_speed) / 2.0
        #next_heading = (new_heading + heading) / 2.0
        #new_velocity = next_speed * xnp.array((xnp.cos(next_heading), xnp.sin(next_heading)))

        #new_center = center + (velocity + new_velocity) * dt / 2.0
        #return xnp.concatenate(
        #    (
        #        new_center,
        #        new_velocity,
        #        xnp.stack([new_heading, new_angular_velocity, new_acceleration]),
        #    )
        #)

        print("GOAL: ", goal)
        # calculate attractive pf
        attractive = attractive_pf(agent, K=1)
        print("attractive: ", attractive)

        # calculate repulsive pf
        repulsive = repulsive_pf(agent, dynamic_agents, static_agents, K=1, r=100)
        print("repulsive: ", repulsive)
        return attractive + repulsive

    return entity_dynamics


class Entity:
    def __init__(
        self,
        center: Point,
        heading: float,
        movable: bool = True,
        friction: float = 0.0,
        min_speed: float = 0.0,
        max_speed: float = math.inf,
        min_acc: float = -math.inf,
        max_acc: float = math.inf,
    ):
        self.center = center  # this is x, y
        self.heading = heading
        self.movable = movable
        self.color = "ghost white"
        self.collidable = True
        self.obj = None  # MUST be set by subclasses.
        if movable:
            self.friction = friction
            self.velocity = Point(0, 0)  # this is xp, yp
            self.acceleration = 0  # this is vp (or speedp)
            self.angular_velocity = 0  # this is headingp
            self.inputSteering = 0
            self.inputAcceleration = 0
            self.min_speed = min_speed
            self.max_speed = max_speed
            self.min_acc = min_acc
            self.max_acc = max_acc
            self.entity_dynamics = get_entity_dynamics(np.array([120,120]), xnp=np
            )

    @property
    def speed(self) -> float:
        return self.velocity.norm(p=2) if self.movable else 0

    def set_control(self, inputSteering: float, inputAcceleration: float):
        self.inputSteering = inputSteering
        self.inputAcceleration = inputAcceleration

    @property
    def state(self):
        return np.array(
            (
                self.x,
                self.y,
                #self.xp,
                #self.yp,
                #self.heading,
                #self.angular_velocity,
                #self.acceleration,
            )
        )

    @state.setter
    def state(self, new_x):
        self.center = Point(new_x[0], new_x[1])
        self.velocity = Point(new_x[2], new_x[3])
        self.heading = new_x[4]
        self.angular_velocity = new_x[5]
        self.acceleration = new_x[6]
        self.buildGeometry()

    def tick(self, agent, dynamic_agents, static_agents, dt):
        if self.movable:
            print("TICKING!!")
            print(agent.center.x, agent.center.y)
            print(self.center.x, self.center.y)
            dx = self.entity_dynamics(agent, dynamic_agents, static_agents)
            dx *= dt
            self.center.x += dx[0]
            self.center.y += dx[1]

    def buildGeometry(self):  # builds the obj
        raise NotImplementedError

    def collidesWith(self, other: Union["Point", "Entity"]) -> bool:
        if isinstance(other, Entity):
            return self.obj.intersectsWith(other.obj)
        elif isinstance(other, Point):
            return self.obj.intersectsWith(other)
        else:
            raise NotImplementedError

    def distanceTo(self, other: Union["Point", "Entity"]) -> float:
        if isinstance(other, Entity):
            return self.obj.distanceTo(other.obj)
        elif isinstance(other, Point):
            return self.obj.distanceTo(other)
        else:
            raise NotImplementedError

    def copy(self):
        return copy.deepcopy(self)

    @property
    def x(self):
        return self.center.x

    @property
    def y(self):
        return self.center.y

    @property
    def xp(self):
        return self.velocity.x

    @property
    def yp(self):
        return self.velocity.y


class RectangleEntity(Entity):
    def __init__(
        self,
        center: Point,
        heading: float,
        size: Point,
        movable: bool = True,
        friction: float = 0,
        **kwargs
    ):
        super(RectangleEntity, self).__init__(center, heading, movable, friction, **kwargs)
        self.size = size
        self.buildGeometry()

    @property
    def edge_centers(self):
        edge_centers = np.zeros((4, 2), dtype=np.float32)
        x = self.center.x
        y = self.center.y
        w = self.size.x
        h = self.size.y
        edge_centers[0] = [
            x + w / 2.0 * np.cos(self.heading),
            y + w / 2.0 * np.sin(self.heading),
        ]
        edge_centers[1] = [
            x - h / 2.0 * np.sin(self.heading),
            y + h / 2.0 * np.cos(self.heading),
        ]
        edge_centers[2] = [
            x - w / 2.0 * np.cos(self.heading),
            y - w / 2.0 * np.sin(self.heading),
        ]
        edge_centers[3] = [
            x + h / 2.0 * np.sin(self.heading),
            y - h / 2.0 * np.cos(self.heading),
        ]
        return edge_centers

    @property
    def corners(self):
        ec = self.edge_centers
        c = np.array([self.center.x, self.center.y])
        corners = []
        corners.append(Point(*(ec[1] + ec[0] - c)))
        corners.append(Point(*(ec[2] + ec[1] - c)))
        corners.append(Point(*(ec[3] + ec[2] - c)))
        corners.append(Point(*(ec[0] + ec[3] - c)))
        return corners

    def buildGeometry(self):
        C = self.corners
        self.obj = Rectangle(*C[:-1])  # pylint: disable=no-value-for-parameter


class CircleEntity(Entity):
    def __init__(
        self,
        center: Point,
        heading: float,
        radius: float,
        movable: bool = True,
        friction: float = 0,
        **kwargs
    ):
        super(CircleEntity, self).__init__(center, heading, movable, friction, **kwargs)
        self.radius = radius
        self.buildGeometry()

    def buildGeometry(self):
        self.obj = Circle(self.center, self.radius)


class TextEntity(Entity):
    def __init__(self, center: Point, **kwargs):
        heading = 0
        super(TextEntity, self).__init__(center, heading, movable=False, **kwargs)
        self.text = ""

    def buildGeometry(self):
        # Represent text geometry as a tiny circle. Not accurate.
        self.obj = Circle(self.center, 0.01)
