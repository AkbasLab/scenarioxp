import abc
from typing import Callable
from copy import copy

import pandas as pd
pd.set_option('mode.chained_assignment', None)

import numpy as np
import treelib
import rtree

import scipy.stats.qmc as qmc


def project(a : float, b : float, n : float, inc : float = None) -> float:
    """
    Project a normal val @n between @a and @b with an discretization 
    increment @inc.
    """
    assert n >= 0 and n <= 1
    assert b >= a

    # If no increment is provided, return the projection
    if inc is None:
        return n * (b - a) + a

    # Otherwise, round to the nearest increment
    n_inc = (b-a) / inc
    
    x = np.round(n_inc * n)
    return min(a + x*inc, b)
    

def normalize(u: np.ndarray):
    return u / np.linalg.norm(u)


def orthonormalize(
        u: np.ndarray, 
        v: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates orthonormal vectors given two vectors @u, @v which form a span.

    -- Parameters --
    u, v : np.ndarray
        Two n-d vectors of the same length
    -- Return --
    (un, vn)
        Orthonormal vectors for the span defined by @u, @v
    """
    u = u.squeeze()
    v = v.squeeze()

    assert len(u) == len(v)

    u = u[np.newaxis]
    v = v[np.newaxis]

    un = normalize(u)
    vn = v - np.dot(un, v.T) * un
    vn = normalize(vn)

    if not (np.dot(un, vn.T) < 1e-4):
        raise Exception("Vectors %s and %s are already orthogonal." \
            % (un, vn))

    return un, vn


def generateRotationMatrix(
        u: np.ndarray, 
        v: np.ndarray
    ) -> Callable[[float], np.ndarray]:
    """
    Creates a function that can construct a matrix that rotates by a given angle.

    Args:
        u, v : ndarray
            The two vectors that represent the span to rotate across.

    Raises:
        Exception: fails if @u and @v aren't vectors or if they have differing
            number of dimensions.

    Returns:
        Callable[[float], ndarray]: A function that returns a rotation matrix
            that rotates that number of degrees using the provided span.
    """
    u = u.squeeze()
    v = v.squeeze()

    if u.shape != v.shape:
        raise Exception("Dimension mismatch...")
    elif len(u.shape) != 1:
        raise Exception("Arguments u and v must be vectors...")

    u, v = orthonormalize(u, v)

    I = np.identity(len(u.T))

    coef_a = v * u.T - u * v.T
    coef_b = u * u.T + v * v.T

    return lambda theta: I + np.sin(theta) * \
        coef_a + (np.cos(theta) - 1) * coef_b


class ScenarioManager():
    def __init__(self, params : pd.DataFrame):
        """
        """
        req_indices = ["feat", "min", "max", "inc"]
        assert all([feat in req_indices for feat in params.columns])

        self._params = params

        # Determine the adjusted increment for each feature if min and max
        # were 0 and 1 respectively.
        b = params["max"]
        a = params["min"]
        i = params["inc"]
        params["inc_norm"] = i/(b-a)
        del b, a, i
        return

    @property
    def params(self) -> pd.DataFrame:
        return self._params

    def project(self, arr : np.ndarray) -> pd.Series:
        """
        Projects a normalized array @arr to selected concrete values from
        parameter ranges
        """
        # all values in arr must be in [0,1]
        assert all(arr >= 0) and all(arr <= 1)

        df = self.params.copy() \
            .assign(n = arr)
        
        projected = df.apply(
            lambda s: project(s["min"], s["max"], s["n"], s["inc"]), 
            axis=1
        )

        projected.index = self.params["feat"]
        return projected
        



class Scenario(abc.ABC):
    def __init__(self, params : pd.Series):
        """
        The abstract class for the scenario module.
        The scenario takes @params which are generated from a ScenarioManager.
        """
        assert isinstance(params, pd.Series)
        self._params = params
        return

    @property
    def params(self) -> pd.Series:
        """
        Input configuration for this scenario.
        """
        return self._params

    @abc.abstractproperty
    def score(self) -> pd.Series:
        """
        Scenario score.
        """
        raise NotImplementedError




class Explorer(abc.ABC):
    def __init__(self, 
        scenario_manager : ScenarioManager,
        scenario : Callable[[pd.Series], Scenario],
        target_score_classifier : Callable[[pd.Series], bool]
    ):
        assert isinstance(scenario_manager, ScenarioManager)
        assert isinstance(scenario, Callable)
        assert isinstance(target_score_classifier, Callable)

        self._scenario_manager = scenario_manager
        self._scenario = scenario
        self._target_score_classifier = target_score_classifier

        self._arr_history = []
        self._params_history = []
        self._score_history = []
        self._tsc_history = []
        return
    
    @property
    def arr_history(self) -> np.ndarray:
        return np.array(self._arr_history)

    @property
    def params_history(self) -> pd.DataFrame:
        return pd.DataFrame(self._params_history)

    @property
    def score_history(self) -> pd.DataFrame:
        return pd.DataFrame(self._score_history)

    @property
    def tsc_history(self) -> np.ndarray:
        return np.array(self._tsc_history)

    @abc.abstractmethod
    def next_arr(self) -> np.ndarray:
        """
        This gets the next va
        """
        raise NotImplementedError

    def step(self) -> bool:
        """
        Perform one exploration step.
        Returns if the scenario test is in the target score set.
        """
        arr = self.next_arr()                        # Long walk
        params = self._scenario_manager.project(arr) # Generate paramas
        test = self._scenario(params)                # Run scenario
        is_target_score = self._target_score_classifier(test.score)

        self._arr_history.append(arr)
        self._params_history.append(params)
        self._score_history.append(test.score)
        self._tsc_history.append(is_target_score)
        return is_target_score





class SequenceExplorer(Explorer):
    MONTE_CARLO = "random"
    HALTON = "halton"
    SOBOL = "sobol"

    def __init__(self, 
        strategy : str,
        seed : int,
        scenario_manager : ScenarioManager,
        scenario : Callable[[pd.Series], Scenario],
        target_score_classifier : Callable[[pd.Series], bool],
        scramble : bool = True,
        fast_foward : int = 0
    ):
        """
        The simplest explorer is the SequenceExplore, which samples the next
        parameter for a test using a quasi-random sequence.

        -- Params --
        strategy : str
            The sampling strategy. "random", "halton", and "sobol" strategies
            are supported. Random uses the numpy generator, while halton and
            sobol use scipy.stats.qmc generators.
        seed : int
            Seed for the rng which scrambls the sequence if @scramble flag is
            used.
        scramble : bool (default=True)
            Scramble the sequence
        fast_forward : int (default=0)
            The sequence will @fast_foward n iterations during initialization.
        """
        super().__init__(scenario_manager, scenario, target_score_classifier)
        assert strategy in ["random", "halton", "sobol"]

        d = len(scenario_manager.params.index)

        if strategy == self.MONTE_CARLO:
            seq = np.random.RandomState(seed = seed)
            if fast_foward:
                seq.random(size=d)
        elif strategy == self.HALTON:
            seq = qmc.Halton(d=d, scramble=scramble)
            if fast_foward:
                seq.fast_forward(fast_foward)
        elif strategy == self.SOBOL:
            seq = qmc.Sobol(d=d, scramble=scramble)
            if fast_foward:
                seq.fast_forward(fast_foward)
        else:
            raise NotImplementedError

        self._d = d
        self._seq = seq
        self._strategy = strategy
        return

    def next_arr(self) -> np.ndarray:
        if self._strategy == self.MONTE_CARLO:
            return self._seq.random(size = self._d)
        elif self._strategy in [self.SOBOL, self.HALTON]:
            return self._seq.random(1)[0]
        raise NotImplementedError


class FindSurfaceExplorer(Explorer):
    def __init__(self, 
        root : np.ndarray,
        seed : int,
        scenario_manager : ScenarioManager,
        scenario : Callable[[pd.Series], Scenario],
        target_score_classifier : Callable[[pd.Series], bool]
    ):
        """
        Navigates from @root someplace in an target envelope to the surface.

        -- Additional parameters--
        @seed : int
            Seed for the RNG.
        """
        super().__init__(scenario_manager, scenario, target_score_classifier)

        self.root = root

        # Jump distance
        self._d = scenario_manager.params["inc_norm"]
        
        # Find the surface
        rng = np.random.RandomState(seed=seed)
        self._v = rng.rand(len(root))
        self._s = self._v * self._d
        self._interm = [root]

        self._prev = None
        self._cur = root

        self._stage = 0
        return

    def step(self):
        # Reach within d distance from the surface
        if self._stage == 0:
            self._prev = self._cur
            self._interm += [self._prev]
            self._cur = self._round_to_limits(
                self._prev + self._s,
                np.zeros(len(self.root)),
                np.ones(len(self.root))
            )

            # Stage end condition
            if all(self._cur == self._prev):
                print("0: At parameter boundary.")
                self._stage = 1
            elif not super().step():
                print("0: Past parameter boundary.")
                self._stage = 1
            
            return False
            
        # Transition to d/2
        elif self._stage == 1:
            print("1: Within d distance from surface." )
            self._s *= 0.5
            self._cur = self._round_to_limits(
                self._prev + self._s,
                np.zeros(len(self.root)),
                np.ones(len(self.root))
            )

            if all(self._cur == self._prev):
                print("2: At parameter boundary. Done.")
                return True
            elif not super().step():
                print("2: Past parameter boundary. Done")
                return True

            self._stage = 2
            return False
            
        # Get closer until within d/2 distance from surface
        elif self._stage == 2:
            self._prev = self._cur
            self._interm += [self._prev]
            self._cur = self._round_to_limits(
                self._prev + self._s,
                np.zeros(len(self.root)),
                np.ones(len(self.root))
            )

            if all(self._cur == self._prev):
                print("2: At parameter boundary. Done.")
                return True
            elif not super().step():
                print("2: Past parameter boundary. Done")
                return True
        raise NotImplemented

    def next_arr(self) -> np.ndarray:
        return self._cur

    def _round_to_limits(
        self,
        arr : np.ndarray, 
        min : np.ndarray, 
        max : np.ndarray
    ) -> np.ndarray:
        """
        Rounds each dimensions in @arr to limits within @min limits and @max limits.
        """
        is_lower = arr < min
        is_higher = arr > max
        for i in range(len(arr)):
            if is_lower[i]:
                arr[i] = min[i]
            elif is_higher[i]:
                arr[i] = max[i]
        return arr





class BoundaryLostException(Exception):
    "When a boundary Adherer fails to find the boundary, this exception is thrown"

    def __init__(self, msg="Failed to locate boundary!"):
        self.msg = msg
        super().__init__(msg)

    def __str__(self):
        return f"<BoundaryLostException: Angle: {self.theta}, Jump Distance: {self.d}>"





class BoundaryAdhererExplorer(Explorer):
    def __init__(self, 
        root : np.ndarray,
        n : np.ndarray,
        direction : np.ndarray,
        d: np.ndarray,
        theta: float,
        scenario_manager : ScenarioManager,
        scenario : Callable[[pd.Series], Scenario],
        target_score_classifier : Callable[[pd.Series], bool]
    ):
        """
        A module which navigates from one classification to another.

        Boundary error, e, is within the range: 0 <= e <= d * theta. 
        Average error is d * theta / 2

        Args:
            classifier (Callable[[Point], bool]): The function that returns 
                true or false depending on whether or not the provided Point 
                lies within or outside of the target envelope.
            p (Point): Parent boundary point - used as a starting point for 
                finding the neighboring boundary point.
            n (ndarray): The parent boundary point's estimated orthogonal 
                surface vector.
            direction (ndarray): The general direction to travel in 
                (MUST NOT BE PARALLEL WITH @n)
            d (np.ndarray): How far to travel from @p
            theta (float): How far to rotate to find the boundary.
        """
        super().__init__(scenario_manager, scenario, target_score_classifier)

        self.root = root
        self._theta = theta
        
        n = normalize(n)
        
        self._rotater_function = generateRotationMatrix(n, direction)
        self._s = (copy(n.squeeze()) * d).squeeze()

        self.ANGLE_90 = np.pi / 2

        A = self.ANGLE_90
        self._s = np.dot(A, self._s)

        self._prev = None
        self._prev_class = None

        self._cur = self.root + self._s

        self.STAGE_FIRST_ROTATION = 0
        self.STAGE_NEXT_SAMPLE = 1
        self._stage = self.STAGE_FIRST_ROTATION
        return
    
    @property
    def b(self) -> np.ndarray:
        """The identified boundary point"""
        return self._b

    @property
    def n(self) -> np.ndarray:
        """The identified boundary point's estimated orthogonal surface vector"""
        return self._n

    @property
    def boundary(self) -> tuple[np.ndarray, np.ndarray]:
        return (self._b, self._n)

    @property
    def sub_samples(self):
        return self._sub_samples


    def next_arr(self) -> np.ndarray:
        return self._cur

    def step(self) -> bool:
        if self._stage == self.STAGE_FIRST_ROTATION:
            # print("0: First stage rotation.")
            self._cur_class = super().step()
            if self._cur_class:
                self._rotate = self._rotater_function(self._theta)
            else:
                self._rotate = self._rotater_function(-self._theta)

            self._b = None
            self._n = None

            self._sub_samples = []

            self._iteration = 0
            self._max_iteration = (2 * np.pi) // self._theta

            self._stage = self.STAGE_NEXT_SAMPLE
            return False

        elif self.STAGE_NEXT_SAMPLE:
            self._prev = self._cur
            self._s = np.dot(self._rotate, self._s)
            self._cur = self.root + self._s

            self._prev_class = self._cur_class
            self._cur_class = super().step()

            # Went over the boarder
            if self._cur_class != self._prev_class:
                self._b = self._cur if self._cur_class else self._prev
                self._n = normalize(
                    np.dot(self._rotater_function(self.ANGLE_90), self._s)
                )
                if not (self._b is None):
                    # print("1: Boundary point located after"\
                    # + " %d iterations." % self._iteration \
                    # + " Done.")
                    return True

            elif self._iteration > self._max_iteration:
                raise BoundaryLostException()

            self._sub_samples.append(self._cur)
            self._iteration += 1
            return False
        raise NotImplementedError



class BoundaryAdherenceExplorerGenerator:
    def __init__(self, d : np.ndarray, theta : float):
        """
        A generator to generate BoundaryAdhereExplorer objects.
        This is neccesary to tune parameters for the boundary explorer 
        strategies.

        Args:
            d (np.ndarray): How far to travel from @p
            theta (float): How far to rotate to find the boundary.
        """
        self._d = d
        self._theta = theta
        return

    def generate(self, 
        root : np.ndarray,
        n : np.ndarray,
        direction : np.ndarray,
        scenario_manager : ScenarioManager,
        scenario : Callable[[pd.Series], Scenario],
        target_score_classifier : Callable[[pd.Series], bool]
    ) -> BoundaryAdhererExplorer:
        return BoundaryAdhererExplorer(
            root, n, direction, self._d, self._theta, 
            scenario_manager, scenario, target_score_classifier
        )

class RRTBoundaryExplorer(Explorer):
    DATA_LOCATION = "location"
    DATA_NORMAL = "normal"

    def __init__(self,
        root : np.ndarray,
        n0 : np.ndarray,
        bae_generator : BoundaryAdherenceExplorerGenerator,
        seed : int,
        scenario_manager : ScenarioManager,
        scenario : Callable[[pd.Series], Scenario],
        target_score_classifier : Callable[[pd.Series], bool]
    ):
        super().__init__(scenario_manager, scenario, target_score_classifier)

        self._bae_generator = bae_generator
        self._rng = np.random.RandomState(seed = seed)

        # From JRRT Abstract class
        self._boundary = [root]
        self._all_points = []

        self._prev = (root, n0)


        # JRRT
        self._ndims = len(root)
        self._tree = treelib.Tree()
        self._root = treelib.Node(identifier=0, 
            data=self._create_data(*self._prev))
        self._next_id = 1

        p = rtree.index.Property()
        p.set_dimension(self._ndims)
        self._index = rtree.index.Index(properties=p)

        self._index.insert(0, root)
        self._tree.add_node(self._root)

        self._prev_dir: np.ndarray = None

        
        # First boundary adherence step
        b, n = self._select_parent()
        direction = self._pick_direction()
        self._bae = self._bae_generator.generate(
            b, n, direction, self._scenario_manager,
            self._scenario, self._target_score_classifier
        )

        self._bae_history = [self._bae]
        
        # List to store bounday classification
        self._is_boundary = []
        
        return
    
    def step(self) -> bool:
        # Run a BAE module step
        boundary_point_located = self._bae.step()

        # Copy the history of the last point
        self._arr_history.append(self._bae._arr_history[-1])
        self._params_history.append(self._bae._params_history[-1])
        self._score_history.append(self._bae._score_history[-1])
        self._tsc_history.append(self._bae._tsc_history[-1])
        
        # Store boundary information so it may be classified later.
        self._is_boundary.append(boundary_point_located)

        if boundary_point_located:
            # Store boudary point info in the tree
            self._add_child(*self._bae.boundary)
            self._prev = self._bae.boundary

            # Create another BAE
            b, n = self._select_parent()
            direction = self._pick_direction()
            self._bae = self._bae_generator.generate(
                b, n, direction, self._scenario_manager,
                self._scenario, self._target_score_classifier
            )
            self._bae_history.append(self._bae)
            return False

        # TODO: No automated exit condition for BRRT is known.
        return False

    @property
    def bae_history(self) -> list[BoundaryAdhererExplorer]:
        """
        All BoundaryAdhererExplorer modules used in this RRT.
        """
        return self._bae_history

    @property
    def is_boundary(self) -> list[bool]:
        """
        Boundary point classification for every 
        """
        return self._is_boundary
    
        
    @property
    def prev(self):
        return self._prev

    @property
    def sub_samples(self) -> list[np.ndarray]:
        "All samples taken, including the boundary points."
        return self._all_points

    @property
    def boundary(self):
        return self._boundary

    @property
    def previous_node(self) -> treelib.Node:
        return self._tree.get_node(self._next_id - 1)

    @property
    def previous_direction(self) -> np.ndarray:
        return self._prev_dir

    def _select_parent(self) -> tuple[np.ndarray, np.ndarray]:
        "Select which boundary point to explore from next."
        self._r = self._random_point()
        self._parent = self._find_nearest(self._r)
        self._p = self._parent.data[self.DATA_LOCATION]
        return (self._parent.data[self.DATA_LOCATION], 
                    self._parent.data[self.DATA_NORMAL])

    def _pick_direction(self) -> np.ndarray:
        "Select a direction to explore towards."
        return self._r - self._p

    def _add_child(self, bk: np.ndarray, nk: np.ndarray):
        "Add a newly found boundary point and its surface vector."
        self._add_node(bk, nk, self._parent.identifier)

    def _add_node(self, p: np.ndarray, n: np.ndarray, parentID: int):
        """
        Add a node to the tree.
        """
        node = treelib.Node(identifier=self._next_id, data=self._create_data(p, n))
        self._tree.add_node(node, parentID)
        self._index.insert(self._next_id, p)
        self._next_id += 1

    def _random_point(self) -> np.ndarray:
        """
        Generate a random point for the tree.
        """
        return self._rng.rand(self._ndims) * 2

    def _find_nearest(self, p: np.ndarray) -> treelib.Node:
        node = self._tree.get_node(next(self._index.nearest(p)))
        return node

    @staticmethod
    def _create_data(location, normal) -> dict:
        return {
            RRTBoundaryExplorer.DATA_LOCATION: location, 
            RRTBoundaryExplorer.DATA_NORMAL: normal
        }

    def next_arr(self):
        pass

    

    def expand(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Take one step along the boundary; i.e., Finds a new
        boundary point.

        Returns:
            tuple[Point, ndarray]: The newly added point on the surface
                and its estimated orthonormal surface vector

        Throws:
            BoundaryLostException: Thrown if the boundary is lost
        """
        b, n = self._select_parent()
        direction = self._pick_direction()
        adherer = self._adhererF.adhere_from(b, n, direction)
        for b in adherer:
            self._all_points.append(b)

        self._add_child(*adherer.boundary)
        self._prev = adherer.boundary

        return adherer.boundary