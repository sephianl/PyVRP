import pickle

import numpy as np
import pytest
from numpy.testing import assert_, assert_equal, assert_raises

from pyvrp import (
    Client,
    Depot,
    ProblemData,
    Route,
    SameVehicleGroup,
    Solution,
    VehicleType,
)
from pyvrp.Model import Model


def test_same_vehicle_group_constructor():
    """
    Tests that SameVehicleGroup correctly initializes its fields.
    """
    group = SameVehicleGroup([1, 2, 3], name="test")

    assert_equal(list(group), [1, 2, 3])
    assert_equal(group.clients, [1, 2, 3])
    assert_equal(len(group), 3)
    assert_equal(group.name, "test")
    assert_equal(str(group), "test")


def test_same_vehicle_group_empty():
    """
    Tests that an empty SameVehicleGroup can be created.
    """
    group = SameVehicleGroup()
    assert_equal(len(group), 0)
    assert_equal(list(group), [])


def test_same_vehicle_group_add_client():
    """
    Tests that clients can be added to a SameVehicleGroup.
    """
    group = SameVehicleGroup()
    group.add_client(1)
    group.add_client(2)

    assert_equal(list(group), [1, 2])
    assert_equal(len(group), 2)


def test_same_vehicle_group_raises_for_duplicate_client():
    """
    Tests that adding a duplicate client raises an error.
    """
    group = SameVehicleGroup([1, 2])

    with assert_raises(Exception):
        group.add_client(1)

    # Also test via constructor
    with assert_raises(Exception):
        SameVehicleGroup([1, 1, 2])


def test_same_vehicle_group_clear():
    """
    Tests that the clear method empties the group.
    """
    group = SameVehicleGroup([1, 2, 3])
    assert_equal(len(group), 3)

    group.clear()
    assert_equal(len(group), 0)


def test_same_vehicle_group_equality():
    """
    Tests equality comparison between SameVehicleGroups.
    """
    group1 = SameVehicleGroup([1, 2], name="test")
    group2 = SameVehicleGroup([1, 2], name="test")
    group3 = SameVehicleGroup([1, 3], name="test")
    group4 = SameVehicleGroup([1, 2], name="other")

    assert_(group1 == group2)
    assert_(not (group1 == group3))
    assert_(not (group1 == group4))


def test_same_vehicle_group_pickle():
    """
    Tests that SameVehicleGroup can be pickled and unpickled.
    """
    group = SameVehicleGroup([1, 2, 3], name="pickled")
    pickled = pickle.dumps(group)
    unpickled = pickle.loads(pickled)

    assert_(group == unpickled)


def test_problem_data_with_same_vehicle_groups():
    """
    Tests that ProblemData correctly stores and returns same-vehicle groups.
    """
    clients = [Client(x=i, y=i) for i in range(1, 4)]
    depot = Depot(x=0, y=0)
    vehicle_type = VehicleType(num_available=2)
    distances = np.array([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]])
    durations = np.zeros_like(distances)

    # Create a same-vehicle group for clients 1 and 2 (indices 1 and 2)
    same_veh_group = SameVehicleGroup([1, 2], name="group1")

    data = ProblemData(
        clients=clients,
        depots=[depot],
        vehicle_types=[vehicle_type],
        distance_matrices=[distances],
        duration_matrices=[durations],
        groups=[],
        same_vehicle_groups=[same_veh_group],
    )

    assert_equal(data.num_same_vehicle_groups, 1)
    assert_equal(list(data.same_vehicle_group(0)), [1, 2])
    assert_equal(data.same_vehicle_group(0).name, "group1")


def test_problem_data_raises_for_invalid_same_vehicle_group():
    """
    Tests that ProblemData raises when a same-vehicle group references
    invalid client indices.
    """
    clients = [Client(x=1, y=1)]
    depot = Depot(x=0, y=0)
    vehicle_type = VehicleType()
    distances = np.array([[0, 1], [1, 0]])
    durations = np.zeros_like(distances)

    # Client index 5 doesn't exist (only have index 1 = first client)
    invalid_group = SameVehicleGroup([1, 5])

    with assert_raises(Exception):
        ProblemData(
            clients=clients,
            depots=[depot],
            vehicle_types=[vehicle_type],
            distance_matrices=[distances],
            duration_matrices=[durations],
            groups=[],
            same_vehicle_groups=[invalid_group],
        )


def test_problem_data_raises_for_empty_same_vehicle_group():
    """
    Tests that ProblemData raises when an empty same-vehicle group is passed.
    """
    depot = Depot(x=0, y=0)
    vehicle_type = VehicleType()
    distances = np.array([[0]])
    durations = np.zeros_like(distances)

    empty_group = SameVehicleGroup()

    with assert_raises(Exception):
        ProblemData(
            clients=[],
            depots=[depot],
            vehicle_types=[vehicle_type],
            distance_matrices=[distances],
            duration_matrices=[durations],
            groups=[],
            same_vehicle_groups=[empty_group],
        )


def test_solution_feasibility_with_same_vehicle_constraint():
    """
    Tests that a solution is marked infeasible when clients from a same-vehicle
    group are on different routes.
    """
    clients = [Client(x=i, y=i) for i in range(1, 4)]
    depot = Depot(x=0, y=0)
    vehicle_type = VehicleType(num_available=2)
    distances = np.array([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]])
    durations = np.zeros_like(distances)

    # Clients 1 and 2 must be on the same vehicle
    same_veh_group = SameVehicleGroup([1, 2])

    data = ProblemData(
        clients=clients,
        depots=[depot],
        vehicle_types=[vehicle_type],
        distance_matrices=[distances],
        duration_matrices=[durations],
        groups=[],
        same_vehicle_groups=[same_veh_group],
    )

    # Solution where clients 1 and 2 are on the same route - should be feasible
    feasible_sol = Solution(data, [[1, 2], [3]])
    assert_(feasible_sol.is_group_feasible())
    assert_(feasible_sol.is_feasible())

    # Solution where clients 1 and 2 are on different routes - should be infeasible
    infeasible_sol = Solution(data, [[1, 3], [2]])
    assert_(not infeasible_sol.is_group_feasible())
    assert_(not infeasible_sol.is_feasible())


def test_solution_feasibility_partial_group_visit():
    """
    Tests that visiting only some clients from a same-vehicle group is allowed,
    as long as the visited ones are on the same route.
    """
    clients = [
        Client(x=1, y=1, required=False),
        Client(x=2, y=2, required=False),
        Client(x=3, y=3, required=False),
    ]
    depot = Depot(x=0, y=0)
    vehicle_type = VehicleType(num_available=1)
    distances = np.array([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]])
    durations = np.zeros_like(distances)

    # All three clients must be on the same vehicle if visited
    same_veh_group = SameVehicleGroup([1, 2, 3])

    data = ProblemData(
        clients=clients,
        depots=[depot],
        vehicle_types=[vehicle_type],
        distance_matrices=[distances],
        duration_matrices=[durations],
        groups=[],
        same_vehicle_groups=[same_veh_group],
    )

    # Only visit clients 1 and 2 on the same route - should be feasible
    sol = Solution(data, [[1, 2]])
    assert_(sol.is_group_feasible())

    # Visit only client 1 - should be feasible
    sol_single = Solution(data, [[1]])
    assert_(sol_single.is_group_feasible())

    # Visit no clients - should be feasible (empty solution)
    sol_empty = Solution(data, [])
    assert_(sol_empty.is_group_feasible())


def test_model_add_same_vehicle_group():
    """
    Tests adding same-vehicle groups through the Model interface.
    """
    model = Model()
    depot = model.add_depot(x=0, y=0)
    client1 = model.add_client(x=1, y=1)
    client2 = model.add_client(x=2, y=2)
    client3 = model.add_client(x=3, y=3)
    model.add_vehicle_type(num_available=2)

    # Add edges
    for frm in model.locations:
        for to in model.locations:
            dist = 0 if frm is to else 1
            model.add_edge(frm, to, distance=dist)

    # Add same-vehicle group
    group = model.add_same_vehicle_group(client1, client2, name="test_group")

    assert_equal(len(model.same_vehicle_groups), 1)
    assert_(group in model.same_vehicle_groups)
    assert_equal(group.name, "test_group")

    # Verify the constraint is propagated to ProblemData
    data = model.data()
    assert_equal(data.num_same_vehicle_groups, 1)


def test_model_same_vehicle_group_client_not_in_model():
    """
    Tests that adding a client not in the model raises an error.
    """
    model = Model()
    model.add_depot(x=0, y=0)
    client1 = model.add_client(x=1, y=1)

    # Create a client that's not added to the model
    other_client = Client(x=5, y=5)

    with assert_raises(ValueError):
        model.add_same_vehicle_group(client1, other_client)


def test_model_depot_addition_updates_same_vehicle_groups():
    """
    Tests that adding a depot after same-vehicle groups correctly updates
    the client indices in those groups.
    """
    model = Model()
    depot1 = model.add_depot(x=0, y=0)
    client1 = model.add_client(x=1, y=1)
    client2 = model.add_client(x=2, y=2)
    model.add_vehicle_type(num_available=1)

    # Add same-vehicle group
    model.add_same_vehicle_group(client1, client2)

    # Add another depot - this shifts client indices
    depot2 = model.add_depot(x=5, y=5)

    # Add edges for the new depot
    for frm in model.locations:
        for to in model.locations:
            dist = 0 if frm is to else 1
            model.add_edge(frm, to, distance=dist)

    # The constraint should still work correctly
    data = model.data()
    assert_equal(data.num_same_vehicle_groups, 1)

    # Client indices should now be 2 and 3 (after 2 depots)
    group_clients = list(data.same_vehicle_group(0))
    assert_equal(group_clients, [2, 3])
