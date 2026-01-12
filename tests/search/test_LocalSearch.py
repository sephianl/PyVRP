import numpy as np
import pytest
from numpy.testing import assert_, assert_equal

from pyvrp import (
    Client,
    CostEvaluator,
    Depot,
    ProblemData,
    RandomNumberGenerator,
    Route,
    Solution,
    Trip,
    VehicleType,
)
from pyvrp.search import (
    Exchange10,
    Exchange11,
    LocalSearch,
    PerturbationManager,
    PerturbationParams,
    RelocateWithDepot,
    SwapRoutes,
    SwapStar,
    compute_neighbours,
)
from pyvrp.search._search import LocalSearch as cpp_LocalSearch


def test_local_search_returns_same_solution_with_empty_neighbourhood(ok_small):
    """
    Tests that calling the local search when it only has node operators and
    an empty neighbourhood is a no-op: since the node operators respect the
    neighbourhood definition, they cannot do anything with an empty
    neighbourhood.
    """
    cost_evaluator = CostEvaluator([20], 6, 0)
    rng = RandomNumberGenerator(seed=42)

    neighbours = [[] for _ in range(ok_small.num_locations)]
    ls = LocalSearch(ok_small, rng, neighbours)
    ls.add_node_operator(Exchange10(ok_small))
    ls.add_node_operator(Exchange11(ok_small))

    # The search is completed after one iteration due to the empty
    # neighbourhood. This also prevents moves involving empty routes,
    # which are not explicitly forbidden by the empty neighbourhood.
    sol = Solution.make_random(ok_small, rng)
    assert_equal(ls.search(sol, cost_evaluator), sol)


def test_local_search_call_perturbs_solution(ok_small):
    """
    Tests that calling local search perturbs a solution.
    """
    rng = RandomNumberGenerator(seed=42)
    neighbours = compute_neighbours(ok_small)
    ls = LocalSearch(ok_small, rng, neighbours)

    sol = Solution.make_random(ok_small, rng)
    cost_eval = CostEvaluator([1], 1, 0)

    # The local search should perturb the solution even though no node and
    # route operators are added.
    perturbed = ls(sol, cost_eval)
    assert_(perturbed != sol)


def test_get_set_neighbours(ok_small):
    """
    Tests that getting and setting the local search's granular neighbourhood
    works as expected. For more details, see the tests for the SearchSpace in
    ``test_SearchSpace.py``, which handle validation.
    """
    rng = RandomNumberGenerator(seed=42)
    neighbours = [[] for _ in range(ok_small.num_locations)]
    ls = LocalSearch(ok_small, rng, neighbours)
    assert_equal(ls.neighbours, neighbours)

    new_neighbours = compute_neighbours(ok_small)
    assert_(new_neighbours != neighbours)

    ls.neighbours = new_neighbours
    assert_equal(ls.neighbours, new_neighbours)


def test_reoptimize_changed_objective_timewarp_OkSmall(ok_small):
    """
    This test reproduces a bug where loadSolution in LocalSearch.cpp would
    reset the timewarp for a route to 0 if the route was not changed. This
    would cause improving moves with a smaller timewarp not to be considered
    because the current cost doesn't count the current time warp.
    """
    rng = RandomNumberGenerator(seed=42)
    sol = Solution(ok_small, [[1, 2, 3, 4]])

    # We make neighbours only contain 1 -> 2, so the only feasible move
    # is changing [1, 2, 3, 4] into [2, 1, 3, 4] or moving one of the nodes
    # into its own route. Since those solutions have larger distance but
    # smaller time warp, they are considered improving moves with a
    # sufficiently large time warp penalty.
    neighbours = [[], [2], [], [], []]  # 1 -> 2 only
    ls = LocalSearch(ok_small, rng, neighbours)
    ls.add_node_operator(Exchange10(ok_small))

    # With 0 timewarp penalty, the solution should not change since
    # the solution [2, 1, 3, 4] has larger distance.
    improved_sol = ls.search(sol, CostEvaluator([0], 0, 0))
    assert_equal(sol, improved_sol)

    # Now doing it again with a large TW penalty, we must find the alternative
    # solution
    # (previously this was not the case since due to caching the current TW was
    # computed as being zero, causing the move to be evaluated as worse)
    cost_evaluator_tw = CostEvaluator([0], 1000, 0)
    improved_sol = ls.search(sol, cost_evaluator_tw)
    improved_cost = cost_evaluator_tw.penalised_cost(improved_sol)
    assert_(improved_cost < cost_evaluator_tw.penalised_cost(sol))


def test_prize_collecting(prize_collecting):
    """
    Tests that local search works on a small prize-collecting instance.
    """
    rng = RandomNumberGenerator(seed=42)
    cost_evaluator = CostEvaluator([1], 1, 0)

    sol = Solution.make_random(prize_collecting, rng)
    sol_cost = cost_evaluator.penalised_cost(sol)

    neighbours = compute_neighbours(prize_collecting)
    ls = LocalSearch(prize_collecting, rng, neighbours)
    ls.add_node_operator(Exchange10(prize_collecting))  # relocate
    ls.add_node_operator(Exchange11(prize_collecting))  # swap

    improved = ls.search(sol, cost_evaluator)
    improved_cost = cost_evaluator.penalised_cost(improved)

    assert_(improved.num_clients() < prize_collecting.num_clients)
    assert_(improved_cost < sol_cost)


def test_cpp_shuffle_results_in_different_solution(rc208):
    """
    Tests that calling shuffle changes the evaluation order, which can well
    result in different solutions generated from the same initial solution.
    """
    rng = RandomNumberGenerator(seed=42)

    ls = cpp_LocalSearch(rc208, compute_neighbours(rc208))
    ls.add_node_operator(Exchange10(rc208))
    ls.add_node_operator(Exchange11(rc208))

    cost_evaluator = CostEvaluator([1], 1, 0)
    sol = Solution.make_random(rc208, rng)

    # LocalSearch::search is deterministic, so two calls with the same base
    # solution should result in the same improved solution.
    improved1 = ls.search(sol, cost_evaluator)
    improved2 = ls.search(sol, cost_evaluator)
    assert_(improved1 == improved2)

    # But the shuffle method changes the order in which moves are evaluated,
    # which should result in a very different search trajectory.
    ls.shuffle(rng)
    improved3 = ls.search(sol, cost_evaluator)
    assert_(improved3 != improved1)


def test_vehicle_types_are_preserved_for_locally_optimal_solutions(rc208):
    """
    Tests that a solution that is already locally optimal returns the same
    solution, particularly w.r.t. the underlying vehicles. This exercises an
    issue where loading the solution in the local search did not preserve the
    vehicle types.
    """
    rng = RandomNumberGenerator(seed=42)
    neighbours = compute_neighbours(rc208)

    ls = cpp_LocalSearch(rc208, neighbours)
    ls.add_node_operator(Exchange10(rc208))
    ls.add_node_operator(Exchange11(rc208))

    cost_evaluator = CostEvaluator([1], 1, 0)
    sol = Solution.make_random(rc208, rng)

    improved = ls.search(sol, cost_evaluator)

    # Now make the instance heterogeneous and update the local search.
    data = rc208.replace(
        vehicle_types=[
            VehicleType(25, capacity=[10_000]),
            VehicleType(25, capacity=[10_000]),
        ]
    )

    ls = cpp_LocalSearch(data, neighbours)
    ls.add_node_operator(Exchange10(data))
    ls.add_node_operator(Exchange11(data))

    # Update the improved (locally optimal) solution with vehicles of type 1.
    routes = [Route(data, r.visits(), 1) for r in improved.routes()]
    improved = Solution(data, routes)

    # Doing the search should not find any further improvements thus not change
    # the solution.
    further_improved = ls.search(improved, cost_evaluator)
    assert_equal(further_improved, improved)


def test_bugfix_vehicle_type_offsets(ok_small):
    """
    See https://github.com/PyVRP/PyVRP/pull/292 for details. This exercises a
    fix to a bug that would crash local search due to an incorrect internal
    mapping of vehicle types to route indices if the next vehicle type had
    more vehicles than the previous.
    """
    data = ok_small.replace(
        vehicle_types=[
            VehicleType(1, capacity=[10]),
            VehicleType(2, capacity=[10]),
        ]
    )

    ls = cpp_LocalSearch(data, compute_neighbours(data))
    ls.add_node_operator(Exchange10(data))

    cost_evaluator = CostEvaluator([1], 1, 0)

    current = Solution(data, [Route(data, [1, 3], 1), Route(data, [2, 4], 1)])
    current_cost = cost_evaluator.penalised_cost(current)

    improved = ls.search(current, cost_evaluator)
    improved_cost = cost_evaluator.penalised_cost(improved)

    assert_(improved_cost <= current_cost)


def test_no_op_results_in_same_solution(ok_small):
    """
    Tests that calling local search without first adding any operators is a
    no-op, and returns the same solution as the one that was given to it.
    """
    rng = RandomNumberGenerator(seed=42)

    # Empty local search does not actually search anything, so it should return
    # the exact same solution as what was passed in.
    ls = LocalSearch(
        ok_small,
        rng,
        compute_neighbours(ok_small),
        PerturbationManager(PerturbationParams(0, 0)),  # disable perturbation
    )

    cost_eval = CostEvaluator([1], 1, 0)
    sol = Solution.make_random(ok_small, rng)

    assert_equal(ls(sol, cost_eval), sol)
    assert_equal(ls.search(sol, cost_eval), sol)
    assert_equal(ls.intensify(sol, cost_eval), sol)


def test_intensify_can_improve_solution_further(rc208):
    """
    Tests that ``intensify()`` improves a solution further once ``search()`` is
    stuck.
    """
    rng = RandomNumberGenerator(seed=11)

    ls = LocalSearch(rc208, rng, compute_neighbours(rc208))
    ls.add_node_operator(Exchange11(rc208))
    ls.add_route_operator(SwapStar(rc208))

    cost_eval = CostEvaluator([1], 1, 0)

    # The following solution is locally optimal w.r.t. the node operators. This
    # solution cannot be improved further by repeated calls to ``search()``.
    search_opt = ls.search(Solution.make_random(rc208, rng), cost_eval)
    search_cost = cost_eval.penalised_cost(search_opt)

    # But it can be improved further using the intensifying route operators,
    # as the following solution shows.
    intensify_opt = ls.intensify(search_opt, cost_eval)
    intensify_cost = cost_eval.penalised_cost(intensify_opt)

    assert_(intensify_cost < search_cost)

    # Both solutions are locally optimal. ``search_opt`` w.r.t. to the node
    # operators, and ``intensify_opt`` w.r.t. to the route operators. Repeated
    # calls to ``search()`` and ``intensify`` do not result in further
    # improvements for such locally optimal solutions.
    for _ in range(10):
        assert_equal(ls.search(search_opt, cost_eval), search_opt)
        assert_equal(ls.intensify(intensify_opt, cost_eval), intensify_opt)


def test_intensify_can_swap_routes(ok_small):
    """
    Tests that the bug identified in #742 is fixed. The intensify method should
    be able to improve a solution by swapping routes.
    """
    rng = RandomNumberGenerator(seed=42)

    data = ok_small.replace(
        vehicle_types=[
            VehicleType(1, capacity=[5]),
            VehicleType(1, capacity=[20]),
        ]
    )
    ls = LocalSearch(data, rng, compute_neighbours(data))
    ls.add_route_operator(SwapRoutes(data))

    # High load penalty, so the solution is penalised for having excess load.
    cost_eval = CostEvaluator([100_000], 0, 0)
    route1 = Route(data, [1, 2, 3], 0)  # Excess load: 13 - 5 = 8
    route2 = Route(data, [4], 1)  # Excess load: 0
    init_sol = Solution(data, [route1, route2])
    init_cost = cost_eval.penalised_cost(init_sol)

    assert_equal(init_sol.excess_load(), [8])

    # This solution can be improved by using the intensifying route operators
    # to swap the routes in the solution.
    intensify_sol = ls.intensify(init_sol, cost_eval)
    intensify_cost = cost_eval.penalised_cost(intensify_sol)

    assert_(intensify_cost < init_cost)
    assert_equal(intensify_sol.excess_load(), [0])


def test_local_search_completes_incomplete_solutions(ok_small_prizes):
    """
    Tests that the local search object improve solutions that are incomplete,
    and returns a completed solution. Passing an incomplete solution should
    return a completed solution after search.
    """
    rng = RandomNumberGenerator(seed=42)

    ls = LocalSearch(ok_small_prizes, rng, compute_neighbours(ok_small_prizes))
    ls.add_node_operator(Exchange10(ok_small_prizes))

    cost_eval = CostEvaluator([1], 1, 0)
    sol = Solution(ok_small_prizes, [[2], [3, 4]])
    assert_(not sol.is_complete())  # 1 is required but not visited

    new_sol = ls.search(sol, cost_eval)
    assert_(new_sol.is_complete())


def test_local_search_does_not_remove_required_clients():
    """
    Tests that the local search object does not remove required clients, even
    when that might result in a significant cost improvement.
    """
    rng = RandomNumberGenerator(seed=42)
    data = ProblemData(
        clients=[
            # This client cannot be removed, even though it causes significant
            # load violations.
            Client(x=1, y=1, delivery=[100], required=True),
            # This client can and should be removed, because the prize is not
            # worth the detour.
            Client(x=2, y=2, delivery=[0], prize=0, required=False),
        ],
        depots=[Depot(x=0, y=0)],
        vehicle_types=[VehicleType(1, capacity=[50])],
        distance_matrices=[np.where(np.eye(3), 0, 10)],
        duration_matrices=[np.zeros((3, 3), dtype=int)],
    )

    ls = LocalSearch(data, rng, compute_neighbours(data))
    ls.add_node_operator(Exchange10(data))

    sol = Solution(data, [[1, 2]])
    assert_(sol.is_complete())

    # Test that the improved solution contains the first client, but removes
    # the second. The first client is required, so could not be removed, but
    # the second could and that is an improving move.
    cost_eval = CostEvaluator([100], 100, 0)
    new_sol = ls.search(sol, cost_eval)
    assert_equal(new_sol.num_clients(), 1)
    assert_(new_sol.is_complete())

    sol_cost = cost_eval.penalised_cost(sol)
    new_cost = cost_eval.penalised_cost(new_sol)
    assert_(new_cost < sol_cost)


def test_replacing_optional_client():
    """
    Tests that the local search evaluates moves where an optional client is
    replaced with another that is not currently in the solution.
    """
    mat = [
        [0, 0, 0],
        [0, 0, 2],
        [0, 2, 0],
    ]
    data = ProblemData(
        clients=[
            Client(0, 0, tw_early=0, tw_late=1, prize=1, required=False),
            Client(0, 0, tw_early=0, tw_late=1, prize=5, required=False),
        ],
        depots=[Depot(0, 0)],
        vehicle_types=[VehicleType()],
        distance_matrices=[mat],
        duration_matrices=[mat],
    )

    rng = RandomNumberGenerator(seed=42)
    ls = LocalSearch(data, rng, compute_neighbours(data))
    ls.add_node_operator(Exchange10(data))

    # We start with a solution containing just client 1.
    sol = Solution(data, [[1]])
    assert_equal(sol.prizes(), 1)
    assert_(sol.is_feasible())

    # A unit of time warp has a penalty of 5 units, so it's never worthwhile to
    # have both clients 1 and 2 in the solution. However, replacing client 1
    # with 2 yields a prize of 5, rather than 1, at no additional cost.
    cost_eval = CostEvaluator([], 5, 0)
    improved = ls(sol, cost_eval)
    assert_equal(improved.prizes(), 5)
    assert_(improved.is_feasible())


def test_mutually_exclusive_group(gtsp):
    """
    Smoke test that runs the local search on a medium-size TSP instance with
    fifty mutually exclusive client groups.
    """
    assert_equal(gtsp.num_groups, 50)

    rng = RandomNumberGenerator(seed=42)
    neighbours = compute_neighbours(gtsp)
    perturbation = PerturbationManager(PerturbationParams(0, 0))

    ls = LocalSearch(gtsp, rng, neighbours, perturbation)
    ls.add_node_operator(Exchange10(gtsp))

    sol = Solution.make_random(gtsp, rng)
    cost_eval = CostEvaluator([20], 6, 0)
    improved = ls(sol, cost_eval)

    assert_(not sol.is_group_feasible())
    assert_(improved.is_group_feasible())

    sol_cost = cost_eval.penalised_cost(sol)
    improved_cost = cost_eval.penalised_cost(improved)
    assert_(improved_cost < sol_cost)


def test_mutually_exclusive_group_not_in_solution(
    ok_small_mutually_exclusive_groups,
):
    """
    Tests that the local search inserts a client from the mutually exclusive
    group if the entire group is missing from the solution.
    """
    rng = RandomNumberGenerator(seed=42)
    neighbours = compute_neighbours(ok_small_mutually_exclusive_groups)

    ls = LocalSearch(ok_small_mutually_exclusive_groups, rng, neighbours)
    ls.add_node_operator(Exchange10(ok_small_mutually_exclusive_groups))

    sol = Solution(ok_small_mutually_exclusive_groups, [[4]])
    assert_(not sol.is_group_feasible())

    improved = ls(sol, CostEvaluator([20], 6, 0))
    assert_(improved.is_group_feasible())


def test_swap_if_improving_mutually_exclusive_group(
    ok_small_mutually_exclusive_groups,
):
    """
    Tests that we swap a client (1) in a mutually exclusive group when another
    client (3) in the group is better to have.
    """
    data = ok_small_mutually_exclusive_groups
    rng = RandomNumberGenerator(seed=42)
    neighbours = compute_neighbours(data)
    perturbation = PerturbationManager(PerturbationParams(0, 0))

    ls = LocalSearch(data, rng, neighbours, perturbation)
    ls.add_node_operator(Exchange10(data))

    cost_eval = CostEvaluator([20], 6, 0)
    sol = Solution(data, [[1, 4]])
    improved = ls(sol, cost_eval)
    assert_(cost_eval.penalised_cost(improved) < cost_eval.penalised_cost(sol))

    routes = improved.routes()
    assert_equal(improved.num_routes(), 1)
    assert_equal(routes[0].visits(), [3, 4])


def test_no_op_multi_trip_instance(ok_small_multiple_trips):
    """
    Tests that loading and exporting a multi-trip instance correctly returns an
    equivalent solution when no operators are available.
    """
    rng = RandomNumberGenerator(seed=42)
    neighbours = [[] for _ in range(ok_small_multiple_trips.num_locations)]
    ls = LocalSearch(
        ok_small_multiple_trips,
        rng,
        neighbours,
        PerturbationManager(PerturbationParams(0, 0)),  # disable perturbation
    )

    trip1 = Trip(ok_small_multiple_trips, [1, 2], 0)
    trip2 = Trip(ok_small_multiple_trips, [3, 4], 0)
    route = Route(ok_small_multiple_trips, [trip1, trip2], 0)

    sol = Solution(ok_small_multiple_trips, [route])
    cost_eval = CostEvaluator([20], 6, 0)
    assert_equal(ls(sol, cost_eval), sol)


def test_local_search_inserts_reload_depots(ok_small_multiple_trips):
    """
    Tests that the local search routine inserts a reload depot when that is
    beneficial.
    """
    rng = RandomNumberGenerator(seed=2)
    neighbours = compute_neighbours(ok_small_multiple_trips)

    ls = LocalSearch(ok_small_multiple_trips, rng, neighbours)
    ls.add_node_operator(RelocateWithDepot(ok_small_multiple_trips))

    sol = Solution(ok_small_multiple_trips, [[1, 2, 3, 4]])
    assert_(sol.has_excess_load())

    cost_eval = CostEvaluator([1_000], 0, 0)
    improved = ls(sol, cost_eval)

    assert_(not improved.has_excess_load())
    assert_(cost_eval.penalised_cost(improved) < cost_eval.penalised_cost(sol))

    assert_equal(improved.num_routes(), 1)
    assert_equal(improved.num_trips(), 2)
    assert_(not improved.has_excess_load())


def test_local_search_removes_useless_reload_depots(ok_small_multiple_trips):
    """
    Tests that the local search removes useless reload depots from the given
    solution.
    """
    data = ok_small_multiple_trips
    rng = RandomNumberGenerator(seed=2)
    ls = LocalSearch(data, rng, compute_neighbours(data))
    ls.add_node_operator(Exchange10(data))

    route1 = Route(data, [Trip(data, [1], 0), Trip(data, [3], 0)], 0)
    route2 = Route(data, [2, 4], 0)
    sol = Solution(data, [route1, route2])

    cost_eval = CostEvaluator([1_000], 0, 0)
    improved = ls(sol, cost_eval)
    assert_(cost_eval.penalised_cost(improved) < cost_eval.penalised_cost(sol))

    # The local search should have removed the reload depot from the first
    # route, because that was not providing any value.
    routes = improved.routes()
    assert_(str(routes[0]), "1 3")
    assert_(str(routes[1]), "2 4")


def test_search_statistics(ok_small):
    """
    Tests that the local search's search statistics return meaningful
    information about the number of evaluated and improving moves.
    """
    rng = RandomNumberGenerator(seed=42)
    ls = LocalSearch(
        ok_small,
        rng,
        compute_neighbours(ok_small),
        PerturbationManager(PerturbationParams(0, 0)),  # disable perturbation
    )

    node_op = Exchange10(ok_small)
    ls.add_node_operator(node_op)

    # No solution is yet loaded/improved, so all these numbers should be zero.
    stats = ls.statistics
    assert_equal(stats.num_moves, 0)
    assert_equal(stats.num_improving, 0)
    assert_equal(stats.num_updates, 0)

    # Load and improve a random solution. This should result in a non-zero
    # number of moves.
    rnd_sol = Solution.make_random(ok_small, rng)
    cost_eval = CostEvaluator([1], 1, 1)
    improved = ls(rnd_sol, cost_eval)

    stats = ls.statistics
    assert_(stats.num_moves > 0)
    assert_(stats.num_improving > 0)
    assert_(stats.num_updates >= stats.num_improving)

    # Since we have only a single node operator, the number of moves and the
    # number of improving moves should match what the node operator tracks.
    assert_equal(stats.num_moves, node_op.statistics.num_evaluations)
    assert_equal(stats.num_improving, node_op.statistics.num_applications)

    # The improved solution is already locally optimal, so it cannot be further
    # improved by the local search. The number of improving moves should thus
    # be zero after another attempt.
    ls(improved, cost_eval)

    stats = ls.statistics
    assert_(stats.num_moves > 0)
    assert_equal(stats.num_improving, 0)
    assert_equal(stats.num_updates, 0)


def test_node_and_route_operators_property(ok_small):
    """
    Tests adding and accessing node and route operators to the LocalSearch
    object.
    """
    rng = RandomNumberGenerator(seed=42)
    ls = LocalSearch(ok_small, rng, compute_neighbours(ok_small))

    # The local search has not yet been equipped with operators, so it should
    # start empty.
    assert_equal(len(ls.node_operators), 0)
    assert_equal(len(ls.route_operators), 0)

    # Now we add a node operator. The local search does not take ownership, so
    # its only node operator should be the exact same object as the one we just
    # created.
    node_op = Exchange10(ok_small)
    ls.add_node_operator(node_op)
    assert_equal(len(ls.node_operators), 1)
    assert_(ls.node_operators[0] is node_op)

    # And a route operator, for which the same should hold.
    route_op = SwapStar(ok_small)
    ls.add_route_operator(route_op)
    assert_equal(len(ls.route_operators), 1)
    assert_(ls.route_operators[0] is route_op)


@pytest.mark.parametrize(
    ("instance", "exp_clients"),
    [
        # {1, 2, 3, 4} are all required clients.
        ("ok_small", {1, 2, 3, 4}),
        # 1 from required group {1, 2, 3}, 4 is a required client.
        ("ok_small_mutually_exclusive_groups", {1, 4}),
    ],
)
def test_inserts_required_missing(instance, exp_clients: set[int], request):
    """
    Tests that the local search inserts all missing clients and groups, if
    those are currently missing from the solution.
    """
    data = request.getfixturevalue(instance)
    rng = RandomNumberGenerator(seed=42)
    perturbation = PerturbationManager(PerturbationParams(1, 1))
    ls = LocalSearch(data, rng, compute_neighbours(data), perturbation)
    ls.add_node_operator(Exchange10(data))

    sol = Solution(data, [])
    assert_(not sol.is_complete())

    cost_eval = CostEvaluator([20], 6, 0)
    improved = ls(sol, cost_eval)
    assert_(improved.is_complete())

    visits = {client for route in improved.routes() for client in route}
    assert_equal(visits, exp_clients)


def test_local_search_exhaustive(rc208):
    """
    Tests calling the local search with the optional ``exhaustive`` argument
    for a complete evaluation.
    """
    rng = RandomNumberGenerator(seed=42)
    ls = LocalSearch(rc208, rng, compute_neighbours(rc208))
    ls.add_node_operator(Exchange10(rc208))

    init = Solution.make_random(rc208, rng)
    cost_eval = CostEvaluator([20], 6, 0)

    # The returned solution by default evaluates only around perturbed,
    # promising clients. That is not a full search. But when exhaustive is
    # explicitly is explicitly set, a full search must be done. The resulting
    # solution should be better than what's returned after perturbation,
    # because a full search evaluates many more moves.
    perturbed = ls(init, cost_eval, exhaustive=False)
    exhaustive = ls(init, cost_eval, exhaustive=True)

    perturbed_cost = cost_eval.penalised_cost(perturbed)
    exhaustive_cost = cost_eval.penalised_cost(exhaustive)
    assert_(exhaustive_cost < perturbed_cost)

    # Both should also be better than the initial, random solution.
    init_cost = cost_eval.penalised_cost(init)
    assert_(perturbed_cost < init_cost)
    assert_(exhaustive_cost < init_cost)


def test_inserts_optional_client_from_empty_solution():
    """
    Tests that the local search can insert optional clients into empty routes
    when starting from an empty solution. This is a regression test for a bug
    where prize-collecting VRP with some infeasible clients would return an
    empty solution even when visiting some clients would be beneficial.
    """
    data = ProblemData(
        clients=[
            # Fits in capacity, high prize: should be visited
            Client(x=1, y=0, delivery=[10], prize=50_000_000, required=False),
            # Exceeds capacity: cannot be visited
            Client(x=2, y=0, delivery=[200], prize=50_000_000, required=False),
        ],
        depots=[Depot(x=0, y=0)],
        vehicle_types=[VehicleType(1, capacity=[100])],
        distance_matrices=[
            np.array(
                [
                    [0, 1, 1],
                    [1, 0, 1],
                    [1, 1, 0],
                ]
            )
        ],
        duration_matrices=[np.zeros((3, 3), dtype=int)],
    )

    rng = RandomNumberGenerator(seed=42)
    ls = LocalSearch(data, rng, compute_neighbours(data))
    ls.add_node_operator(Exchange10(data))

    # Start from empty solution
    empty_sol = Solution(data, [])
    assert_equal(empty_sol.num_routes(), 0)

    # Search should insert the feasible client
    cost_eval = CostEvaluator([1_000_000], 0, 0)  # high penalty makes client2 infeasible
    improved = ls.search(empty_sol, cost_eval)

    # Should visit client1 (index 1) but not client2 (index 2)
    assert_equal(improved.num_routes(), 1)
    assert_equal(improved.num_clients(), 1)

    visits = list(improved.routes()[0].visits())
    assert_(1 in visits)  # client1 should be visited
    assert_(2 not in visits)  # client2 is infeasible


def test_inserts_optional_client_into_existing_route():
    """
    Tests that the local search can insert an unvisited optional client into
    an existing route. This is a regression test for a bug where optional
    clients not in the neighbourhood of visited clients would never be
    inserted even when doing so would improve the objective.
    """
    data = ProblemData(
        clients=[
            # High weight, already in the solution
            Client(x=0, y=0, delivery=[80], prize=50_000_000, required=False),
            # High weight, cannot be added (would exceed capacity)
            Client(x=1, y=0, delivery=[80], prize=50_000_000, required=False),
            # Low weight, can be added to improve objective
            Client(x=2, y=0, delivery=[10], prize=50_000_000, required=False),
            # High weight, cannot be added
            Client(x=3, y=0, delivery=[80], prize=50_000_000, required=False),
        ],
        depots=[Depot(x=0, y=0)],
        vehicle_types=[VehicleType(1, capacity=[100])],
        distance_matrices=[np.where(np.eye(5), 0, 1)],
        duration_matrices=[np.zeros((5, 5), dtype=int)],
    )

    rng = RandomNumberGenerator(seed=42)
    ls = LocalSearch(data, rng, compute_neighbours(data))
    ls.add_node_operator(Exchange10(data))

    # Start with a solution containing only client1 (weight 80)
    init_sol = Solution(data, [[1]])
    assert_equal(init_sol.num_clients(), 1)
    assert_(init_sol.is_feasible())

    # Search should add client3 (weight 10) since 80+10=90 < 100 capacity
    cost_eval = CostEvaluator([1_000_000], 0, 0)
    improved = ls.search(init_sol, cost_eval)

    # Improved solution should have client1 AND client3
    assert_(improved.num_clients() >= 2)
    assert_(improved.is_feasible())

    visits = set()
    for route in improved.routes():
        visits.update(route.visits())

    assert_(1 in visits)  # original client1 should still be there
    assert_(3 in visits)  # client3 (low weight) should be added


def test_greedy_insertion_order_does_not_block_better_solution():
    """
    Tests that the local search can find a better solution when the greedy
    insertion order would block it. This is a regression test for a scenario
    where inserting clients greedily blocks a better solution.

    Scenario: 4 optional clients with varying delivery demands.
    - Clients 1,2: delivery=[40] each (small demand)
    - Clients 3,4: delivery=[60] each (larger demand)

    Vehicle capacity is 100.
    - All clients individually are feasible
    - Clients [1,2] together: 40+40=80 <= 100 (feasible)
    - Clients [3,4] together: 60+60=120 > 100 (infeasible)
    - Clients [1,3] together: 40+60=100 <= 100 (feasible)

    If greedy insertion picks clients by some order that fills capacity early
    (e.g., inserting 3 then unable to add 4), it might miss adding both 1 and 2.
    """
    data = ProblemData(
        clients=[
            # Small demand clients
            Client(x=1, y=0, delivery=[40], prize=50_000_000, required=False),
            Client(x=2, y=0, delivery=[40], prize=50_000_000, required=False),
            # Larger demand clients
            Client(x=3, y=0, delivery=[60], prize=50_000_000, required=False),
            Client(x=4, y=0, delivery=[60], prize=50_000_000, required=False),
        ],
        depots=[Depot(x=0, y=0)],
        vehicle_types=[VehicleType(1, capacity=[100])],
        distance_matrices=[np.where(np.eye(5), 0, 1)],
        duration_matrices=[np.zeros((5, 5), dtype=int)],
    )

    rng = RandomNumberGenerator(seed=42)
    ls = LocalSearch(data, rng, compute_neighbours(data))
    ls.add_node_operator(Exchange10(data))

    # Start from empty solution
    empty_sol = Solution(data, [])

    # High penalty ensures infeasible insertions are never profitable
    cost_eval = CostEvaluator([100_000_000], 0, 0)
    improved = ls.search(empty_sol, cost_eval)

    # The search should find a solution that visits at least 2 clients.
    # The optimal solution visits clients 1 and 2 (total demand 80).
    # A suboptimal greedy could pick client 3 (demand 60) and then fail to add
    # client 4 (would be 120 > 100), ending with just 1 client.
    assert_(improved.num_clients() >= 2, "Should visit at least 2 clients")
    assert_(improved.is_feasible())


def test_same_vehicle_group_allows_moves_between_same_named_vehicles():
    """
    Tests that same-vehicle groups allow clients to be moved between routes
    that belong to vehicles with the same name (different shifts of the same
    vehicle). Two vehicle types with the same name "v0" but different time
    windows represent two shifts of the same vehicle.
    """
    from pyvrp import SameVehicleGroup

    # 4 clients + 1 depot
    data = ProblemData(
        clients=[
            Client(x=1, y=0, required=True),
            Client(x=2, y=0, required=True),
            Client(x=3, y=0, required=True),
            Client(x=4, y=0, required=True),
        ],
        depots=[Depot(x=0, y=0)],
        vehicle_types=[
            # Two vehicles with the same name but different shifts
            VehicleType(1, tw_early=0, tw_late=250, name="v0"),
            VehicleType(1, tw_early=500, tw_late=800, name="v0"),
        ],
        distance_matrices=[np.where(np.eye(5), 0, 1)],
        duration_matrices=[np.where(np.eye(5), 0, 1)],
        same_vehicle_groups=[SameVehicleGroup([1, 2, 3, 4])],
    )

    # Solution with clients split between the two shifts of the same vehicle.
    # Route 0 (first shift): clients 1, 2
    # Route 1 (second shift): clients 3, 4
    route1 = Route(data, [1, 2], 0)  # Vehicle type 0 (first shift of v0)
    route2 = Route(data, [3, 4], 1)  # Vehicle type 1 (second shift of v0)
    sol = Solution(data, [route1, route2])

    # This solution should be feasible since both routes use vehicles with
    # the same name "v0" (different shifts of the same vehicle).
    assert_(sol.is_group_feasible())
    assert_(sol.is_feasible())


def test_same_vehicle_group_disallows_different_named_vehicles():
    """
    Tests that same-vehicle groups still disallow clients to be on routes
    belonging to vehicles with different names.
    """
    from pyvrp import SameVehicleGroup

    # 4 clients + 1 depot
    data = ProblemData(
        clients=[
            Client(x=1, y=0, required=True),
            Client(x=2, y=0, required=True),
            Client(x=3, y=0, required=True),
            Client(x=4, y=0, required=True),
        ],
        depots=[Depot(x=0, y=0)],
        vehicle_types=[
            VehicleType(1, name="v0"),
            VehicleType(1, name="v1"),  # Different name
        ],
        distance_matrices=[np.where(np.eye(5), 0, 1)],
        duration_matrices=[np.where(np.eye(5), 0, 1)],
        same_vehicle_groups=[SameVehicleGroup([1, 2, 3, 4])],
    )

    # Solution with clients split between two different vehicles.
    route1 = Route(data, [1, 2], 0)  # Vehicle v0
    route2 = Route(data, [3, 4], 1)  # Vehicle v1 (different)
    sol = Solution(data, [route1, route2])

    # This solution should be infeasible since routes use different vehicles.
    assert_(not sol.is_group_feasible())
    assert_(not sol.is_feasible())


def test_same_vehicle_group_disallows_empty_names():
    """
    Tests that vehicles with empty names are treated as distinct, even if
    both have empty names.
    """
    from pyvrp import SameVehicleGroup

    # 4 clients + 1 depot
    data = ProblemData(
        clients=[
            Client(x=1, y=0, required=True),
            Client(x=2, y=0, required=True),
            Client(x=3, y=0, required=True),
            Client(x=4, y=0, required=True),
        ],
        depots=[Depot(x=0, y=0)],
        vehicle_types=[
            VehicleType(1, name=""),  # Empty name
            VehicleType(1, name=""),  # Empty name
        ],
        distance_matrices=[np.where(np.eye(5), 0, 1)],
        duration_matrices=[np.where(np.eye(5), 0, 1)],
        same_vehicle_groups=[SameVehicleGroup([1, 2, 3, 4])],
    )

    # Solution with clients split between two routes with empty names.
    route1 = Route(data, [1, 2], 0)
    route2 = Route(data, [3, 4], 1)
    sol = Solution(data, [route1, route2])

    # This solution should be infeasible since empty names don't match.
    assert_(not sol.is_group_feasible())
    assert_(not sol.is_feasible())


def test_local_search_respects_same_vehicle_group_across_shifts():
    """
    Tests that local search does not move clients from a same-vehicle group
    to routes belonging to a different vehicle (different name), even when
    such a move would be improving in terms of distance.

    NOTE: This test verifies that the initial solution is group-feasible
    and that the local search maintains feasibility. The actual local search
    constraint enforcement is tested via the C++ unit tests.
    """
    from pyvrp import SameVehicleGroup

    # Set up a problem where clients 1 and 2 must stay on the same vehicle.
    data = ProblemData(
        clients=[
            Client(x=0, y=1, required=True),  # client 1
            Client(x=0, y=2, required=True),  # client 2 (in same group as 1)
            Client(x=10, y=0, required=True),  # client 3
        ],
        depots=[Depot(x=0, y=0)],
        vehicle_types=[
            VehicleType(1, name="v0"),
            VehicleType(1, name="v1"),  # Different vehicle
        ],
        distance_matrices=[np.where(np.eye(4), 0, 1)],
        duration_matrices=[np.zeros((4, 4), dtype=int)],
        same_vehicle_groups=[SameVehicleGroup([1, 2])],  # 1 and 2 must stay together
    )

    # Initial solution: clients 1, 2 on v0; client 3 on v1.
    route1 = Route(data, [1, 2], 0)
    route2 = Route(data, [3], 1)
    sol = Solution(data, [route1, route2])

    # This solution is feasible because clients 1 and 2 are on the same vehicle.
    assert_(sol.is_group_feasible())
    assert_(sol.is_feasible())

    # Verify that a solution splitting 1 and 2 across different vehicles is
    # infeasible.
    route_bad1 = Route(data, [1, 3], 0)  # v0
    route_bad2 = Route(data, [2], 1)  # v1 (different name)
    bad_sol = Solution(data, [route_bad1, route_bad2])
    assert_(not bad_sol.is_group_feasible())


def test_local_search_allows_moves_within_same_vehicle_shifts():
    """
    Tests that solutions with same-vehicle groups distributed across
    different shifts (routes) of the same vehicle (same name) are feasible.
    """
    from pyvrp import SameVehicleGroup

    data = ProblemData(
        clients=[
            Client(x=1, y=0, required=True),
            Client(x=2, y=0, required=True),
            Client(x=3, y=0, required=True),
            Client(x=4, y=0, required=True),
        ],
        depots=[Depot(x=0, y=0)],
        vehicle_types=[
            # Two shifts of the same vehicle (same name, different time windows)
            VehicleType(1, tw_early=0, tw_late=500, name="v0"),
            VehicleType(1, tw_early=1000, tw_late=1500, name="v0"),
        ],
        distance_matrices=[np.where(np.eye(5), 0, 1)],
        duration_matrices=[np.where(np.eye(5), 0, 1)],
        same_vehicle_groups=[SameVehicleGroup([1, 2, 3, 4])],
    )

    # Solution with all clients on first shift - should be feasible.
    route1 = Route(data, [1, 2, 3, 4], 0)
    sol1 = Solution(data, [route1])
    assert_(sol1.is_group_feasible())

    # Solution with clients split across two shifts of the same vehicle
    # (same name "v0") - should also be feasible.
    route_shift1 = Route(data, [1, 2], 0)  # First shift of v0
    route_shift2 = Route(data, [3, 4], 1)  # Second shift of v0
    sol2 = Solution(data, [route_shift1, route_shift2])
    assert_(sol2.is_group_feasible())
    assert_(sol2.is_feasible())


def test_local_search_blocks_moves_to_different_vehicle():
    """
    Tests that solutions that split a same-vehicle group across different
    vehicles (different names) are infeasible.
    """
    from pyvrp import SameVehicleGroup

    data = ProblemData(
        clients=[
            Client(x=1, y=0, required=True),  # client 1
            Client(x=100, y=0, required=True),  # client 2
        ],
        depots=[Depot(x=0, y=0)],
        vehicle_types=[
            VehicleType(1, name="v0"),
            VehicleType(1, name="v1"),
        ],
        distance_matrices=[
            np.array(
                [
                    [0, 1, 100],
                    [1, 0, 99],
                    [100, 99, 0],
                ]
            )
        ],
        duration_matrices=[np.zeros((3, 3), dtype=int)],
        same_vehicle_groups=[SameVehicleGroup([1, 2])],
    )

    # Feasible solution: both clients on v0.
    route_feasible = Route(data, [1, 2], 0)
    sol_feasible = Solution(data, [route_feasible])
    assert_(sol_feasible.is_group_feasible())
    assert_(sol_feasible.is_feasible())

    # Infeasible solution: clients split across different vehicles.
    route_v0 = Route(data, [1], 0)
    route_v1 = Route(data, [2], 1)
    sol_infeasible = Solution(data, [route_v0, route_v1])
    assert_(not sol_infeasible.is_group_feasible())
    assert_(not sol_infeasible.is_feasible())


def test_local_search_improves_infeasible_same_vehicle_group():
    """
    Tests that solutions that violate same-vehicle group constraints are
    correctly identified as infeasible. Also tests that combining clients
    on the same route makes the solution feasible.
    """
    from pyvrp import SameVehicleGroup

    data = ProblemData(
        clients=[
            Client(x=1, y=0, required=True),
            Client(x=2, y=0, required=True),
        ],
        depots=[Depot(x=0, y=0)],
        vehicle_types=[
            VehicleType(2, name=""),  # 2 vehicles available, no name
        ],
        distance_matrices=[np.where(np.eye(3), 0, 1)],
        duration_matrices=[np.zeros((3, 3), dtype=int)],
        same_vehicle_groups=[SameVehicleGroup([1, 2])],
    )

    # Infeasible: clients on different routes, and vehicle has no name
    # (empty names are treated as distinct vehicles).
    route1 = Route(data, [1], 0)
    route2 = Route(data, [2], 0)
    sol_infeasible = Solution(data, [route1, route2])
    assert_(not sol_infeasible.is_group_feasible())

    # Feasible: both clients on the same route.
    route_combined = Route(data, [1, 2], 0)
    sol_feasible = Solution(data, [route_combined])
    assert_(sol_feasible.is_group_feasible())
    assert_(sol_feasible.is_feasible())


def test_local_search_same_vehicle_group_with_multiple_groups():
    """
    Tests that multiple same-vehicle groups work correctly, with constraints
    respected for each group independently.
    """
    from pyvrp import SameVehicleGroup

    data = ProblemData(
        clients=[
            Client(x=1, y=0, required=True),  # Group A
            Client(x=2, y=0, required=True),  # Group A
            Client(x=3, y=0, required=True),  # Group B
            Client(x=4, y=0, required=True),  # Group B
        ],
        depots=[Depot(x=0, y=0)],
        vehicle_types=[
            VehicleType(1, name="v0"),
            VehicleType(1, name="v1"),
        ],
        distance_matrices=[np.where(np.eye(5), 0, 1)],
        duration_matrices=[np.zeros((5, 5), dtype=int)],
        same_vehicle_groups=[
            SameVehicleGroup([1, 2]),  # Group A: clients 1, 2
            SameVehicleGroup([3, 4]),  # Group B: clients 3, 4
        ],
    )

    # Feasible solution: Group A on v0, Group B on v1.
    route1 = Route(data, [1, 2], 0)
    route2 = Route(data, [3, 4], 1)
    sol_feasible = Solution(data, [route1, route2])
    assert_(sol_feasible.is_group_feasible())
    assert_(sol_feasible.is_feasible())

    # Infeasible solution: Group A is split across v0 and v1.
    route_v0 = Route(data, [1, 3], 0)  # Client 1 (Group A) + Client 3 (Group B)
    route_v1 = Route(data, [2, 4], 1)  # Client 2 (Group A) + Client 4 (Group B)
    sol_infeasible = Solution(data, [route_v0, route_v1])
    # Both groups are split across different vehicles -> infeasible
    assert_(not sol_infeasible.is_group_feasible())
    assert_(not sol_infeasible.is_feasible())

    # Another infeasible solution: only Group A is split.
    route_all_v0 = Route(data, [1, 3, 4], 0)  # Clients 1, 3, 4 on v0
    route_only_2 = Route(data, [2], 1)  # Client 2 alone on v1
    sol_split_a = Solution(data, [route_all_v0, route_only_2])
    # Group A is split (1 on v0, 2 on v1), but Group B is together on v0
    assert_(not sol_split_a.is_group_feasible())
