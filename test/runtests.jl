using SolvingBIB, PDDL, Plinf, Test, Random
using SolvingBIB: discretize_to_grid

@testset "Forward Planning" begin
    path = joinpath(dirname(pathof(SolvingBIB)), "..", "pddl")
    domain = SolvingBIB.DOMAIN
    problem_names = [
        "object-preference",
        "agent-efficiency",
        "inaccessible-goal",
        "instrumental-action",
        "unfamiliar-agent",
    ]

    manhattan = ManhattanHeuristic(@pddl("xpos", "ypos"))
    planner = AStarPlanner(heuristic=manhattan)
    goal = pddl"(and (= xpos 2) (= ypos 3))"

    for pname in problem_names
        problem = load_problem(joinpath(path, "problem-$pname.pddl"))
        state = initstate(domain, problem)
        plan, traj = planner(domain, state, goal)
        @test satisfy(domain, traj[end], goal)
    end
end

@testset "Agent Efficiency (Efficient Habituation)" begin
    # Generate synthetic dataset of agent efficiency trials
    Random.seed!(0)
    plans, trajs = generate_efficiency_trajs(8)

    # Run inference on synthetically generated trajectories
    ratings, metadata = scene_inference(trajs, 1, 5)

    # Check that at least some of the ratings are implausible
    test_ratings = last(ratings)
    @test any(test_ratings .< 0.5)

    # Check that the test trial is implausible overall
    test_metadata = last(metadata)
    test_probs = [test_metadata[t][:noise_probs]
                  for t in 1:length(test_metadata)]
    @test total_variation(test_probs[1], test_probs[end]) >= 0.5
    rating, _ = decide_scene(test_metadata)
    @test rating < 0.5
end

@testset "Agent Efficiency (Inefficient Habituation)" begin
    # Generate synthetic dataset of agent efficiency trials
    Random.seed!(0)
    efficiencies = fill(false, 8)
    plans, trajs = generate_efficiency_trajs(8, efficiencies=efficiencies)

    # Run inference on synthetically generated trajectories
    ratings, metadata = scene_inference(trajs, 1, 5)

    # Check that no ratings are implausible
    test_ratings = last(ratings)
    @test !any(test_ratings .< 0.5)

    # Check that the test trial is plausible overall
    test_metadata = last(metadata)
    test_probs = [test_metadata[t][:noise_probs]
                  for t in 1:length(test_metadata)]
    @test total_variation(test_probs[1], test_probs[end]) < 0.5
    rating, _ = decide_scene(test_metadata)
    @test rating > 0.5
end

@testset "Object Preference" begin
    # Generate synthetic dataset of object preference trials
    Random.seed!(0)
    plans, trajs = generate_preference_trajs(8)

    # Run inference on synthetically generated trajectories
    ratings, metadata = scene_inference(trajs, 2, 10)

    # Check that at least some of the ratings are implausible
    test_ratings = last(ratings)
    @test any(test_ratings .< 0.5)

    # Check that the test trial is implausible overall
    test_metadata = last(metadata)
    test_probs = [test_metadata[t][:goal_probs]
                  for t in 1:length(test_metadata)]
    @test total_variation(test_probs[1], test_probs[end]) >= 0.5
    rating, _ = decide_scene(test_metadata)
    @test rating < 0.5
end

@testset "Inaccessible Goal" begin
    # Generate synthetic dataset of with inaccessible goal in test trial
    Random.seed!(0)
    plans, trajs = generate_inaccessible_trajs(8)

    # Run inference on synthetically generated trajectories
    ratings, metadata = scene_inference(trajs, 2, 10)

    # Check that all the ratings are plausible
    test_ratings = last(ratings)
    @test !any(test_ratings .< 0.5)

    # Check that the test trial is plausible overall
    test_metadata = last(metadata)
    test_probs = [test_metadata[t][:goal_probs]
                  for t in 1:length(test_metadata)]
    @test total_variation(test_probs[1], test_probs[end]) < 0.5
    rating, _ = decide_scene(test_metadata)
    @test rating > 0.5
end

@testset "Multiple Agents" begin
    # Generate synthetic dataset with unfamiliar agent in test trial
    Random.seed!(0)
    plans, trajs = generate_multiagent_trajs(8)

    # Run inference on synthetically generated trajectories
    ratings, metadata = scene_inference(trajs, 2, 10)

    # Check that the test trial is (relatively) plausible overall
    test_metadata = last(metadata)
    test_probs = [test_metadata[t][:goal_probs]
                  for t in 1:length(test_metadata)]
    @test total_variation(test_probs[1], test_probs[end]) < 0.5
    rating, _ = decide_scene(test_metadata)
    @test rating > 0.4
end

@testset "Instrumental Action (Missing Barrier)" begin
    # Generate synthetic dataset of agent efficiency trials
    Random.seed!(0)
    has_barriers = [fill(true, 7); false]
    plans, trajs = generate_instrumental_trajs(8, has_barriers=has_barriers)

    # Run inference on synthetically generated trajectories
    ratings, metadata = scene_inference(trajs, 2, 10)

    # Check that at least some of the ratings are implausible
    test_ratings = last(ratings)
    @test any(test_ratings .< 0.5)

    # Check that the test trial is implausible overall
    test_metadata = last(metadata)
    test_probs = [test_metadata[t][:goal_probs]
                  for t in 1:length(test_metadata)]
    @test total_variation(test_probs[1], test_probs[end]) >= 0.5
    rating, _ = decide_scene(test_metadata)
    @test rating < 0.5
end

@testset "Instrumental Action (Unnecessary Key)" begin
    # Generate synthetic dataset of agent efficiency trials
    Random.seed!(0)
    has_barriers = fill(true, 8)
    need_keys = [fill(true, 7); false]
    plans, trajs = generate_instrumental_trajs(8, has_barriers=has_barriers,
                                               need_keys=need_keys)

    # Run inference on synthetically generated trajectories
    ratings, metadata = scene_inference(trajs, 2, 10)

    # Check that at least some of the ratings are implausible
    test_ratings = last(ratings)
    @test any(test_ratings .< 0.5)

    # Check that the test trial is implausible overall
    test_metadata = last(metadata)
    test_probs = [test_metadata[t][:goal_probs]
                  for t in 1:length(test_metadata)]
    @test total_variation(test_probs[1], test_probs[end]) >= 0.5
    rating, _ = decide_scene(test_metadata)
    @test rating < 0.5
end
