module SolvingBIB

using Printf
using Gen, GenParticleFilters
using Julog, PDDL, Plinf, SymbolicPlanners

export init_inference, reset_inference!, step_inference!, scene_inference
export decide_step, decide_scene, total_variation
export load_json_scene, json_to_pddl, print_ascii
export solve_scene

# Enable the use of array-valued fluents in PDDL
PDDL.Arrays.@register()

include("planning.jl")
include("ascii.jl")
include("conversion.jl")
include("postprocess.jl")
include("decision.jl")
include("inference.jl")
include("generate.jl")

function __init__()
    # Compile and store domains for various object signatures
    problem_filenames = ["problem-agent-efficiency.pddl",
                         "problem-object-preference.pddl",
                         "problem-instrumental-action.pddl",
                         "problem-key-and-shape.pddl"]
    for fn in problem_filenames
        problem = load_problem(joinpath(PDDL_PATH, fn))
        state = magnify_state(initstate(DOMAIN, problem), SCALE_FACTOR)
        objects = Set(collect(PDDL.get_objects(state)))
        COMPILED_DOMAINS[objects] = compiled(DOMAIN, state)[1]
    end
end

"Solve scene by running inference, returning final scene decision."
function solve_scene(
    path::AbstractString;
    n_sample_mult::Int=5, scale::Int=SCALE_FACTOR, verbose::Bool=false,
    step_coeffs::Dict=STEP_DECISION_COEFFS,
    scene_coeffs::Dict=SCENE_DECISION_COEFFS
)
    # Load, convert, and preprocess scene
    if verbose println("Loading scene...") end
    scene = load_json_scene(path)
    trials = json_to_pddl(scene; scale=scale) .|> skip_barrier_removal .|> diagonalize_traj
    # Extract number of goals as number of items in initial state
    init_state = trials[1][1]
    n_goals = length(PDDL.get_objects(DOMAIN, init_state, :item))
    # Run inference over all trajectories for this scene
    if verbose println("\nRunning inference...") end
    all_ratings, all_metadata =
        scene_inference(trials, n_goals, n_sample_mult * n_goals;
                        step_coeffs=step_coeffs, verbose=verbose)
    if length(all_ratings) == 0 || length(all_metadata) == 0
        @warn "Inference produced empty outputs. Falling back to default values."
        rating, metadata = 1.0, Dict{Symbol,Any}()
        return rating, metadata
    end
    # Compute final decision based on metadata for test trial
    test_metadata = all_metadata[end]
    rating, features = decide_scene(test_metadata; coeffs=scene_coeffs)
    if verbose
        println("== Results ==")
        @printf("Scene Plausibility: %.2f\n", rating)
        println("Decision Features:")
        display(features)
    end
    clear_heuristic_cache!()
    clear_action_cache!(ASTAR_PLANNER)
    return rating, features
end

end
