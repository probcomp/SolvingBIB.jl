using OrderedCollections, Parameters
using Gen: ParticleFilterState
using Plinf: ObserveParams, PolicyState

include("utils.jl")

"Observer parameters, specifying a model of the agent and its environment."
mutable struct Observer
    n_goals::Int # Number of possible goals
    n_particles::Int # Number of particles
    domain::Domain # PDDL domain specification
    obs_params::ObserveParams # Observation noise parameters
    goal_params::Dict{Int,Vector{Float64}} # Per-agent Dirichlet hyperprior
    noise_priors::Dict{Int,NTuple{2,Vector{Float64}}} # Per-agent noise prior
    pf_state::Union{ParticleFilterState,Nothing} # Particle filter state
end

"Customized action proposal for the agency domain."
@gen function agency_act_proposal(t, domain, agent_state, state, next_obs)
    sx, sy = state[pddl"xpos"], state[pddl"ypos"]
    dx = next_obs[pddl"xpos"] - sx
    dy = next_obs[pddl"ypos"] - sy
    intended = Plinf.get_action(agent_state.plan_state)
    proposed = Compound[]
    if dx > 0 # Agent moved eastward
        act = dy > 0 ? pddl"southeast" : dy < 0 ? pddl"northeast" : pddl"east"
        act = available(domain, state, act) ? act : pddl"east"
        push!(proposed, Compound(act.name, []))
    elseif dx < 0 # Agent moved westward
        act = dy > 0 ? pddl"southwest" : dy < 0 ? pddl"northwest" : pddl"west"
        act = available(domain, state, act) ? act : pddl"west"
        push!(proposed, Compound(act.name, []))
    elseif dy != 0 # Agent moved north or south
        act = dy > 0 ? pddl"south" : pddl"north"
        push!(proposed, Compound(act.name, []))
    else # Agent stayed in the same location
        cur_items = PDDL.get_objects(domain, state, :item)
        obs_items = PDDL.get_objects(domain, next_obs, :item)
        # Check for items that may have been picked up
        for o in cur_items
            if (!state[Compound(:has, [o])] &&
                o in obs_items && next_obs[Compound(:has, [o])])
                push!(proposed, Compound(:pickup, [o]))
            end
        end
        # Check for potential unlock actions
        if all(next_obs[pddl"barriers"] .== false)
            for k in PDDL.get_objects(state, :key)
                if !state[Compound(:has, [k])] continue end
                if k in obs_items && next_obs[Compound(:has, [k])] continue end
                for l in PDDL.get_objects(state, :lock)
                    push!(proposed, Compound(:unlock, [k, l]))
                end
            end
        end
    end
    # Filter out non-available actions
    filtered = Term[]
    for act in available(domain, state)
        if Compound(act.name, act.args) in proposed push!(filtered, act) end
    end
    # Default to planned action if no proposed actions are valid
    if length(filtered) == 0 push!(filtered, intended) end
    return @trace(labeled_unif(filtered), :act)
end

@dist sym_binom(mean::Int, scale::Int) =
    binom(2*scale, 0.5) - scale + mean

"Initialize inference for a new scene by constructing an Observer."
function init_inference(n_goals::Int=2, n_particles::Int=50)
    # Load PDDL domain
    domain = DOMAIN
    clear_heuristic_cache!()
    clear_action_cache!(ASTAR_PLANNER)
    # Define observation noise model
    obs_params = observe_params(
        (pddl"xpos", sym_binom, 8), (pddl"ypos", sym_binom, 8)
    )
    # Set up dictionary of goal hyperpriors (one entry per agent)
    goal_params = Dict{Int,Vector{Float64}}()
    # Set up dictionary of action noise priors (one per agent)
    noise_priors = Dict{Int,NTuple{2,Vector{Float64}}}()
    return Observer(n_goals, n_particles, domain, obs_params,
                    goal_params, noise_priors, nothing)
end

"Resets the particle filter and update parameters at the start of a new trial."
function reset_inference!(observer::Observer, state::State)
    # Lookup compiled domain based on object signature for faster inference
    observer.domain = lookup_compiled_domain(PDDL.get_objects(state))
    if observer.domain isa CompiledDomain
        state = compilestate(observer.domain, state)
    end
    # Compute which goal objects are reachable
    reachable = find_reachable_goals(observer.domain, state)
    # Update goal hyperprior and noise prior probs for last observed agent
    if observer.pf_state !== nothing
        prev_agent_id = get_prev_agent_id(observer.pf_state)
        # Approximate conjugate update for goal hyperprior
        @assert prev_agent_id in keys(observer.goal_params)
        goal_probs = get_goal_probs(observer.pf_state, observer.n_goals)
        observer.goal_params[prev_agent_id] += goal_probs
        # Replace noise prior with posterior
        @assert prev_agent_id in keys(observer.noise_priors)
        noise_support, noise_probs = get_noise_probs(observer.pf_state)
        observer.noise_priors[prev_agent_id] = (noise_support, noise_probs)
    end
    agent_id = state[pddl"agentid"]
    # Initialize goal hyperprior for new agents to Dirichlet(1, 1, ...)
    goal_params = get!(observer.goal_params, agent_id, ones(observer.n_goals))
    # Initialize action noise prior for new agents as a gridded Gamma(3, 1)
    if agent_id in keys(observer.noise_priors)
        noise_support, noise_probs = observer.noise_priors[agent_id]
    else
        noise_support = [0.2, 0.283, 0.4, 0.566, 0.8]
        noise_weights = map(x -> exp(logpdf(gamma, x, 1, 1)), noise_support)
        noise_probs = noise_weights ./ sum(noise_weights)
        observer.noise_priors[agent_id] = (noise_support, noise_probs)
    end
    # Construct goal specifications, accounting for goal accessibility
    objects = sort(collect(PDDL.get_objects(observer.domain, state, :item)),
                           by=o->o.name)
    goal_specs = map(1:observer.n_goals) do goal_idx
        goal_obj = select_reachable_goal(goal_idx, objects, reachable)
        return goal_object_spec(goal_obj)
    end
    # Set new goal prior as expected value of the Dirichlet hyperprior
    goal_probs = collect(goal_params) ./ sum(goal_params)
    @gen function goal_prior()
        # Sample index of preferred object and select corresponding goal spec
        goal_idx = {:goal} ~ categorical(goal_probs)
        return goal_specs[goal_idx]
    end
    # Construct new action noise prior
    @gen function act_args_prior()
        act_noise ~ labeled_cat(noise_support, noise_probs)
        return (act_noise,)
    end
    # Set up Boltzmann agent model using RTDP planner
    heuristic = PlannerHeuristic(ASTAR_PLANNER) # Planner-in-the-loop
    planner = RTDPlanner(heuristic=heuristic, act_noise=0.0, n_rollouts=0)
    policies = Dict(goal => Plinf.solve(planner, observer.domain, state, goal)
                    for goal in goal_specs)
    plan_init = goal -> PolicyState(policies[goal], 0.0, Term[pddl"--"], [1.0])
    agent_init = AgentInit(goal_prior, plan_init)
    agent_config = BoltzmannAgentConfig(observer.domain, planner,
                                        act_args=act_args_prior)
    # Initialize world with new initial state, goal prior, and agent model
    world_init = WorldInit(agent_init, state, state)
    world_config = WorldConfig(observer.domain, agent_config,
                               observer.obs_params)
    # Set up initial observations
    obs_terms = collect(keys(observer.obs_params))
    obs_choices = state_choicemap(state, observer.domain, obs_terms,
                                  :init => :obs)
    # (Re)initialize particle filter with stratified sampling
    goal_addr = :init => :agent => :goal => :goal
    noise_addr = :init => :agent => :act_args => :act_noise
    strata = Dict(goal_addr => 1:observer.n_goals, noise_addr => noise_support)
    pf_state = Plinf.initialize_pf_stratified(
        world_model, (0, world_init, world_config),
        obs_choices, strata, observer.n_particles)
    observer.pf_state = pf_state
    return observer, goal_probs, (noise_support, noise_probs)
end

"""
    step_inference!(observer, state[, new_trial]; options...)

Step inference forward given a newly observed PDDL `state`.

# Options:
- `coeffs::Dict`: Stepwise plausibility rating coefficients.
- `verbose::Bool`: Flag to print extra information as inference runs.
"""
function step_inference!(
    observer::Observer, state::State, new_trial::Bool=false;
    coeffs::Dict=STEP_DECISION_COEFFS, verbose::Bool=false
)
    if new_trial # Reset particle filter and update parameters
        _, goal_probs, noise_prior = reset_inference!(observer, state)
        if verbose
            print_goal_probs(goal_probs, "Goal Prior: ")
            print_noise_probs(noise_prior, "Noise Prior: ")
        end
    end
    # Get previous goal/noise probabilities and log marginal likelihood estimate
    prev_goal_probs = get_goal_probs(observer.pf_state, observer.n_goals)
    _, prev_noise_probs = get_noise_probs(observer.pf_state)
    prev_lml_est = log_ml_estimate(observer.pf_state)
    # Update particle filter with newly observed state
    t, world_init, world_config = Gen.get_args(observer.pf_state.traces[1])
    obs_terms = collect(keys(observer.obs_params))
    domain = state isa CompiledState ? observer.domain : DOMAIN
    obs_choices = state_choicemap(state, domain, obs_terms,
                                  :timestep => t+1 => :obs)
    pf_update!(observer.pf_state, (t+1, world_init, world_config),
               (UnknownChange(), NoChange(), NoChange()), obs_choices,
               Plinf.propose_act, (state, agency_act_proposal, ()))
    # Compute metrics to return
    goal_probs = get_goal_probs(observer.pf_state, observer.n_goals)
    _, noise_probs = get_noise_probs(observer.pf_state)
    lml_est = log_ml_estimate(observer.pf_state)
    # Compute total variation between previous and current posteriors
    goal_tv = total_variation(prev_goal_probs, goal_probs)
    noise_tv = total_variation(prev_noise_probs, noise_probs)
    # Compute earth mover distances between noise posteriors
    noise_emd = earth_mover_dist(prev_noise_probs, noise_probs)
    # Compute marginal likelihood estimate of the newest observation
    log_likelihood = lml_est - prev_lml_est
    metadata = Dict(:goal_tv => goal_tv,
                    :noise_tv => noise_tv,
                    :noise_emd => noise_emd,
                    :log_likelihood => log_likelihood,
                    :goal_probs => goal_probs,
                    :noise_probs => noise_probs)
    # Decide plausibility rating
    rating = decide_step(metadata; coeffs=coeffs)
    metadata[:rating] = rating
    if verbose # Print values
        for p in goal_probs @printf("%.3f\t", p) end
        @printf("Goal TV: %.2f\t", goal_tv)
        @printf("Noise TV: %.2f\t", noise_tv)
        @printf("Noise EMD: %.2f\t", noise_emd)
        @printf("Log Likelihood: %.2f\t", log_likelihood)
        @printf("Plausibility: %.2f\t", rating)
        println()
    end
    return (rating, metadata)
end

"""
    scene_inference(trajs[, n_goals=2, n_samples=10]; options...)

Run hierarchical goal inference over multiple trajectories in a scene.

# Options:
- `step_coeffs::Dict`: Stepwise plausibility decision coefficients.
- `verbose::Bool`: Flag to print extra information as inference runs.
"""
function scene_inference(
    trajs, n_goals::Int=2, n_samples::Int=n_goals*5;
    step_coeffs::Dict=STEP_DECISION_COEFFS, verbose::Bool=false
)
    observer = init_inference(n_goals, n_samples)
    # Collect plausibility ratings and inference metadata
    all_ratings, all_metadata = Vector{Float64}[], []
    # Iterate over trajectories
    for (i, traj) in enumerate(trajs)
        ratings, metadata = Float64[], Dict{Symbol,Any}[]
        if verbose println("== Trial $i ==") end
        for (t, state) in enumerate(traj)
            # Step inference forward
            r, mdata = step_inference!(observer, state, t==1;
                                       coeffs=step_coeffs,
                                       verbose=verbose)
            push!(ratings, r)
            push!(metadata, mdata)
        end
        # Save buffers
        push!(all_ratings, ratings)
        push!(all_metadata, metadata)
    end
    return all_ratings, all_metadata
end
