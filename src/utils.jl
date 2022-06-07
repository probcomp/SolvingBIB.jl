"Unzips an array of tuples."
unzip(a) = (getfield.(a, x) for x in fieldnames(eltype(a)))

"Return the list of reachable goals."
function find_reachable_goals(domain::Domain, state::State)
    objects = PDDL.get_objects(domain, state, :item)
    reachable = Const[]
    for obj in objects
        goal = Compound(:has, Term[obj])
        plan, traj = ASTAR_PLANNER(domain, state, goal)
        if plan !== nothing push!(reachable, obj) end
    end
    return reachable
end

"Select goal based on which objects are reachable."
function select_reachable_goal(idx::Int, objects, reachable)
    # Select preferred object if reachable, otherwise choose first reachable
    goal_obj = objects[idx]
    return goal_obj in reachable ? goal_obj : reachable[1]
end

"Extract ID of most recently observed agent."
function get_prev_agent_id(pf_state::ParticleFilterState)
    trace = get_traces(pf_state)[1]
    return trace[:init => :env][pddl"agentid"]
end

"Compute goal probabilities from particle filter state."
function get_goal_probs(pf_state::ParticleFilterState, n_goals::Int)
    traces = get_traces(pf_state)
    weights = get_norm_weights(pf_state)
    goal_probs = zeros(Float64, n_goals)
    for (tr, w) in zip(traces, weights)
        goal_idx = tr[:init => :agent => :goal => :goal]
        goal_probs[goal_idx] += w
    end
    return goal_probs
end

"Print formatted goal probabilities on a single line."
function print_goal_probs(goal_probs, prefix="Goal Probs: ", newline=true)
    print(prefix)
    for p in goal_probs @printf("%.3f ", p) end
    if newline println() end
end

"Compute action noise probabilities from particle filter state."
function get_noise_probs(pf_state::ParticleFilterState)
    traces = get_traces(pf_state)
    weights = get_norm_weights(pf_state)
    noise_probs = Dict{Float64,Float64}()
    for (tr, w) in zip(traces, weights)
        act_noise = tr[:init => :agent => :act_args => :act_noise]
        noise_probs[act_noise] = get(noise_probs, act_noise, 0.0) + w
    end
    noise_support = sort(collect(keys(noise_probs)))
    noise_probs = [noise_probs[v] for v in noise_support]
    return noise_support, noise_probs
end

"Print formatted noise probabilities on a single line."
function print_noise_probs(noise_dist, prefix="Noise Probs: ", newline=true)
    print(prefix)
    noise_support, noise_probs = noise_dist
    for (v, p) in zip(noise_support, noise_probs)
        @printf("%.2f => %.2f | ", v, p)
    end
    if newline println() end
end

"Compute KL divergence between two PMFs as a surprise measure."
function kl_divergence(probs_1, probs_2)
    return sum(p1 * (log(p1) - log(p2)) for (p1, p2) in zip(probs_1, probs_2))
end

"Compute total variation between two PMFs as a surprise measure."
function total_variation(probs_1, probs_2)
    return 0.5 * sum(abs(p1-p2) for (p1, p2) in zip(probs_1, probs_2))
end

"Compute earth mover distance between two PMFs as a surprise measure."
function earth_mover_dist(probs_1, probs_2)
    @assert length(probs_1) == length(probs_2)
    n = length(probs_1)
    emds = zeros(n+1)
    for i in 1:n
        emds[i+1] = probs_1[i] + emds[i] - probs_2[i]
    end
    return sum(abs.(emds))
end
