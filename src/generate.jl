# Code for generating synthetic datasets

"Generate random wall maze as a bit array."
function random_wall_maze(;dims=(10, 10))
    walls = falses(dims)
    # Fill in outer walls
    walls[1,:] .= true
    walls[end,:] .= true
    walls[:,1] .= true
    walls[:,end] .= true
    # Randomly generate inner walls
    n_walls = 15
    for i in 1:n_walls
        wall_length = neg_binom(2, 0.5)
        if wall_length == 0 continue end
        x, y = rand(1:dims[2]), rand(1:dims[1])
        walls[y, x] = true
        for j in 1:(wall_length - 1)
            xdir, ydir = rand([(-1, 0), (1, 0), (0, -1), (0, 1)])
            xtmp, ytmp = x + xdir, y + ydir
            if checkbounds(Bool, walls, ytmp, xtmp) && !walls[ytmp, xtmp]
                x, y = xtmp, ytmp
                walls[y, x] = true
            end
        end
    end
    return walls
end

"Generate a random free location within an occupancy grid."
function random_free_loc(grid::AbstractMatrix, occupied=[])
    if all(grid .== true) error("No free locations in grid.") end
    height, width = size(grid)
    while true
        x, y = rand(1:width), rand(1:height)
        if grid[y, x] || (x, y) in occupied continue end
        return x, y
    end
end

function random_free_loc(state::State)
    objlocs = [(state[parse_pddl("(xloc $o)")], state[parse_pddl("(yloc $o)")])
               for o in PDDL.get_objects(state)]
    grid = state[pddl"walls"] .| state[pddl"barriers"]
    return random_free_loc(grid, objlocs)
end

"Search in a spiral until a nearby reachable location is found."
function nearby_reachable_loc(loc, start, grid::AbstractMatrix, occupied=[])
    x, y = loc
    seg_len, seg_count, dir_x, dir_y = 1, 0, 1, 0
    while (!checkbounds(Bool, grid, y, x) || grid[y, x] ||
           (x, y) in occupied || !check_path(grid, start, (x, y)))
        x += dir_x
        y += dir_y
        seg_count += 1
        if seg_count == seg_len
            seg_len += dir_y != 0 ? 1 : 0
            dir_x, dir_y = -dir_y, dir_x
            seg_count = 0
        end
    end
    return (x, y)
end

function nearby_reachable_loc(loc, state::State)
    objlocs = [(state[parse_pddl("(xloc $o)")], state[parse_pddl("(yloc $o)")])
               for o in PDDL.get_objects(state)]
    start = state[pddl"xpos"], state[pddl"ypos"]
    grid = state[pddl"walls"] .| state[pddl"barriers"]
    return nearby_reachable_loc(loc, start, grid, objlocs)
end

"Check if an unobstructed path exists between two grid locations."
function check_path(grid::AbstractMatrix, start, stop)
    # Perform DFS to check if a path exists
    stack = [start]
    visited = Set([start])
    while !isempty(stack)
        loc = pop!(stack)
        x, y = loc
        if loc == stop return true end
        for (nx, ny) in ((x-1, y), (x+1, y), (x, y-1), (x, y+1))
            if !checkbounds(Bool, grid, ny, nx) || grid[ny, nx] continue end
            if (nx, ny) in visited continue end
            push!(stack, (nx, ny))
            push!(visited, (nx, ny))
        end
    end
    return false
end

"Enclose grid location(s) with walls."
function enclose_locs!(grid::AbstractMatrix, locs...)
    height, width = size(grid)
    xs, ys = first.(locs), last.(locs)
    min_x, max_x = max(minimum(xs)-1, 1), min(maximum(xs)+1, width)
    min_y, max_y = max(minimum(ys)-1, 1), min(maximum(ys)+1, height)
    grid[min_y,min_x:max_x] .= true
    grid[max_y,min_x:max_x] .= true
    grid[min_y:max_y,min_x] .= true
    grid[min_y:max_y,max_x] .= true
    return grid
end

"Add random instrumental features to a PDDL state."
function add_instrumental_features(state::State; goal_loc=nothing,
                                   has_barrier=true, need_key=true,
                                   max_tries=100)
    if goal_loc === nothing
        goal_loc = state[pddl"(xloc shape1)"], state[pddl"(yloc shape1)"]
    end
    n_tries = 0
    walls = state[pddl"walls"]
    dims = size(walls)
    start_loc = state[pddl"xpos"], state[pddl"ypos"]
    objlocs = [(state[parse_pddl("(xloc $o)")], state[parse_pddl("(yloc $o)")])
               for o in PDDL.get_objects(state)]
    if has_barrier
        # Generate key with barrier
        key_loc, lock_loc, barriers = nothing, nothing, nothing
        while n_tries < max_tries
            n_tries += 1
            # Generate random key location
            key_loc = random_free_loc(walls, objlocs)
            if need_key # Generate barriers enclosing start or goal
                barriers = enclose_locs!(falses(dims), start_loc, key_loc)
            else # Generate inconsequential barrier
                loc1, loc2 = random_free_loc(walls), random_free_loc(walls)
                barriers = enclose_locs!(falses(dims), loc1, loc2)
            end
            grid = walls .| barriers
            # Generate random lock location
            lock_loc = rand(findall(barriers))
            # Check that key is reachable
            if grid[key_loc[2], key_loc[1]] continue end
            if !check_path(grid, start_loc, key_loc) continue end
            # Check that goal is (un)reachable with barrier
            if grid[goal_loc[2], goal_loc[1]] continue end
            if check_path(grid, start_loc, goal_loc) == need_key continue end
            # Check that lock isn't on a wall location and is reachable
            grid[lock_loc] = false | walls[lock_loc]
            lock_loc = (lock_loc[2], lock_loc[1])
            if !check_path(grid, start_loc, lock_loc) continue end
            break
        end
        if n_tries >= max_tries
            @debug "Could not add instrumental features after $max_tries tries."
            return nothing
        end
        # Construct modified PDDL state
        objtypes = PDDL.get_objtypes(state)
        objtypes[pddl"key1"] = :key
        objtypes[pddl"lock1"] = :lock
        fluents = Dict(PDDL.get_fluents(state))
        new_state = initstate(DOMAIN, objtypes, fluents)
        new_state[pddl"barriers"] = barriers
        new_state[pddl"(locked lock1)"] = true
        set_obj_loc!(new_state, pddl"key1", key_loc)
        set_obj_loc!(new_state, pddl"lock1", lock_loc)
    else
        # Generate key without barrier
        key_loc = random_free_loc(walls, objlocs)
        while n_tries < max_tries && !check_path(walls, start_loc, key_loc)
            key_loc = random_free_loc(walls, objlocs)
            n_tries += 1
        end
        if n_tries >= max_tries
            @debug "Could not add instrumental features after $max_tries tries."
            return nothing
        end
        # Construct modified PDDL state
        objtypes = PDDL.get_objtypes(state)
        objtypes[pddl"key1"] = :key
        objtypes[pddl"lock1"] = :lock
        fluents = Dict(PDDL.get_fluents(state))
        new_state = initstate(DOMAIN, objtypes, fluents)
        new_state[pddl"barriers"] .= false
        set_obj_loc!(new_state, pddl"key1", key_loc)
        set_obj_loc!(new_state, pddl"lock1", (0, 0))
    end
    return new_state
end

"Generate random state in the SolvingBIB domain."
function random_agency_state(; scale=1, kwargs...)
    while true
        state = _random_agency_state(;kwargs...)
        state !== nothing && return magnify_state(state, scale)
    end
end

function _random_agency_state(;
    walls=nothing, dims=(10, 10), max_tries=100,
    n_objects=1, reachable=fill(true, n_objects), agent_id=1,
    instrumental=false, has_barrier=instrumental, need_key=instrumental
)
    # Generate wall maze
    walls = walls === nothing ? random_wall_maze(dims=dims) : BitMatrix(walls)
    dims = size(walls)
    # Generate unreachable object locations
    occupied = Tuple{Int,Int}[]
    objlocs = fill((0, 0), n_objects)
    for i in 1:n_objects
        if reachable[i] continue end
        loc = random_free_loc(walls, occupied)
        enclose_locs!(walls, loc) # Enclose object location
        objlocs[i] = loc
        push!(occupied, loc)
    end
    # Generate start location
    start = random_free_loc(walls)
    occupied = [start]
    # Generate reachable object locations
    n_tries = 0
    for i in 1:n_objects
        if !reachable[i] continue end
        loc = random_free_loc(walls, occupied)
        # Check that object is reachable from start location
        while n_tries < max_tries && !check_path(walls, start, loc)
            loc = random_free_loc(walls, occupied)
            n_tries += 1
        end
        # Give up if we can't generate accessible object locations
        if n_tries >= max_tries
            @debug "Could not generate reachable objects after $n_tries tries."
            return nothing
        end
        objlocs[i] = loc
        push!(occupied, loc)
    end
    # Construct PDDL state
    objtypes = Dict{Const,Symbol}()
    fluents = Dict{Term,Any}(pddl"walls" => walls,
                             pddl"barriers" => falses(dims),
                             pddl"agentid" => agent_id,
                             pddl"xpos" => start[1], pddl"ypos" => start[2])
    for (i, (x, y)) in enumerate(objlocs)
        obj = Const(Symbol("shape$i"))
        objtypes[obj] = :shape
        fluents[Compound(:xloc, Term[obj])] = x
        fluents[Compound(:yloc, Term[obj])] = y
    end
    state = initstate(DOMAIN, objtypes, fluents)
    if instrumental
        state = add_instrumental_features(state;
            has_barrier=has_barrier,
            need_key=need_key,
            max_tries=max_tries
        )
    end
    return state
end

"Generate gridworld trajectories demonstrating agent efficiency."
function generate_efficiency_trajs(
    n::Int; efficiencies=[fill(true, n-1); false], scale=SCALE_FACTOR
)
    # Generate base state with fixed wall structure
    base_state = random_agency_state(n_objects=1)
    walls = base_state[pddl"walls"]
    # Construct optimal planner
    astar = AStarPlanner(heuristic=GoalEuclidean())
    # Generate trajectories for each efficiency flag
    plans_and_trajs = map(efficiencies) do efficient
        # Generate random state using base wall structure
        state = random_agency_state(n_objects=1, walls=walls, scale=scale)
        # Construct a direct route to the goal
        plan, traj = astar(DOMAIN, state, pddl"(has shape1)")
        # Construct longer detour route if inefficient
        detour_ratio = 2.0
        while !efficient
            # Generate reachable detour location
            x, y = nearby_reachable_loc(random_free_loc(state), state)
            loc_goal = parse_pddl("(and (= xpos $x) (= ypos $y))")
            # Plan from start to detour location
            plan1, traj1 = astar(DOMAIN, state, loc_goal)
            # Plan from detour location to goal
            plan2, traj2 = astar(DOMAIN, traj1[end], pddl"(has shape1)")
            # Return detoured plan if long enough
            if length(plan1) + length(plan2) >= detour_ratio * length(plan)
                plan = [plan1; plan2]
                traj = [traj1; traj2[2:end]]
                break
            end
            detour_ratio *= 0.9
        end
        return plan, traj
    end
    plans, trajs = unzip(plans_and_trajs)
    return plans, trajs
end

"Generated paired trajectories where the test trial differs in efficiency."
function generate_paired_efficiency_trajs(
    n::Int, scenario::Symbol=:efficient; scale=SCALE_FACTOR
)
    # Select between efficient or inefficient habituation trials
    if scenario == :efficient
        efficiencies = [fill(true, n); false]
        plans, trajs = generate_efficiency_trajs(n+1, scale=scale,
                                                 efficiencies=efficiencies)
        expected = plans[1:n], trajs[1:n]
        unexpected = plans[[1:n-1;n+1]], trajs[[1:n-1;n+1]]
    else
        test_plan, test_traj = generate_efficiency_trajs(1, efficiencies=[false])
        plans, trajs = generate_efficiency_trajs(n-1, efficiencies=fill(false, n-1))
        expected = append!(plans, test_plan), append!(trajs, test_traj)
        plans, trajs = generate_efficiency_trajs(n-1, efficiencies=fill(true, n-1))
        unexpected = append!(plans, test_plan), append!(trajs, test_traj)
    end
    return (expected, unexpected)
end

"Generate gridworld trajectories demonstrating object preference."
function generate_preference_trajs(
    n::Int;
    goals=[fill(pddl"(has shape1)", n-1); pddl"(has shape2)"],
    scale=SCALE_FACTOR
)
    # Generate base state with fixed wall structure
    base_state = random_agency_state(n_objects=2)
    walls = base_state[pddl"walls"]
    # Construct optimal planner
    astar = AStarPlanner(heuristic=GoalEuclidean())
    # Generate trajectory to each preferred object
    plans_and_trajs = map(goals) do goal
        # Generate state with 2 reachable objects using base wall structure
        state = random_agency_state(n_objects=2, walls=walls, scale=scale)
        # Construct optimal plan to goal object
        plan, traj = astar(DOMAIN, state, goal)
        return plan, traj
    end
    plans, trajs = unzip(plans_and_trajs)
    return plans, trajs
end

"Generated paired trajectories where the test trial differs in preference."
function generate_paired_preference_trajs(
    n::Int;
    preferred=pddl"(has shape1)", dispreferred=pddl"(has shape2)",
    scale=SCALE_FACTOR
)
    goals = [fill(preferred, n); dispreferred]
    plans, trajs = generate_preference_trajs(n+1, goals=goals, scale=scale)
    expected = plans[1:n], trajs[1:n]
    unexpected = plans[[1:n-1;n+1]], trajs[[1:n-1;n+1]]
    return (expected, unexpected)
end

"Generate gridworld trajectories with inaccessible goals at specific trials."
function generate_inaccessible_trajs(
    n::Int;
    goals=[fill(pddl"(has shape1)", n-1); pddl"(has shape2)"],
    accessibilities=[fill(true, n-1); false], scale=SCALE_FACTOR
)
    # Generate base state with fixed wall structure
    base_state = random_agency_state(n_objects=2)
    walls = base_state[pddl"walls"]
    # Construct optimal planner
    astar = AStarPlanner(heuristic=GoalEuclidean())
    # Generate trajectory to first goal if accessible, second otherwise
    plans_and_trajs = map(zip(goals, accessibilities)) do (goal, accessible)
        # Generate state according to object accesibility
        reachable = accessible ? [true, true] : [false, true]
        state = random_agency_state(n_objects=2, reachable=reachable,
                                    walls=walls, scale=scale)
        # Construct optimal plan to goal
        plan, traj = astar(DOMAIN, state, goal)
        return plan, traj
    end
    plans, trajs = unzip(plans_and_trajs)
    return plans, trajs
end

"Generated paired trajectories where the test trial differs in accessibility."
function generate_paired_inaccessible_trajs(
    n::Int;
    preferred=pddl"(has shape1)", dispreferred=pddl"(has shape2)",
    scale=SCALE_FACTOR
)
    goals = [fill(preferred, n-1); fill(dispreferred, 2)]
    accessibilities = [fill(true, n-1); false; true]
    plans, trajs = generate_inaccessible_trajs(n+1, goals=goals, scale=scale,
                                               accessibilities=accessibilities)
    expected = plans[1:n], trajs[1:n]
    unexpected = plans[[1:n-1;n+1]], trajs[[1:n-1;n+1]]
    return (expected, unexpected)
end

"Generate gridworld trajectories with different agent identities"
function generate_multiagent_trajs(
    n::Int;
    goals=[fill(pddl"(has shape1)", n-1); pddl"(has shape2)"],
    agent_ids=[fill(1, n-1); 2], scale=SCALE_FACTOR
)
    # Generate base state with fixed wall structure
    base_state = random_agency_state(n_objects=2)
    walls = base_state[pddl"walls"]
    # Construct optimal planner
    astar = AStarPlanner(heuristic=GoalEuclidean())
    # Generate trajectory to each preferred object
    plans_and_trajs = map(zip(goals, agent_ids)) do (goal, agent_id)
        # Generate state with 2 reachable objects using base wall structure
        state = random_agency_state(n_objects=2, walls=walls,
                                    agent_id=agent_id, scale=scale)
        # Construct optimal plan to goal object
        plan, traj = astar(DOMAIN, state, goal)
        return plan, traj
    end
    plans, trajs = unzip(plans_and_trajs)
    return plans, trajs
end

"Generated paired trajectories where the test trial differs in agent identity."
function generate_paired_multiagent_trajs(
    n::Int, scenario::Symbol=:goalswitch;
    agent_id_1=1, agent_id_2=2, scale=SCALE_FACTOR,
    preferred=pddl"(has shape1)", dispreferred=pddl"(has shape2)"
)
    # Select between switching or not switching the goal object in test trials
    if scenario == :goalswitch
        goals = [fill(preferred, n-1); fill(dispreferred, 2)]
        agent_ids =  [fill(agent_id_1, n-1); agent_id_2; agent_id_1]
    else
        goals = fill(preferred, n+1)
        agent_ids = [fill(agent_id_1, n); agent_id_2]
    end
    plans, trajs = generate_multiagent_trajs(n+1, goals=goals, scale=scale,
                                             agent_ids=agent_ids)
    expected = plans[1:n], trajs[1:n]
    unexpected = plans[[1:n-1;n+1]], trajs[[1:n-1;n+1]]
    return (expected, unexpected)
end

"Generate gridworld trajectories with instrumental actions."
function generate_instrumental_trajs(
    n::Int; goals=[fill(pddl"(has shape1)", n-1); pddl"(has key1)"],
    scale=SCALE_FACTOR, has_barriers=[fill(true, n-1); false],
    need_keys=[fill(true, n-1); false], use_keys=[fill(true, n-1); false]
)
    # Generate base state with fixed wall structure
    base_state = random_agency_state(instrumental=true)
    walls = base_state[pddl"walls"]
    # Construct optimal planner
    astar = AStarPlanner(heuristic=GoalEuclidean())
    # Generate trajectory for each set of options
    options = zip(goals, has_barriers, need_keys, use_keys)
    plans_and_trajs = map(options) do (goal, has_barrier, need_key, use_key)
        # Generate state with instrumental features
        while true
            state = random_agency_state(walls=walls, instrumental=true,
                                        scale=scale, has_barrier=has_barrier,
                                        need_key=need_key)
            if goal != pddl"(has key1)" && !need_key && use_key
                # Get key along the way to goal, even when unnecessary
                unlock_all = pddl"(forall (?l - lock) (not (locked ?l)))"
                plan1, traj1 = astar(DOMAIN, state, unlock_all)
                if plan1 === nothing continue end
                plan2, traj2 = astar(DOMAIN, traj1[end], goal)
                if plan2 === nothing continue end
                plan = [plan1; plan2]
                traj = [traj1; traj2[2:end]]
            else # Construct optimal plan to goal object
                plan, traj = astar(DOMAIN, state, goal)
            end
            if plan === nothing
                continue
            else
                return plan, traj
            end
        end
    end
    plans, trajs = unzip(plans_and_trajs)
    return plans, trajs
end

"Generated paired trajectories where the test trial differs in instrumentality."
function generate_paired_instrumental_trajs(
    n::Int, scenario::Symbol=:nobarrier; scale=SCALE_FACTOR
)
    if scenario == :nobarrier # No barrier in test trials
        has_barriers = [fill(true, n-1); fill(false, 2)]
        need_keys = [fill(true, n-1); fill(false, 2)]
        use_keys = [fill(true, n-1); fill(false, 2)]
        goals = [fill(pddl"(has shape1)", n); pddl"(has key1)"]
    elseif scenario == :redundant # Redundant barrier in test trials
        has_barriers = fill(true, n+1)
        need_keys = [fill(true, n-1); fill(false, 2)]
        use_keys = [fill(true, n-1); fill(false, 2)]
        goals = [fill(pddl"(has shape1)", n); pddl"(has key1)"]
    elseif scenario == :movebarrier # Barrier changes in test trials
        has_barriers = fill(true, n+1)
        need_keys = [fill(true, n); false]
        use_keys = fill(true, n+1)
        goals = fill(pddl"(has shape1)", n+1)
    else
        error("Unrecognized scenario.")
    end
    plans, trajs = generate_instrumental_trajs(n+1, goals=goals, scale=scale,
                                               has_barriers=has_barriers,
                                               need_keys=need_keys,
                                               use_keys=use_keys)
    expected = plans[1:n], trajs[1:n]
    unexpected = plans[[1:n-1;n+1]], trajs[[1:n-1;n+1]]
    return (expected, unexpected)
end

"Generate paired trajectories for the specified subtask and scenario."
function generate_paired_trajs(
    n::Int, subtask::Symbol, scenario::Symbol=:none; options...
)
    return if subtask == :preference
        generate_paired_preference_trajs(n; options...)
    elseif subtask == :efficiency
        generate_paired_efficiency_trajs(n, scenario; options...)
    elseif subtask == :inaccessible
        generate_paired_inaccessible_trajs(n; options...)
    elseif subtask == :multiagent
        generate_paired_multiagent_trajs(n, scenario; options...)
    elseif subtask == :instrumental
        generate_paired_instrumental_trajs(n, scenario; options...)
    end
end

export generate_preference_trajs, generate_paired_preference_trajs,
       generate_efficiency_trajs, generate_paired_efficiency_trajs,
       generate_inaccessible_trajs, generate_paired_inaccessible_trajs,
       generate_multiagent_trajs, generate_paired_multiagent_trajs,
       generate_instrumental_trajs, generate_paired_instrumental_trajs,
       generate_paired_trajs
