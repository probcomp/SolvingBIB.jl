
"Filter a trajectory with zig-zag motion to smooth diagonal motion."
function diagonalize_traj(traj::AbstractVector{<:State})
    state = first(traj)
    domain = lookup_compiled_domain(state)
    new_traj = [state]
    prev_x, prev_y = state[pddl"xpos"], state[pddl"ypos"]
    prev_dx, prev_dy = 0, 0
    prev_diag = false
    for state in traj[2:end]
        cur_x, cur_y = state[pddl"xpos"], state[pddl"ypos"]
        dx, dy = (cur_x - prev_x), (cur_y - prev_y)
        cur_diag = dx != 0 && dy != 0
        # If last two actions combined are diagonal, filter out previous state
        if (!cur_diag && !prev_diag && (prev_dx+dx) != 0 && (prev_dy+dy) != 0 &&
            is_consistent(domain, new_traj[end-1], state))
            pop!(new_traj)
            prev_state = new_traj[end]
            prev_x, prev_y = prev_state[pddl"xpos"], prev_state[pddl"ypos"]
            dx, dy = (cur_x - prev_x), (cur_y - prev_y)
            cur_diag = dx != 0 && dy != 0
        end
        push!(new_traj, state)
        prev_x, prev_y = cur_x, cur_y
        prev_dx, prev_dy = dx, dy
        prev_diag = cur_diag
    end
    return new_traj
end

"Filter out steps where the barriers are disappearing."
function skip_barrier_removal(traj::AbstractVector{<:State})
    state = first(traj)
    # If barriers and locks are present, do nothing
    if !any(state[pddl"barriers"]) return traj end
    started, stopped = false, false
    new_traj = [state]
    prev_state = state
    for (i, state) in enumerate(traj[2:end])
        if !started && !stopped # Check if barrier removal has started
            if (sum(state[pddl"barriers"]) < sum(prev_state[pddl"barriers"]))
                started = true
                @debug "Started: $(i+1)"
            end
        elseif started && !stopped # Check if barrier removal has stopped
            if !any(state[pddl"barriers"])
                stopped = true
                @debug "Stopped: $(i+1)"
                # Ensure agent does not have key
                for key in PDDL.get_objects(state, :key)
                    state[Compound(:has, Term[key])] = false
                end
                # Ensure agent does not move between previous and current state
                prev_x, prev_y = prev_state[pddl"xpos"], prev_state[pddl"ypos"]
                if state[pddl"xpos"] != prev_x || state[pddl"ypos"] != prev_y
                    @debug "Adding state for consistency when unlocking."
                    tmp_state = copy(state)
                    tmp_state[pddl"xpos"] = prev_x
                    tmp_state[pddl"ypos"] = prev_y
                    push!(new_traj, tmp_state)
                end
            end
        end
        if !started || stopped push!(new_traj, state) end
        prev_state = state
    end
    return new_traj
end

"Denoise the motion of any observed keys."
function denoise_key_motion(traj::AbstractVector{<:State}, key=Const(:key1))
    state = first(traj)
    if key ∉ PDDL.get_objects(state) return traj end
    got_key, used_key = false, false
    init_key_loc = get_obj_loc(state, key)
    new_traj = [state]
    prev_state = state
    for state in traj[2:end]
        skip = false
        x, y = state[pddl"xpos"], state[pddl"ypos"]
        if !got_key && !used_key # Check if key has been picked up
            if (state[Compound(:has, [key])] ||
                get_obj_loc(state, key) != init_key_loc)
                state = copy(state)
                # Ensure agent holds key and that their positions are equal
                got_key = true
                state[Compound(:has, [key])] = true
                set_obj_loc!(state, key, x, y)
                # Ensure agent does not move between previous and current state
                prev_x, prev_y = prev_state[pddl"xpos"], prev_state[pddl"ypos"]
                if x != prev_x || y != prev_y
                    @debug "Adding state for consistency when picking up key."
                    prev_state = copy(prev_state)
                    prev_state[pddl"xpos"] = x
                    prev_state[pddl"ypos"] = y
                    prev_state[Compound(:has, [key])] = false
                    push!(new_traj, prev_state)
                end
            end
        elseif got_key && !used_key # Check if key has been used
            if (key ∉ PDDL.get_objects(state) ||
                get_obj_loc(state, :key1) in get_obj_locs(state, :lock))
                used_key = true
            else
                # Ensure agent holds key and that their positions are equal
                state = copy(state)
                state[Compound(:has, [key])] = true
                set_obj_loc!(state, key, x, y)
                # Skip if location is the same as previous state
                if x == prev_state[pddl"xpos"] && y == prev_state[pddl"ypos"]
                    @debug "Skipping extraneous state due to key motion noise."
                    skip = true
                end
            end
        end
        if !skip push!(new_traj, state) end
        prev_state = state
    end
    return new_traj
end

"Repair a trajectory so that it is consistent with the domain theory."
function repair_traj(traj::AbstractVector{<:State})
    state = first(traj)
    domain = DOMAIN
    new_traj = [state]
    planner = AStarPlanner(heuristic=GoalCountHeuristic())
    for next_state in traj[2:end]
        # Check that next state is consistent with a successor
        consistent = false
        for act in available(domain, state)
            succ = execute(domain, state, act, check=false)
            consistent = (succ == next_state ||
                all(succ[f] == v for (f, v) in PDDL.get_fluents(next_state)))
            if consistent
                next_state = succ
                break
            end
        end
        # Continue to next step if consistent
        if consistent
            push!(new_traj, next_state)
            state = next_state
            continue
        end
        # Otherwise, find a plan to the next state
        add = collect(setdiff(next_state.facts, state.facts))
        del = collect(setdiff(state.facts, next_state.facts))
        next_x, next_y = next_state[pddl"xpos"], next_state[pddl"ypos"]
        pos_goals = parse_pddl("(== xpos $next_x)", "(== ypos $next_y)")
        goals = [add; [Compound(:not, [d]) for d in del]; pos_goals]
        _, repair_traj = planner(domain, state, goals)
        if repair_traj === nothing
            @warn "Could not repair by finding partial plan."
            push!(new_traj, next_state)
            state = next_state
            continue
        end
        append!(new_traj, repair_traj[2:end])
        state = repair_traj[end]
    end
    return new_traj
end
