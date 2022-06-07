## Conversion to PDDL representation ##
using Julog, PDDL

"Default grid scaling factor."
const SCALE_FACTOR = 2

"Converts the `(x, y)` centroid position of an object to a grid location."
function discretize_to_grid(origin::Symbol, position::Tuple{Real,Real},
                            grid_dims=(w=10, h=10), cell_dims=(w=20, h=20))
    return discretize_to_grid(origin, position[1], position[2],
                              grid_dims, cell_dims)
end

function discretize_to_grid(origin::Symbol, xpos::Real, ypos::Real,
                            grid_dims=(w=10, h=10), cell_dims=(w=20, h=20))
    if origin == :topleft
        # Subtract 0.5 cells and a pinch more to reduce discretization jitter
        xpos = round(Int, (xpos/cell_dims.w) - 0.55 + grid_dims.w/2)
        ypos = round(Int, (ypos/cell_dims.h) - 0.55 + grid_dims.h/2)
        return (xpos + 1, ypos + 1) # Convert from 0-indexing to 1-indexing
    elseif origin == :bottomleft
        xpos = round(Int, xpos / cell_dims.w)
        ypos = round(Int, (grid_dims.h * cell_dims.h - ypos) / cell_dims.h)
        return (xpos + 1, ypos) # Convert from 0-indexing to 1-indexing
    else
        error("Unrecognized origin: $origin")
    end
end

"Magnifies a grid by an integer scale factor."
function magnify_grid(grid::AbstractMatrix, scale::Int)
    new_dims = size(grid) .* scale
    new_grid = similar(grid, eltype(grid), new_dims)
    for j in 1:size(grid, 2), i in 1:size(grid, 1)
        ii = (i-1) * scale + 1
        jj = (j-1) * scale + 1
        new_grid[ii:(ii+scale-1),jj:(jj+scale-1)] .= grid[i,j]
    end
    return new_grid
end

"Magnify PDDL state by an integer scale factor."
function magnify_state(state::State, scale::Int)
    if scale == 1 return state end
    state = GenericState(state)
    # Magnify grids
    state[pddl"walls"] = magnify_grid(state[pddl"walls"], scale)
    state[pddl"barriers"] = magnify_grid(state[pddl"barriers"], scale)
    # Translate agent position
    state[pddl"xpos"] = scale * state[pddl"xpos"] - scale ÷ 2
    state[pddl"ypos"] = scale * state[pddl"ypos"] - scale ÷ 2
    # Translate objects
    for obj in PDDL.get_objects(state)
        x, y = get_obj_loc(state, obj)
        set_obj_loc!(state, obj, scale*x - scale÷2, scale*y - scale÷2)
    end
    # Make duplicate locks to fill the magnified wall grid
    start_loc = state[pddl"xpos"], state[pddl"ypos"]
    for lock in PDDL.get_objects(state, :lock)
        x, y = get_obj_loc(state, lock)
        x += scale ÷ 2
        y += scale ÷ 2
        count = 1
        for i in (y-scale+1):y, j in (x-scale+1):x # Create new locks
            new_lock = Const(Symbol(lock.name, "_$count"))
            push!(state.types, Compound(:lock, Const[new_lock]))
            set_obj_loc!(state, new_lock, (j, i))
            state[Compound(:locked, Const[new_lock])] = true
            count += 1
        end
        del_obj_loc!(state, lock) # Delete old lock
    end
    return state
end

"Construct PDDL state from individual components."
function build_pddl_state!(
    state::State, # Base state to modify
    walls::AbstractMatrix, # Ideally BitMatrix
    barriers::AbstractMatrix, # Same as above
    agent_loc::NTuple{2,Int}, # 2-tuple
    agent_id::Int,
    object_locs::Dict, # Symbols / strings to 2-tuples
    locks::Union{AbstractMatrix,Nothing}=nothing, # BitMatrix or nothing
    key_loc::Union{NTuple{2,Int},Nothing}=nothing # 2-tuple or nothing
)
    @debug "Agent (x, y) = $agent_loc"
    objects = PDDL.get_objects(state)
    # Set value of walls and removable barries
    state[pddl"walls"] = BitMatrix(walls)
    state[pddl"barriers"] = BitMatrix(barriers)
    # Set agent position and identity
    state[pddl"xpos"] = agent_loc[1]
    state[pddl"ypos"] = agent_loc[2]
    state[pddl"agentid"] = agent_id
    # Set lock locations
    if locks !== nothing
        locks = BitMatrix(locks)
        n_locks = sum(locks)
        is_locked = any(barriers)
        for (i, loc) in enumerate(findall(locks))
            y, x = Tuple(loc)
            obj = n_locks == 1 ? pddl"lock1" : Const(Symbol("lock1_$i"))
            set_obj_loc!(state, obj, x, y) # Set object location
        end
        for lock in PDDL.get_objects(state, :lock)
            state[Compound(:locked, [lock])] = is_locked # Set whether locked
        end
    end
    # Set key position and whether key is being held
    if pddl"key1" in objects
        if key_loc !== nothing
            set_obj_loc!(state, :key1, key_loc)
            state[pddl"(has key1)"] = key_loc == agent_loc
        else
            set_obj_loc!(state, :key1, (-1, -1))
        end
    end
    # Set locations of other objects
    objnames = sort(collect(keys(object_locs)))
    for (i, name) in enumerate(objnames)
        obj_loc = object_locs[name]
        obj = Const(Symbol("shape", i))
        set_obj_loc!(state, obj, obj_loc)
    end
    return state
end

"Construct PDDL state from individual components."
function build_pddl_state(state::State, args...)
    return build_pddl_state!(copy(state), args...)
end

function build_pddl_state(objtypes::Dict, args...)
    state = maybe_compiled_initstate(objtypes)
    return build_pddl_state!(state, args...)
end

## JSON loading and conversion code ##

using JSON3

"Object representation in JSON environment state."
const JSONObjectTuple = Tuple{NTuple{2,Int}, Int64, String, NTuple{3,Int}}

"Wall representation in JSON environment state."
const JSONWallTuple = Tuple{NTuple{2,Int},NTuple{2,Int}}

"JSON environment representation (for a single frame / state)."
mutable struct JSONState
    size::NTuple{2,Int} # Width and height in pixels
    walls::Vector{JSONWallTuple} # Wall locations and extents
    fuse_walls::Vector{JSONWallTuple} # Barrier wall locations and extents
    pin::Vector{JSONObjectTuple} # Key location
    key::Vector{JSONObjectTuple} # Lock location (strangely called key)
    objects::Vector{JSONObjectTuple} # Object locations
    home::JSONObjectTuple # Home location
    agent::JSONObjectTuple # Agent location
    JSONState() = new()
end

JSON3.StructTypes.StructType(::Type{JSONState}) = JSON3.StructTypes.Mutable()

"Sequence of JSON frames, representing a trial."
const JSONTrial = Vector{JSONState}

"Sequence of JSON trials, representing a scene / episode."
const JSONScene = Vector{JSONTrial}

"Context for conversion of current JSON state."
mutable struct JSONConversionContext
    init_state::Union{State,Nothing}
    object_ids::Dict{String,Int}
    agent_ids::Dict{String,Int}
end

JSONConversionContext() =
    JSONConversionContext(nothing, Dict(), Dict())

JSONConversionContext(init_state) =
    JSONConversionContext(init_state, Dict(), Dict())

"Loads a JSON scene fram path."
function load_json_scene(path::AbstractString)
    return open(path, "r") do io
        JSON3.read(io, JSONScene)
    end
end

"Infer object types from JSON state."
function infer_object_types(json::JSONState; scale::Int=SCALE_FACTOR)
    objtypes = Dict{Const,Symbol}()
    # Add (non-key, non-lock) objects
    for i in 1:length(json.objects)
        objtypes[Const(Symbol("shape", i))] = :shape
    end
    # Add keys
    if isdefined(json, :pin)
        for i in 1:length(json.pin)
            objtypes[Const(Symbol("key", i))] = :key
        end
    end
    # Add locks
    if isdefined(json, :key)
        n_copies = scale * scale
        for i in 1:length(json.key), j in 1:n_copies
            objtypes[Const(Symbol("lock", i, "_", j))] = :lock
        end
    end
    return objtypes
end

"Convert JSON state to PDDL state, modifying context as appropriate."
function json_to_pddl!(context::JSONConversionContext, json::JSONState;
                       scale::Int=SCALE_FACTOR, cell_dims=(w=20, h=20))
    # Compute grid dimensions
    width, height = json.size
    grid_dims = (w=round(Int, width/cell_dims.w), h=round(Int, height/cell_dims.w))
    # Construct walls
    walls = falses(grid_dims...)
    for (pos, extent) in json.walls
        x, y = discretize_to_grid(:bottomleft, pos, grid_dims, cell_dims)
        walls[y, x] = true
    end
    walls = magnify_grid(walls, scale)
    # Construct barriers
    barriers = falses(grid_dims...)
    for (pos, extent) in json.fuse_walls
        x, y = discretize_to_grid(:bottomleft, pos, grid_dims, cell_dims)
        barriers[y, x] = true
    end
    barriers = magnify_grid(barriers, scale)
    # Set lock locations
    locks = falses(grid_dims...)
    lock_tuples = isdefined(json, :key) ? json.key : []
    for (pos, size) in json.key
        x, y = discretize_to_grid(:bottomleft, pos, grid_dims, cell_dims)
        locks[y, x] = true
    end
    locks = magnify_grid(locks, scale)
    # Change grid dims and cell dims based on scale factor
    grid_dims = (w=grid_dims.w*scale, h=grid_dims.h*scale)
    cell_dims = (w=cell_dims.w/scale, h=cell_dims.h/scale)
    # Set key location
    key_loc = nothing
    key_tuples = isdefined(json, :pin) ? json.pin : []
    if !isempty(key_tuples)
        pos, size, image_src = json.pin[1]
        x, y = pos[1] + size, pos[2] + size
        key_loc = discretize_to_grid(:bottomleft, x, y, grid_dims, cell_dims)
    end
    # Set object locations
    object_locs = Dict{Symbol,NTuple{2,Int}}()
    for (pos, size, image_src) in json.objects
        # Discretize grid location
        x, y = pos[1] + size, pos[2] + size
        object_loc = discretize_to_grid(:bottomleft, x, y, grid_dims, cell_dims)
        # Get existing object ID or construct new ID
        object_id = get!(context.object_ids, image_src) do
            length(context.object_ids) + 1
        end
        object_name = Symbol("shape", object_id)
        object_locs[object_name] = object_loc
    end
    # Set agent location and ID
    pos, size, image_src = json.agent
    x, y = pos[1] + size, pos[2] + size
    agent_loc = discretize_to_grid(:bottomleft, x, y, grid_dims, cell_dims)
    agent_id = get!(context.agent_ids, image_src) do
        length(context.agent_ids) + 1
    end
    # Construct and return PDDL state
    if context.init_state === nothing
        objtypes = infer_object_types(json; scale=scale)
        context.init_state = maybe_compiled_initstate(objtypes)
        init_state = context.init_state
    else
        init_state = copy(context.init_state)
    end
    state = build_pddl_state!(init_state, walls, barriers, agent_loc, agent_id,
                              object_locs, locks, key_loc)
    return state
end

"Convert JSON state to PDDL state."
function json_to_pddl(json::JSONState; kwargs...)
    context = JSONConversionContext()
    return json_to_pddl!(context, json; kwargs...)
end

"Convert JSON state trajectory / trial to PDDL state."
function json_to_pddl!(context::JSONConversionContext,
                       json_trial::JSONTrial; kwargs...)
    context.init_state = nothing
    if isempty(json_trial) return State[] end
    state = json_to_pddl!(context, json_trial[1]; kwargs...)
    trajectory = [state]
    for json in json_trial[2:end]
        next_state =  json_to_pddl!(context, json; kwargs...)
        if next_state == state continue end
        state = next_state
        push!(trajectory, state)
    end
    return trajectory
end

"Convert JSON state trajectory / trial to PDDL state."
function json_to_pddl(json_trial::JSONTrial; kwargs...)
    context = JSONConversionContext()
    return json_to_pddl!(context, json_trial; kwargs...)
end

"Convert JSON scene / episode to sequence of PDDL state trajectories."
function json_to_pddl(json_scene::JSONScene; kwargs...)
    if isempty(json_scene) return [] end
    context = JSONConversionContext()
    trajs = []
    for trial in json_scene
        next_traj = json_to_pddl!(context, trial; kwargs...)
        push!(trajs, next_traj)
    end
    return trajs
end
