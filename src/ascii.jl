"Converts ASCII gridworlds to PDDL problem."
function ascii_to_pddl_problem(str::String, name="bib-gridworld-problem")
    rows = split(str, "\n", keepempty=false)
    width, height = maximum(length.(strip.(rows))), length(rows)
    locks, keys, shapes = Const[], Const[], Const[]
    walls = parse_pddl("(= walls (new-bit-matrix false $width $height))")
    barriers = parse_pddl("(= barriers (new-bit-matrix false $width $height))")
    init = Term[walls, barriers]
    start, goal = parse_pddl("(= xpos -1)", "(= ypos -1)"), pddl"(true)"
    agent_id = pddl"(= agentid -1)"
    for (y, row) in enumerate(rows)
        for (x, char) in enumerate(strip(row))
            if char == '.' # Unoccupied
                continue
            elseif char == 'W' # Wall
                wall = parse_pddl("(= walls (set-index walls true $y $x))")
                push!(init, wall)
            elseif char == 'B' # Barrier
                barrier = parse_pddl("(= barriers (set-index barriers true $y $x))")
                push!(init, barrier)
            elseif char == 'L' # Lock
                barrier = parse_pddl("(= barriers (set-index barriers true $y $x))")
                push!(init, barrier)
                l = Const(Symbol("lock$(length(locks)+1)"))
                push!(locks, l)
                push!(init, parse_pddl("(locked $l)"))
                append!(init, parse_pddl("(= (xloc $l) $x)", "(= (yloc $l) $y)"))
            elseif char == 'k' # Key
                k = Const(Symbol("key$(length(keys)+1)"))
                push!(keys, k)
                append!(init, parse_pddl("(= (xloc $k) $x)", "(= (yloc $k) $y)"))
            elseif char == 's' # Shape
                s = Const(Symbol("shape$(length(shapes)+1)"))
                push!(shapes, s)
                append!(init, parse_pddl("(= (xloc $s) $x)", "(= (yloc $s) $y)"))
            elseif char in "0123456789" # Agent
                agent_id = parse_pddl("(= agentid $char)")
                start = parse_pddl("(= xpos $x)", "(= ypos $y)")
            end
        end
    end
    append!(init, start)
    push!(init, agent_id)
    objtypes = merge(Dict(l => :lock for l in locks),
                     Dict(k => :key for k in keys),
                     Dict(s => :shape for s in shapes))
    problem = GenericProblem(Symbol(name), Symbol("bib-gridworld"),
                             [locks; keys; shapes],
                             objtypes, init, goal, nothing)
    return problem
end

"Converts ASCII string to PDDL state."
function ascii_to_pddl_state(str::String)
    problem = ascii_to_pddl_problem(str)
    return initstate(DOMAIN, problem)
end

"Load ASCII from path, and convert to PDDL state."
function load_ascii_state(path::String)
    str = open(f->read(f, String), path)
    return ascii_to_pddl_state(str)
end

"Load ASCII from path, and convert to PDDL problem."
function load_ascii_problem(path::String)
    str = open(f->read(f, String), path)
    return ascii_to_pddl_problem(str)
end

"Print PDDL state as ASCII grid."
function print_ascii(io::IO, state::State)
    walls = state[pddl"walls"]
    barriers = state[pddl"barriers"]
    height, width = size(walls)
    shapes = PDDL.get_objects(state, :shape)
    shape_locs = get_obj_locs(state, :shape)
    key_locs = get_obj_locs(state, :key)
    lock_locs = get_obj_locs(state, :lock)
    for y in 1:height
        for x in 1:width
            if state[pddl"xpos"] == x && state[pddl"ypos"] == y # Agent
                print(io, "A")
            elseif (x, y) in shape_locs # Goal objects
                idx = findfirst(loc -> loc == (x, y), shape_locs)
                char = string(shapes[idx].name)[end]
                print(io, char)
            elseif (x, y) in key_locs # Keys
                print(io, 'k')
            elseif (x, y) in lock_locs # Locks
                print(io, 'L')
            elseif walls[y, x] # Walls
                print(io, 'W')
            elseif barriers[y, x] # Barriers
                print(io, 'B')
            else # Empty cells
                print(io, '.')
            end
        end
        println(io)
    end
end

print_ascii(state::State) = print_ascii(stdout, state)

"Print perception outptus as ASCII grid."
function print_ascii(
    io::IO,
    walls::AbstractMatrix, # Ideally BitMatrix
    barriers::AbstractMatrix, # Same as above
    agent_loc::NTuple{2,Int}, # 2-tuple
    agent_id::Int,
    object_locs::Dict, # Symbols / strings to 2-tuples
    locks::Union{AbstractMatrix,Nothing}=nothing, # BitMatrix or nothing
    key_loc::Union{NTuple{2,Int},Nothing}=nothing # 2-tuple or nothing
)
    height, width = size(walls)
    object_ids = sort(collect(keys(object_locs)))
    object_locs = [object_locs[id] for id in object_ids]
    for y in 1:height
        for x in 1:width
            if (x, y) == agent_loc # Agent
                print(io, "A")
            elseif (x, y) in object_locs # Goal objects
                idx = findfirst(loc -> loc == (x, y), object_locs)
                print(io, idx)
            elseif (x, y) == key_loc # Keys
                print(io, 'k')
            elseif locks !== nothing && locks[y, x] # Locks
                print(io, 'L')
            elseif walls[y, x] # Walls
                print(io, 'W')
            elseif barriers[y, x] # Barriers
                print(io, 'B')
            else # Empty cells
                print(io, '.')
            end
        end
        println(io)
    end
end

function print_ascii(
    walls::AbstractMatrix, # Ideally BitMatrix
    barriers::AbstractMatrix, # Same as above
    agent_loc::NTuple{2,Int}, # 2-tuple
    agent_id::Int,
    object_locs::Dict, # Symbols / strings to 2-tuples
    locks::Union{AbstractMatrix,Nothing}=nothing, # BitMatrix or nothing
    key_loc::Union{NTuple{2,Int},Nothing}=nothing # 2-tuple or nothing
)
    return print_ascii_grid(stdout, walls, barriers, agent_loc, agent_id,
                            object_locs, locks, key_loc)
end
