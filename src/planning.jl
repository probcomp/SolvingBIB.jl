const PDDL_PATH = joinpath(dirname(pathof(SolvingBIB)), "../pddl")

const DOMAIN = load_domain(joinpath(PDDL_PATH, "domain.pddl"))

const ACTION_COSTS =
    (pickup=1.0, unlock=1.0, north=1.0, south=1.0, east=1.0, west=1.0,
     northeast=sqrt(2), northwest=sqrt(2), southwest=sqrt(2), southeast=sqrt(2))

const COMPILED_DOMAINS = Dict{Set{Const},Domain}()

## Heuristics for planning or policy initialization ##

"Manhattan distance heuristic to location of goal."
struct GoalManhattan <: Heuristic end

function Plinf.compute(heuristic::GoalManhattan,
                       domain::Domain, state::State, spec::Specification)
    goal = Plinf.get_goal_terms(spec)[1]
    if goal.name == :has # Pick up a certain object
        obj = goal.args[1]
        cur_loc = [state[pddl"xpos"], state[pddl"ypos"]]
        goal_loc = [state[Compound(:xloc, Term[obj])],
                    state[Compound(:yloc, Term[obj])]]
        return sum(abs.(cur_loc - goal_loc))
    else # Default to goal count heuristic
        return GoalCountHeuristic()(domain, state, spec)
    end
end

"Euclidean distance heuristic to location of goal."
struct GoalEuclidean <: Heuristic end

function Plinf.compute(heuristic::GoalEuclidean,
                       domain::Domain, state::State, spec::Specification)
    goal = Plinf.get_goal_terms(spec)[1]
    if goal.name == :has # Pick up a certain object
        obj = goal.args[1]
        cur_loc = [state[pddl"xpos"], state[pddl"ypos"]]
        goal_loc = [state[Compound(:xloc, Term[obj])],
                    state[Compound(:yloc, Term[obj])]]
        return sqrt(sum(abs2.(cur_loc - goal_loc)))
    else # Default to goal count heuristic
        return GoalCountHeuristic()(domain, state, spec)
    end
end

"Manhattan distance heuristic to the nearest of multiple goal rewards."
struct MultiGoalManhattan <: Heuristic end

function Plinf.compute(heuristic::MultiGoalManhattan,
                       domain::Domain, state::State, spec::MultiGoalReward)
    # Find Manhattan distance to each goal, and substract goal reward
    costs = map(zip(spec.goals, spec.rewards)) do (goal, reward)
        dist = GoalManhattan()(domain, state, goal)
        return dist - reward
    end
    # Return minimum cost
    return min(costs)
end

# Unwrap extra action costs
function Plinf.compute(heuristic::MultiGoalManhattan,
                       domain::Domain, state::State, spec::ExtraActionCosts)
    return Plinf.compute(GoalManhattan(), domain, state, spec.spec)
end

# Default to GoalManhattan for singular goal specifications
function Plinf.compute(heuristic::MultiGoalManhattan,
                       domain::Domain, state::State, spec::Specification)
    return Plinf.compute(GoalManhattan(), domain, state, spec)
end

# Planner-in-the-loop to be used in heuristics
const ASTAR_PLANNER = AStarPlanner(GoalEuclidean())

## Utility functions for working with PDDL ##

"Lookup compiled domain based on object signature."
function lookup_compiled_domain(objects)
    # Default to uncompiled domain if not found
    return get(COMPILED_DOMAINS, Set(collect(objects))) do
        @warn "Compiled domain not found for objects: $objects"
        return DOMAIN
    end
end

function lookup_compiled_domain(state::State)
    objects = PDDL.get_objects(state)
    return lookup_compiled_domain(objects)
end

"Constructs initial state from object types, using compiled domain if possible."
function maybe_compiled_initstate(objtypes::Dict)
    domain = lookup_compiled_domain(keys(objtypes))
    if domain isa CompiledDomain
        state = PDDL.statetype(domain)()
    else
        state = initstate(DOMAIN, objtypes, [])
    end
    return state
end

"Return list of object locations."
function get_obj_locs(state::State, type::Symbol=:object)
    return [(state[Compound(:xloc, Term[o])], state[Compound(:yloc, Term[o])])
             for o in PDDL.get_objects(state, type)]
end

"Get location of object as an (x, y) tuple."
function get_obj_loc(state::State, obj::Const)
    return (state[Compound(:xloc, Term[obj])],
            state[Compound(:yloc, Term[obj])])
end
get_obj_loc(state::State, obj::Symbol) = get_obj_loc(state, Const(obj))

"Sets location of object in state."
function set_obj_loc!(state::State, obj::Const, x::Int, y::Int)
    state[Compound(:xloc, Term[obj])] = x
    state[Compound(:yloc, Term[obj])] = y
    return state
end
set_obj_loc!(state::State, obj::Const, loc::Tuple) =
    set_obj_loc!(state, obj, loc[1], loc[2])
set_obj_loc!(state::State, obj::Symbol, x::Int, y::Int) =
    set_obj_loc!(state, Const(obj), x, y)
set_obj_loc!(state::State, obj::Symbol, loc::Tuple) =
    set_obj_loc!(state, Const(obj), loc[1], loc[2])

"Delete object from (generic) state."
function del_obj_loc!(state::GenericState, obj::Const)
    objtype = PDDL.get_objtypes(state)[obj]
    delete!(state.types, Compound(objtype, Const[obj]))
    delete!(state.values[:xloc], (obj.name,))
    delete!(state.values[:yloc], (obj.name,))
    delete!(state.facts, Compound(:locked, Const[obj]))
    delete!(state.facts, Compound(:has, Const[obj]))
    return state
end
del_obj_loc!(state::GenericState, obj::Symbol) =
    del_obj_loc!(state, Const(obj))

"Convert goal object to goal specification."
function goal_object_spec(goal_obj::Const)
    goal = Compound(:has, Term[goal_obj])
    return MinActionCosts(Term[goal], ACTION_COSTS)
end

"Check if two consecutive states are consistent with the domain."
function is_consistent(domain::Domain, s::State, sp::State)
    actions = collect(available(domain, s))
    return any(transition(domain, s, a, check=false) == sp for a in actions)
end

is_consistent(s::State, sp::State) =
    is_consistent(lookup_compiled_domain(s), s, sp)
