using SolvingBIB, PDDL, Plinf, Random, Printf
using Statistics, StatsFuns, DataFrames, CSV, Optim

SCENARIOS = [
    (:efficiency, :irrational, "efficiency_irrational"),
    (:efficiency, :path, "efficiency_path"),
    (:efficiency, :time, "efficiency_time"),
    (:preference, :none, "preference"),
    (:inaccessible, :none, "inaccessible_goal"),
    (:multiagent, :none, "multi_agent"),
    (:instrumental, :inconsequential, "instrumental_inconsequential_barrier"),
    (:instrumental, :no_barrier, "instrumental_no_barrier"),
    (:instrumental, :blocking, "instrumental_blocking_barrier"),
]

DATASET_DIR = normpath(joinpath(@__DIR__, "../datasets/bib_evaluation_v1.1"))

df = DataFrame(
    subtask=Symbol[],
    scenario=Symbol[],
    scene=String[],
    plausible=Bool[],
    rating=Float64[],
    goal_tv=Float64[],
    noise_tv=Float64[],
    noise_emd=Float64[],
    log_likelihood=Float64[],
    ll_diff=Float64[]
)
df_types = eltype.(eachcol(df))


for (subtask, scenario, subdir) in SCENARIOS
    row = Dict{Symbol,Any}(:subtask => subtask, :scenario => scenario)
    println("== Subtask: $subtask ==")
    println("-- Scenario: $scenario --")
    # Load and filter JSON paths
    subdir = joinpath(DATASET_DIR, subdir)
    filenames = readdir(subdir)
    filter!(filenames) do p
        match(r".+\.json", p) !== nothing
    end
    # Iterate over files
    for scene in filenames
        # Set scene name
        row[:scene] = scene
        # Set ground truth plausibility based on filename
        row[:plausible] = scene[end-5] == 'e' # e for expected, u for unexpected
        println(row[:plausible] ?
                "- $scene (plausible) -" : "- $scene (implausible) -")
        # Run inference on scene
        scene_path = joinpath(subdir, scene)
        rating, features = solve_scene(scene_path)
        # Display and store rating
        row[:rating] = rating
        @printf("Rating: %.3f\n", rating)
        # Display and store decision features
        display(features)
        merge!(row, features)
        # Add results to dataframe
        push!(df, row)
    end
end

# Save results and decision features to file
path = joinpath(@__DIR__, "results.csv")
# CSV.write(path, df)
df = CSV.read(path, DataFrame, types=df_types)
# path = joinpath(@__DIR__, "backup_results.csv")

# Compute overall accuracy and pairwise accuracy
y_prob = df[!, :rating]
y_pred = df[!, :rating] .>= 0.5
y_true = df[!, :plausible]
y_reldiff = y_prob[1:2:end] .- y_prob[2:2:end]
accuracy = mean(y_pred .== y_true)
pairwise_accuracy = mean(y_reldiff .> 0)
println()
@printf("Accuracy (overall): %.3f\t", accuracy)
@printf("Pairwise Accuracy (overall): %.3f\t", pairwise_accuracy)

# Compute accuracies for each subtask
subtasks_df = groupby(df, :subtask)
for key in keys(subtasks_df)
    subtask = key[:subtask]
    y_prob = subtasks_df[key][!, :rating]
    y_pred = subtasks_df[key][!, :rating] .>= 0.5
    y_true = subtasks_df[key][!, :plausible]
    y_reldiff = y_prob[1:2:end] .- y_prob[2:2:end]
    accuracy = mean(y_pred .== y_true)
    pairwise_accuracy = mean(y_reldiff .> 0)
    println()
    @printf("Accuracy (%s): %.3f\t", subtask, accuracy)
    @printf("Pairwise Accuracy (%s): %.3f\t", subtask, pairwise_accuracy)
end

# Compute accuracies for each scenario
scenarios_df = groupby(df, [:subtask, :scenario])
for key in keys(scenarios_df)
    subtask = key[:subtask]
    scenario = key[:scenario]
    y_prob = scenarios_df[key][!, :rating]
    y_pred = scenarios_df[key][!, :rating] .>= 0.5
    y_true = scenarios_df[key][!, :plausible]
    y_reldiff = y_prob[1:2:end] .- y_prob[2:2:end]
    accuracy = mean(y_pred .== y_true)
    pairwise_accuracy = mean(y_reldiff .> 0)
    println()
    @printf("Accuracy (%s - %s): %.3f \n", subtask, scenario, accuracy)
    @printf("Pairwise Accuracy (%s - %s): %.3f", subtask, scenario, pairwise_accuracy)
end
