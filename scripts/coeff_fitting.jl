## Generate decision-level features and fit decision coefficients accordingly

using SolvingBIB, PDDL, Plinf, Random, Printf
using Statistics, StatsFuns, DataFrames, CSV, Optim

df = DataFrame(
    subtask=Symbol[],
    scenario=Symbol[],
    plausible=Bool[],
    rating=Float64[],
    goal_tv=Float64[],
    noise_tv=Float64[],
    noise_emd=Float64[],
    log_likelihood=Float64[],
    ll_diff=Float64[]
)

scenarios = [
    (:efficiency, :efficient),
    (:efficiency, :inefficient),
    (:preference, :none),
    (:preference, :none),
    (:inaccessible, :none),
    (:inaccessible, :none),
    (:multiagent, :goalswitch),
    (:multiagent, :nogoalswitch),
    (:instrumental, :nobarrier),
    (:instrumental, :redundant),
    (:instrumental, :movebarrier),
]

n_scenes_per_scenario = 10

# Generate decision-level features for each subtask-scenario pair
Random.seed!(0)
for (subtask, scenario) in scenarios
    row = Dict{Symbol,Any}(:subtask => subtask, :scenario => scenario)
    println("== Subtask: $subtask ==")
    println("-- Scenario: $scenario --")
    n_goals = subtask == :efficiency ? 1 : 2
    for i in 1:n_scenes_per_scenario
        println("Generating decisions features for scene pair $i...")
        (_, expected), (_, unexpected) =
            generate_paired_trajs(8, subtask, scenario, scale=2)
        # Add decision features for plausible test trial
        println("- Plausible -")
        _, metadata = scene_inference(expected, n_goals)
        rating, features = decide_scene(metadata[end])
        row[:plausible] = true
        row[:rating] = rating
        display(features)
        merge!(row, features)
        push!(df, row)
        # Add decision features for implausible test trial
        println("- Implausible -")
        _, metadata = scene_inference(unexpected, n_goals)
        rating, features = decide_scene(metadata[end])
        row[:plausible] = false
        row[:rating] = rating
        display(features)
        merge!(row, features)
        push!(df, row)
    end
end

# Save decision features to file
path = joinpath(dirname(pathof(SolvingBIB)), "..", "scripts/decision_features.csv")
CSV.write(path, df)
df = CSV.read(path, DataFrame)

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

"Predicts plausibility from features via a product of logistic regressions."
function predict(features::Matrix, coeffs::Matrix)
    intercepts = coeffs[1, :]
    slopes = coeffs[2, :]
    logits = features .* slopes' .+ intercepts'
    probs = StatsFuns.logistic.(logits)
    ratings = vec(prod(probs; dims=2))
end

predict(features::Matrix, coeffs::Vector) =
    predict(features, reshape(coeffs, 2, :))

# Compute predictions and accuracy prior to optimization
y_true = df[!, :plausible]
features = Matrix(df[!, [:goal_tv, :noise_tv]])
init_coeffs = [10.0, -20.0, 6.0, -9.0]

# features = Matrix(df[!, [:goal_tv, :noise_tv]])
# init_coeffs = [6.0, -10.0, 6.0, -10.0]

ratings = predict(features, init_coeffs)
accuracy = mean((ratings .>= 0.5) .== y_true)

# Optimize coefficients by minimizing cross-entropy loss
lambda = 0.0001 # Regularization strength
"Cross-entropy objective function."
function objective(coeffs::Vector)
    y_pred = predict(features, coeffs)
    scores = @. -y_true*log(y_pred) - (1-y_true)*log(1-y_pred)
    penalty = lambda * sum(abs2.(coeffs))
    loss = mean(scores) + penalty
end
results = optimize(objective, init_coeffs, LBFGS(); autodiff=:forward)
opt_coeffs = Optim.minimizer(results)
init_cross_entropy = objective(init_coeffs)
cross_entropy = objective(opt_coeffs)

# Display new coefficients and thresholds
println("Coefficients:")
for col in eachcol(reshape(opt_coeffs, 2, :))
    intercept = col[1]
    slope = col[2]
    thresh = -intercept/slope
    @printf("Threshold: %.3f\t Intercept: %.3f\t Slope: %.3f\n",
            thresh, intercept, slope)
end

# Compute optimized predictions
ratings = predict(features, opt_coeffs)
accuracy = mean((ratings .>= 0.5) .== y_true)
y_reldiff = ratings[1:2:end] .- ratings[2:2:end]
pairwise_accuracy = mean(y_reldiff .> 0)
