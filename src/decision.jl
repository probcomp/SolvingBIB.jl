using StatsFuns
using StatsFuns: logistic, logsumexp

"Logistic regression coefficients for step-wise plausibility ratings."
const STEP_DECISION_COEFFS = Dict(
    :goal_tv => (3.0, -10.0),
    :noise_tv => (3.0, -10.0),
    :noise_emd => (Inf, 0.0),
    :log_likelihood => (Inf, 0.0),
)

"Logistic regression coefficients for scene plausibility ratings."
const SCENE_DECISION_COEFFS = Dict(
    :goal_tv => (5.96, -11.81),
    :noise_tv => (5.90, -9.42),
    :noise_emd => (Inf, 0.0),
    :log_likelihood => (Inf, 0.0),
    :ll_diff => (2.0, -0.2)
)

"Decide plausibility rating for a single inference timestep."
function decide_step(metadata::Dict; coeffs::Dict=STEP_DECISION_COEFFS)
    # Decide plausibility using a product of per-feature logistic regressions
    rating = 1.0
    for (key, (intercept, slope)) in coeffs
        logits = intercept + slope * metadata[key]
        rating *= logistic(logits)
    end
    return rating
end

"Decide a scene's plausibility given inference metadata for the test trial."
function decide_scene(metadata; coeffs::Dict=SCENE_DECISION_COEFFS)
    if isempty(metadata) error("Input is empty.") end
    # Dictionary of scene-level decision features
    features = Dict{Symbol,Float64}()
    # Compute scene-level total variation in goal and noise posteriors
    features[:goal_tv] = maximum(metadata) do m
        total_variation(metadata[1][:goal_probs], m[:goal_probs])
    end
    features[:noise_tv] = maximum(metadata) do m
        total_variation(metadata[1][:noise_probs], m[:noise_probs])
    end
    # Compute scene-level earth mover distance in noise posterior
    features[:noise_emd] = maximum(metadata) do m
        earth_mover_dist(metadata[1][:noise_probs], m[:noise_probs])
    end
    # Compute mean log likelihood across entire scene
    lls = getindex.(metadata, :log_likelihood)
    features[:log_likelihood] = mean(lls)
    # Compute max difference in log likelihood from initial state
    features[:ll_diff] = maximum(lls[1] .- lls)
    # Decide plausibility using a product of per-feature logistic regressions
    rating = 1.0
    for (key, (intercept, slope)) in coeffs
        logits = intercept + slope * features[key]
        rating *= logistic(logits)
    end
    # Return rating and decision features
    return rating, features
end
