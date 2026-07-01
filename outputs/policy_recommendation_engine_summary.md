# BNL AMU Policy Recommendation Engine Summary

## Executive Summary

The BNL AMU Policy Recommendation Engine is a configuration-driven reporting and recommendation layer that turns existing weekly pipeline outputs into user, tier, and contract governance guidance. It does not perform mathematical optimization. Instead, it applies deterministic rules from YAML configuration files to the current historical, operational, forecast, risk, and scenario outputs.

The engine runs from the project root, reads processed CSV inputs from the current pipeline, and writes a small set of recommendation files into `outputs/`. The latest implementation was refactored into an object-oriented package so the workflow is easier to maintain and extend. It also adds a new credit impact summary that shows the expected aggregate credit effect of the recommended user tier changes.

## Project Workflow

1. Weekly usage and processing jobs produce the source CSVs used by the recommendation layer.
2. The recommendation engine reads the available outputs, including user history, cap pressure summaries, forecast output, Monte Carlo risk output, and policy scenario output.
3. The engine applies configuration-driven scoring and tier transition rules.
4. It writes four main recommendation CSVs plus a new credit impact summary into `outputs/`.
5. Reviewers can use those files to assess user tier changes, tier cap alignment, and contract sizing implications.

In the larger BNL AMU credit management process, this engine sits after the operational and forecasting steps. It translates raw usage and risk signals into governance actions that are easier for IT and leadership to review.

## What Codex Refactored

Codex split the original monolithic script into a small package under `src/policy_recommendation/` and kept `src/policy_recommendation_engine.py` as a thin entry point.

- `PolicyRecommendationEngine` loads configuration, loads inputs, runs each recommender, writes outputs, and assembles summaries.
- `UserPolicyRecommender` scores users, selects `MOVE_UP`, `MOVE_DOWN`, `MAINTAIN`, or `REVIEW`, and maps the current tier to a recommended tier.
- `TierPolicyRecommender` evaluates whether each tier cap appears too low, too high, or aligned.
- `ContractPolicyRecommender` evaluates contract sizing using forecast and Monte Carlo outputs, plus scenario comparison logic.

This split keeps the code easier to test and makes future separation into dedicated modules straightforward.

## Inputs And Why They Matter

The engine uses the following inputs when they are available:

- `historical_user_segments.csv`: Provides longer-term user usage patterns that help distinguish consistently heavy users from lighter users.
- `cap_pressure_history_user_summary.csv`: Provides per-user cap pressure history and is the main driver for user-level recommendations.
- `tier_recommendations.csv`: Provides tier-level guidance already produced earlier in the pipeline and can be used as a supporting input.
- `forecast_summary.csv`: Provides the deterministic contract forecast, including expected end-of-contract balance.
- `monte_carlo_summary.csv`: Provides contract risk signals such as exhaustion and stranding probability.
- `policy_scenario_summary.csv`: Provides alternative contract and cap scenarios that help compare policy options.
- `tier_policy_config.yaml`: Defines the authoritative tier list and weekly credit caps.
- `policy_recommendation_config.yaml`: Defines the tunable recommendation thresholds, scoring weights, confidence rules, reason codes, and output column structure.

How those inputs are used:

- User recommendations combine cap pressure, utilization history, observed stability, spike behavior, and historical segmentation.
- Tier recommendations use tier-level pressure and the ordered tier structure to judge whether a cap should move up, down, or stay aligned.
- Contract recommendations use forecast balance, Monte Carlo risk, and scenario comparison to suggest a contract size direction.
- Confidence scoring uses the same configurable score thresholds across the recommendation types.
- Net credit impact is derived from the tier change effect on users whose recommended tier differs from their current tier.

## Tier Recommendation Math

The tier logic is rule-based and configuration-driven. It uses the ordered tier list from `tier_policy_config.yaml` and the recommendation thresholds in `policy_recommendation_config.yaml`.

At a high level, the engine looks at:

- Usage history, especially average utilization and the share of weeks above the high-utilization threshold.
- Cap pressure, which signals whether users are regularly close to or beyond their caps.
- Stability, which helps avoid moving users too quickly when history is limited.
- Trend behavior, which can support a move when usage is consistently high or consistently low.
- Spike behavior, which can trigger `REVIEW` when recent pressure is unusual or unstable.
- Tier ordering, which determines the next higher or lower tier for movement.

The action logic works like this:

- `MOVE_UP`: The user shows sustained pressure or heavy usage signals, and the engine maps them to the next higher tier.
- `MOVE_DOWN`: The user shows low utilization or excess capacity signals, and the engine maps them to the next lower tier.
- `MAINTAIN`: The current tier is still appropriate, so no tier change is recommended.
- `REVIEW`: The signals are ambiguous, unstable, or boundary-limited, so the user should be reviewed manually.

Boundary handling is explicit. If a user is already at the highest tier, the engine cannot move them higher. If a user is already at the lowest tier, the engine cannot move them lower. In those cases, the engine falls back to `REVIEW` and adds boundary reason codes so the limitation is visible.

This is not a true optimizer. It is a deterministic scoring and threshold system designed to be easy to understand and replace later if a more advanced optimizer is introduced.

## Confidence Logic

The confidence value is a simple reviewer aid, not a strict statistical probability of correctness unless the implementation is later changed to make it one.

The engine assigns confidence by comparing the raw recommendation score with a configured maximum score and then mapping that ratio into labels such as `HIGH`, `MEDIUM`, and `LOW`.

In general:

- Confidence increases when multiple signals point to the same action.
- Confidence increases when the signal is strong and unambiguous.
- Confidence decreases when the user has limited history, conflicting signals, or boundary constraints.
- `REVIEW` is often used when the score is lower or the signals are mixed.

Reviewers should treat confidence as a prioritization guide. High confidence means the rule set saw a strong and consistent signal. Low confidence means the recommendation is weaker, more ambiguous, or more dependent on manual judgment.

## Net Credit Impact

Net credit impact shows how the recommended tier changes are expected to affect total credit capacity.

The sign convention is explicit:

- Positive net credit impact means more credits consumed or more capacity allocated.
- Negative net credit impact means credits saved or less capacity allocated.

For each user whose recommended tier differs from the current tier, the engine compares the current tier cap with the recommended tier cap. That per-user difference is then aggregated across all changed users to produce the net credit impact summary.

This is useful for two reasons:

- Governance: it shows whether the recommendation set expands or contracts overall capacity.
- Contract planning: it gives leadership a quick way to see whether user tier changes are likely to increase or reduce total credit demand before contract decisions are finalized.

## What The Outputs Mean

The engine writes these main files into `outputs/`:

- `policy_recommendation_user_recommendations.csv`: User-level action rows with the recommended action, target tier, confidence, reason codes, and estimated credit impact.
- `policy_recommendation_tier_recommendations.csv`: Tier-level cap alignment guidance showing whether caps appear too low, too high, or aligned.
- `policy_recommendation_contract_recommendations.csv`: Contract-level sizing guidance based on forecast and Monte Carlo signals.
- `policy_recommendation_summary.csv`: One-row run summary with high-level counts, contract recommendation text, net credit impact, and main risk themes.
- `policy_recommendation_credit_impact_summary.csv`: Aggregated credit impact view for recommended user tier changes.

## Credit Impact Summary File

`policy_recommendation_credit_impact_summary.csv` explains the aggregate effect of user tier changes.

- `current_total_estimated_credit_impact`: The total current-tier cap value for users whose recommended tier differs from their current tier.
- `recommended_total_estimated_credit_impact`: The total cap value if those users moved to their recommended tiers.
- `net_credit_impact`: The difference between recommended total and current total. Positive means more credits; negative means fewer credits.
- `num_users_recommended_up`: The number of users whose recommended tier is higher than their current tier.
- `num_users_recommended_down`: The number of users whose recommended tier is lower than their current tier.
- `num_users_reviewed`: The number of users assigned `REVIEW`.
- `notes`: Plain-English explanation of how the summary was calculated.

The summary is intentionally compact so it can be dropped into a slide deck or memo without further formatting.

## Operational Notes

- The engine is configuration-driven and runs from the project root.
- It reads the current pipeline artifacts if they exist, and it handles missing optional files with warnings rather than crashing.
- It remains deterministic: the same inputs and YAML settings produce the same outputs.
- It was refactored into smaller modules so future work can split user, tier, and contract logic further without changing the external workflow.

## Current Run Signals

In the latest run, the engine produced real tier movements and a negative net credit impact, which indicates the recommended set of user tier changes would reduce overall allocated capacity. The exact counts and values are captured in the generated CSV outputs.
