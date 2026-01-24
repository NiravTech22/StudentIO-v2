# ============================================================================
# StudentIO HTTP Visualization Server
# ============================================================================
#
# Provides REST API endpoints for real-time visualization of:
# - Latent state trajectories
# - Belief state evolution
# - Policy decisions and rationale
# - Diagnostic metrics
#
# ============================================================================

# Note: This is now part of StudentIO module directly
# using Oxygen, HTTP, JSON3  <-- Already used in parent module

# Global session storage (in production, use proper state management)
const ACTIVE_SESSIONS = Dict{String,StudentSession}()

# ============================================================================
# API Endpoints
# ============================================================================

# Health check
@get "/health" function ()
    Dict("status" => "ok", "module" => "StudentIO Visualization Server")
end

# Create a new learning session
@post "/api/session/create" function (req::HTTP.Request)
    try
        data = JSON3.read(req.body)
        session_id = get(data, :session_id, string("session_", time()))
        student_id = get(data, :student_id, "student_1")

        # Create model (use pre-trained if available)
        model = create_default_model()
        session = create_session(model)

        ACTIVE_SESSIONS[session_id] = session

        return Dict(
            "session_id" => session_id,
            "student_id" => student_id,
            "status" => "created",
            "config" => Dict(
                "state_dim" => model.config.state_dim,
                "mastery_dim" => model.config.mastery_dim,
                "misconception_dim" => model.config.misconception_dim,
                "abstraction_dim" => model.config.abstraction_dim,
                "belief_dim" => model.config.belief_dim
            )
        )
    catch e
        return HTTP.Response(500, [], JSON3.write(Dict("error" => string(e))))
    end
end

# Process a student interaction and get next action
@post "/api/session/:session_id/step" function (req::HTTP.Request, session_id::String)
    try
        if !haskey(ACTIVE_SESSIONS, session_id)
            return HTTP.Response(404, [], JSON3.write(Dict("error" => "Session not found")))
        end

        session = ACTIVE_SESSIONS[session_id]
        data = JSON3.read(req.body)

        # Create observation from user input
        observation = Dict(
            :correctness => get(data, :correctness, 0.5),
            :response_time => get(data, :response_time, 1.0),
            :confidence => get(data, :confidence, 0.5)
        )

        # Step the session
        action = step!(session, observation)

        # Get current belief state and uncertainty
        belief, uncertainty = get_belief_state(session)

        # Decode belief to estimated student state
        x_hat = decode_state(session.model.filter, belief.belief)

        # Decompose state into components
        components = decompose_state(session.model.transition, x_hat)

        # Extract gate activations for this step
        if session.step_count > 1
            last_obs = encode_observation(session.model.observation, observation)
            last_action_vec = if isnothing(session.last_action)
                zeros(Float32, session.model.config.action_dim)
            else
                encode_action(session.model.policy, session.last_action)
            end

            gates = extract_gate_activations(
                session.model.filter,
                belief.belief,
                last_obs,
                last_action_vec
            )
        else
            gates = nothing
        end

        return Dict(
            "step" => session.step_count,
            "action" => action,
            "state_estimate" => Dict(
                "mastery" => collect(components.mastery),
                "misconceptions" => collect(components.misconceptions),
                "abstractions" => collect(components.abstractions),
                "mean_mastery" => mean(components.mastery),
                "has_misconceptions" => any(components.misconceptions .> 0.3)
            ),
            "belief" => Dict(
                "hidden_state" => collect(belief.belief),
                "uncertainty" => uncertainty,
                "norm" => norm(belief.belief)
            ),
            "gates" => if !isnothing(gates)
                Dict(
                    "information_integration" => gates.information_integration,
                    "belief_correction" => gates.belief_correction
                )
            else
                nothing
            end
        )

    catch e
        @error "Step error" exception = (e, catch_backtrace())
        return HTTP.Response(500, [], JSON3.write(Dict("error" => string(e))))
    end
end

# Get full session history
@get "/api/session/:session_id/history" function (req::HTTP.Request, session_id::String)
    try
        if !haskey(ACTIVE_SESSIONS, session_id)
            return HTTP.Response(404, [], JSON3.write(Dict("error" => "Session not found")))
        end

        session = ACTIVE_SESSIONS[session_id]

        # Extract trajectories from history
        steps = []
        for entry in session.history
            belief_state = entry.belief_state

            # Decode to state estimate
            x_hat = decode_state(session.model.filter, belief_state)
            components = decompose_state(session.model.transition, x_hat)

            push!(steps, Dict(
                "step" => entry.step,
                "mastery" => collect(components.mastery),
                "misconceptions" => collect(components.misconceptions),
                "abstractions" => collect(components.abstractions),
                "uncertainty" => entry.uncertainty,
                "action" => entry.action,
                "rationale" => entry.rationale
            ))
        end

        return Dict(
            "session_id" => session_id,
            "total_steps" => length(steps),
            "trajectory" => steps
        )

    catch e
        @error "History error" exception = (e, catch_backtrace())
        return HTTP.Response(500, [], JSON3.write(Dict("error" => string(e))))
    end
end

# Reset session
@post "/api/session/:session_id/reset" function (req::HTTP.Request, session_id::String)
    try
        if !haskey(ACTIVE_SESSIONS, session_id)
            return HTTP.Response(404, [], JSON3.write(Dict("error" => "Session not found")))
        end

        reset!(ACTIVE_SESSIONS[session_id])

        return Dict("status" => "reset", "session_id" => session_id)
    catch e
        return HTTP.Response(500, [], JSON3.write(Dict("error" => string(e))))
    end
end

# Get diagnostics for session
@get "/api/session/:session_id/diagnostics" function (req::HTTP.Request, session_id::String)
    try
        if !haskey(ACTIVE_SESSIONS, session_id)
            return HTTP.Response(404, [], JSON3.write(Dict("error" => "Session not found")))
        end

        session = ACTIVE_SESSIONS[session_id]

        if length(session.history) < 2
            return Dict("error" => "Insufficient data for diagnostics")
        end

        # Extract belief trajectory
        belief_traj = hcat([entry.belief_state for entry in session.history]...)
        uncertainties = [entry.uncertainty for entry in session.history]

        # Compute diagnostics
        drift = compute_belief_drift(belief_traj)
        collapse = detect_uncertainty_collapse(uncertainties)

        return Dict(
            "belief_drift" => collect(drift),
            "uncertainties" => collect(uncertainties),
            "uncertainty_collapsed" => collapse,
            "mean_drift" => mean(drift),
            "mean_uncertainty" => mean(uncertainties)
        )

    catch e
        @error "Diagnostics error" exception = (e, catch_backtrace())
        return HTTP.Response(500, [], JSON3.write(Dict("error" => string(e))))
    end
end

# ============================================================================
# Server Startup
# ============================================================================

function start_server(; port::Int=8080, host::String="0.0.0.0")
    println("🚀 Starting StudentIO Visualization Server")
    println("📊 Real-time POMDP visualization endpoints active")
    println("🌐 Server: http://$host:$port")

    # Add CORS middleware
    function cors_middleware(handle)
        function (req::HTTP.Request)
            response = handle(req)
            if response isa HTTP.Response
                push!(response.headers, "Access-Control-Allow-Origin" => "*")
                push!(response.headers, "Access-Control-Allow-Methods" => "GET, POST, OPTIONS")
                push!(response.headers, "Access-Control-Allow-Headers" => "Content-Type")
            end
            return response
        end
    end

    serve(host=host, port=port, middleware=[cors_middleware])
end
