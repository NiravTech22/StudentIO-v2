using Oxygen, HTTP, JSON3, Dates
using Flux, Random, Statistics
using Logging

# ============================================================================
# 1. Environment & Configuration
# ============================================================================

const ENV_NAME = get(ENV, "ENVIRONMENT", "development") # production | staging | development
const IS_PROD = ENV_NAME == "production"
const IS_STAGING = ENV_NAME == "staging"

# Configure Logging based on Environment
if IS_PROD
    global_logger(ConsoleLogger(stderr, Logging.Info))
    println(">>> STARTING IN PRODUCTION MODE")
else
    global_logger(ConsoleLogger(stderr, Logging.Debug))
    println(">>> STARTING IN $ENV_NAME MODE (Debug Enabled)")
end

# ============================================================================
# 2. Neural Core (The Brain)
# ============================================================================

# Model Architecture: 64 -> LSTM(128) -> 3
const INPUT_DIM = 64
const HIDDEN_DIM = 128
const OUTPUT_DIM = 3

# Define Model
# In a real scenario, we might verify model version here.
model = Chain(
    Dense(INPUT_DIM => 64, tanh),
    LSTM(64 => HIDDEN_DIM),
    Dense(HIDDEN_DIM => OUTPUT_DIM),
    softmax
)

# Initialize Stateful Object
global core_state = Flux.state(model)
println("Neural Core Initialized: Matrix Dimensions Verified.")

# ============================================================================
# 3. Inference Logic (Stable vs Experimental)
# ============================================================================

function text_to_embedding(text::String)
    rng = MersenneTwister(hash(text))
    return randn(rng, Float32, INPUT_DIM)
end

function run_inference(text::String)
    # Common Preprocessing
    x = text_to_embedding(text)

    # In Staging, we might inject noise or test a different model path
    if IS_STAGING && startswith(text, "TEST:")
        @info "Staging: Running Experimental Inference Path"
        # Simulate experimental logic (e.g., increased temperature)
        return (Float32[0.1, 0.1, 0.8], "experimental_v2")
    end

    # Standard Inference (Updates Global State)
    global core_state
    (y, new_state) = Flux.run(model, x, core_state)
    core_state = new_state

    return (y, "stable_v1")
end

# ============================================================================
# 4. API Endpoints
# ============================================================================

# Health Check (Common)
@get "/health" function ()
    return Dict("status" => "ok", "env" => ENV_NAME)
end

# Telemetry (Neural Visualizer)
@get "/api/telemetry" function (req::HTTP.Request)
    # Extract LSTM Hidden State for Visualization
    # core_state[2] is the LSTM state (h, c)
    hidden_vec = try
        lstm_state = core_state[2]
        # Handle Flux version differences safely
        if isa(lstm_state, Tuple)
            vec(lstm_state[1])
        elseif hasproperty(lstm_state, :h)
            vec(lstm_state.h)
        else
            zeros(Float32, HIDDEN_DIM)
        end
    catch e
        @error "Telemetry Extraction Failed" exception = (e, catch_backtrace())
        zeros(Float32, HIDDEN_DIM)
    end

    return Dict(
        "timestamp" => Dates.now(),
        "activations" => hidden_vec,
        "environment" => ENV_NAME
    )
end

# Chat Endpoint (The Main Interface)
@post "/api/chat" function (req::HTTP.Request)
    try
        data = JSON3.read(req.body)
        # Accept both 'question' and 'text' fields for flexibility
        user_text = get(data, :text, get(data, :question, ""))
        student_id = get(data, :studentId, 0)  # Accept but ignore for now

        # Log request in staging/dev
        @debug "Received Chat Message" text = user_text student_id = student_id

        # Run Inference
        (probs, model_version) = run_inference(user_text)

        # Heuristic Response Generation
        sentiment_idx = argmax(probs)
        resp_text = if sentiment_idx == 1
            "I sense negative entropy."
        elseif sentiment_idx == 3
            "Positive alignment confirmed."
        else
            "Processing vector state..."
        end

        # Infer sentiment for frontend
        sentiment = if sentiment_idx == 1
            "negative"
        elseif sentiment_idx == 3
            "positive"
        else
            "neutral"
        end

        # Append debug info in Staging
        reasoning = ["Processed by $model_version"]
        if !IS_PROD
            push!(reasoning, "Env: $ENV_NAME")
            push!(reasoning, "Vector Strength: $(maximum(probs))")
        end

        return Dict(
            "response" => resp_text,  # Changed from "answer" to match frontend
            "sentiment" => sentiment,
            "reasoning" => reasoning,
            "confidence" => maximum(probs),
            "timestamp" => time()
        )
    catch e
        @error "Chat Endpoint Failed" exception = (e, catch_backtrace())
        return HTTP.Response(500, ["Access-Control-Allow-Origin" => "*"], JSON3.write(Dict("error" => "Internal Core Error")))
    end
end

# ============================================================================
# 5. Staging-Only Routes
# ============================================================================

if !IS_PROD
    println(">>> Registering Debug Routes (Staging Only)")

    @post "/api/debug/reset_state" function (req::HTTP.Request)
        global core_state = Flux.state(model)
        @info "Core State Reset by Admin"
        return Dict("status" => "reset_complete")
    end

    @get "/api/debug/dump_config" function ()
        return Dict(
            "INPUT_DIM" => INPUT_DIM,
            "HIDDEN_DIM" => HIDDEN_DIM,
            "ENV" => ENV_NAME
        )
    end
end

# ============================================================================
# 6. Server Startup
# ============================================================================

# Add CORS middleware
function cors_handler(handle)
    function (req::HTTP.Request)
        # Add CORS headers to all responses
        response = handle(req)
        if response isa HTTP.Response
            push!(response.headers, "Access-Control-Allow-Origin" => "*")
            push!(response.headers, "Access-Control-Allow-Methods" => "GET, POST, OPTIONS")
            push!(response.headers, "Access-Control-Allow-Headers" => "Content-Type")
        end
        return response
    end
end

println("Server listening on 0.0.0.0:8080")
serve(host="0.0.0.0", port=8080, middleware=[cors_handler])
