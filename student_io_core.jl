<<<<<<< Updated upstream
<<<<<<< Updated upstream
using Oxygen, HTTP, JSON3, Dates
using Flux

# Load the core module
try
    println("Loading StudentIO from: ", joinpath(@__DIR__, "src", "StudentIO.jl"))
    include(joinpath(@__DIR__, "src", "StudentIO.jl"))
catch e
    println("ERROR LOADING STUDENTIO MODULE:")
    showerror(stdout, e)
    println()
    rethrow(e)
end
using .StudentIO

# ============================================================================
# Global State
# ============================================================================

# Model instance (meta-learned parameters)
global model = create_default_model()
try
    global model = Flux.cpu(model) # Ensure on CPU for inference server
catch e
    println("Warning: Could not move model to CPU: $e")
end

# Active sessions: student_id -> Session
const sessions = Dict{String,Any}()
const session_lock = ReentrantLock()

println("StudentIO Model Initialized.")

# ============================================================================
# Validation / Helper Logic
# ============================================================================

function get_or_create_session(student_id::String)
    lock(session_lock) do
        if !haskey(sessions, student_id)
            sessions[student_id] = create_session(model)
            println("Created new session for student: $student_id")
        end
        return sessions[student_id]
=======
=======
>>>>>>> Stashed changes
using Oxygen, HTTP, JSON3, Flux, Random, Statistics, Dates

# ============================================================================
# Neural Core Configuration
# ============================================================================

# 1. Define the Model Architecture
#    Input: 64 dim (Text Embedding Hashed)
#    Recurrent: LSTM (64 -> 128)
#    Output: Dense (128 -> 3) [Sentiment Class: Negative, Neutral, Positive]
const INPUT_DIM = 64
const HIDDEN_DIM = 128
const OUTPUT_DIM = 3

model = Chain(
    Dense(INPUT_DIM => 64, tanh),
    LSTM(64 => HIDDEN_DIM),
    Dense(HIDDEN_DIM => OUTPUT_DIM),
    softmax
)

# 2. Initialize State
#    Flux models with recurrent layers are stateful. 
#    We maintain a global state for the single "Student" instance.
global core_state = Flux.state(model)

println("Neural Core Initialized: $model")

# ============================================================================
# Helper Functions
# ============================================================================

"""
    text_to_embedding(text::String, dim::Int)
    
Hashes the input text into a stable random vector of size `dim`.
This serves as a primitive "embedding" to feed the neural network 
without needing a massive loaded language model.
"""
function text_to_embedding(text::String, dim::Int)
    rng = MersenneTwister(hash(text))
    return randn(rng, Float32, dim)
end

"""
    get_heuristic_response_text(text::String, sentiment_idx::Int)
    
Generates a coherent English response based on input and model-predicted sentiment.
"""
function get_heuristic_response_text(text::String, sentiment_idx::Int)
    lower_text = lowercase(text)

    # Heuristics for text content
    if occursin("hello", lower_text) || occursin("hi", lower_text)
        return "Greetings. My neural pathways are active and ready."
    elseif occursin("what", lower_text) && occursin("vector", lower_text)
        return "A vector is a geometric object that has magnitude and direction. In my core, it represents a state of activation."
    elseif occursin("help", lower_text)
        return "I can assist you. Please state your query."
<<<<<<< Updated upstream
=======
    end

    # Fallback based on model sentiment
    # Index 1: Negative, 2: Neutral, 3: Positive
    if sentiment_idx == 1
        return "I detect uncertainty. Could you rephrase that?"
    elseif sentiment_idx == 3
        return "Input processed successfully. I am aligned with this concept."
    else
        return "Processing input. State updated."
>>>>>>> Stashed changes
    end

    # Fallback based on model sentiment
    # Index 1: Negative, 2: Neutral, 3: Positive
    if sentiment_idx == 1
        return "I detect uncertainty. Could you rephrase that?"
    elseif sentiment_idx == 3
        return "Input processed successfully. I am aligned with this concept."
    else
        return "Processing input. State updated."
>>>>>>> Stashed changes
    end
end

# Simple heuristic to map text input to an "Observation"
# In a real system, this would use an NLP encoder.
# Here we just mock it to test the loop.
function text_to_observation(text::String)
    # Mocking: Check for keywords to simulate correctness/confidence
    is_correct = occursin("correct", lowercase(text)) || occursin("yes", lowercase(text))
    confidence_val = length(text) > 20 ? 0.9 : 0.5

    # Create an observation vector (ObservationType = CORRECTNESS)
    # We'll put correctness in dim 1, response time in dim 2, confidence in dim 3
    values = zeros(Float32, 8) # observation_dim
    values[1] = is_correct ? 1.0 : 0.0
    values[2] = 2.0 # Fake response time
    values[3] = confidence_val # Confidence

    return Observation{Float32}(CORRECTNESS, values, Dict{Symbol,Any}(:text => text))
end

# Map the model's numerical action back to text
function action_to_text(action::NamedTuple)
    type_str = string(action.action_type)

    responses = Dict(
        "PRESENT_PROBLEM" => "Let's try a problem. Difficulty: $(round(action.difficulty, digits=2)).",
        "PROVIDE_HINT" => "Here is a hint to help you move forward.",
        "PROVIDE_SOLUTION" => "Let's review the solution together to clear up any confusion.",
        "REVIEW_CONCEPT" => "I think we should review this concept again.",
        "ADJUST_DIFFICULTY" => "I'm going to adjust the difficulty to better suit your pace.",
        "SWITCH_TOPIC" => "Let's switch to a related topic.",
        "ENCOURAGE" => "You're doing great! Keep going.",
        "PAUSE" => "Let's take a short break."
    )

    base_response = get(responses, type_str, "I have a recommendation for you.")

    return "$base_response (Action: $type_str)"
end

# ============================================================================
# API Endpoints
# ============================================================================

<<<<<<< Updated upstream
<<<<<<< Updated upstream
@post "/api/ask" function (req::HTTP.Request)
    try
        data = JSON3.read(req.body)
        question = data.question
        student_id = get(data, "studentId", "default_student")

        session = get_or_create_session(student_id)

        # 1. Process Input -> Observation
        # In a real 'ask' scenario, the user input is treated as a query.
        # However, for the loop, we treat it as an observation of the student's current state/intent.
        obs = text_to_observation(question)

        # 2. Update State & Get Action
        action = step!(session, obs)

        # 3. Generate Response
        response_text = action_to_text(action)

        # 4. Get reasoning/rationale
        # The step! function already adds it to history, but we can reconstruct or simple explain
        # We can extract the rationale from the last history item if we want
        latest_history = session.history[end]
        rationale_obj = latest_history.rationale

        # Convert simple types for JSON
        reasoning_steps = [
            "Analyzed input for correctness/intent",
            "Updated belief state (uncertainty: $(round(session.uncertainty, digits=2)))",
            "Selected action: $(action.action_type)",
            "Value estimate: $(round(rationale_obj.value_estimate, digits=2))"
        ]

        # 5. Format Knowledge Dimensions (Visualization)
        belief, var = get_belief_state(session)

        # We just map the first few dims for simple viz
        knowledge_dims = []
        for i in 1:min(50, length(belief))
            push!(knowledge_dims, Dict(
                "dimension" => i,
                "belief" => belief[i],
                "variance" => i <= length(var) ? var[i] : 0.0,
                "active" => belief[i] > 0.1
            ))
        end

        response = Dict(
            "answer" => response_text,
            "reasoning" => reasoning_steps,
            "confidence" => 1.0 - session.uncertainty,
            "knowledgeDimensions" => knowledge_dims,
            "timestamp" => time()
        )

        return HTTP.Response(200, ["Access-Control-Allow-Origin" => "*", "Content-Type" => "application/json"], JSON3.write(response))

    catch e
        println("Error in /api/ask: $e")
        Base.show_backtrace(stdout, catch_backtrace())
        return HTTP.Response(500, ["Access-Control-Allow-Origin" => "*"], JSON3.write(Dict("error" => string(e))))
    end
end

@post "/api/feedback" function (req::HTTP.Request)
    # Placeholder for feedback endpoint
    return HTTP.Response(200, ["Access-Control-Allow-Origin" => "*"], JSON3.write(Dict("status" => "received")))
end

@get "/api/telemetry" function (req::HTTP.Request)
    # Return states of all active sessions
    students_data = []

    lock(session_lock) do
        for (sid, session) in sessions
            # Extract basic belief stats
            b, v = get_belief_state(session)
            # Just take the first 50 dims if larger
            b_trunc = b[1:min(length(b), 50)]
            v_trunc = length(v) > 0 ? v[1:min(length(v), 50)] : zeros(Float32, length(b_trunc))

            # Use hash of string to get a consistent integer ID for the UI
            id_hash = hash(sid) % 1000

            push!(students_data, Dict(
                "id" => abs(id_hash),
                "belief" => b_trunc,
                "variance" => v_trunc,
                "timestep" => session.step_count
            ))
        end
    end

    # If no sessions, return dummy data so UI isn't empty
    if isempty(students_data)
        push!(students_data, Dict(
            "id" => 1,
            "belief" => zeros(Float32, 50),
            "variance" => ones(Float32, 50) .* 0.5,
            "timestep" => 0
        ))
    end

    response = Dict(
        "timestamp" => time(),
        "students" => students_data
    )

    return HTTP.Response(200, ["Access-Control-Allow-Origin" => "*", "Content-Type" => "application/json"], JSON3.write(response))
end

# Options for CORS
@post "/api/ask" function (req::HTTP.Request)
    return HTTP.Response(200, ["Access-Control-Allow-Origin" => "*", "Access-Control-Allow-Headers" => "*", "Access-Control-Allow-Methods" => "POST, OPTIONS"])
end
Oxygen.options("/api/ask", (req) -> HTTP.Response(200, ["Access-Control-Allow-Origin" => "*", "Access-Control-Allow-Headers" => "Content-Type", "Access-Control-Allow-Methods" => "POST, OPTIONS"]))
Oxygen.options("/api/feedback", (req) -> HTTP.Response(200, ["Access-Control-Allow-Origin" => "*", "Access-Control-Allow-Headers" => "Content-Type", "Access-Control-Allow-Methods" => "POST, OPTIONS"]))


println("StudentIO Julia Server running on port 8080...")
println("Ready to accept connections.")
serve(host="0.0.0.0", port=8080)
=======
@get "/api/telemetry" function (req::HTTP.Request)
    # Extract hidden state from the LSTM layer (layer 2 in the Chain)
    # The state structure depends on the Flux version, but typically for LSTM
    # it's a tuple (h, c). We want 'h' (the hidden output).

    # Safe access to recurrent state
    # LSTM state in Flux.state(chain) is usually named similarly to the layer
    # We will grab the raw hidden state vector.

    # Note: In recent Flux, state is a NamedTuple. 
    # We need to extract the hidden matrix.
    # For a simple demo, we will re-run the model on 'noise' if idle to get a live look,
    # OR better, just expose the current 'h' from the global state if accessible.

    # Accessing the specific LSTM state (layer 2)
    # core_state[2] should be the LSTM state -> (h, c)
    lstm_state = core_state[2]

    # Depending on Flux version, this might be a tuple or NamedTuple.
    # We'll convert to vector safely.
    hidden_vec = Float32[]

    try
        # Common structure: (h, c)
        if isa(lstm_state, Tuple)
            hidden_vec = vec(lstm_state[1]) # h
        elseif hasproperty(lstm_state, :h) # NamedTuple or struct
            hidden_vec = vec(lstm_state.h)
        else
            # Fallback: just return zeros if structure is unexpected
            hidden_vec = zeros(Float32, HIDDEN_DIM)
        end
    catch e
        println("Telemetry Error: $e")
        hidden_vec = zeros(Float32, HIDDEN_DIM)
    end

    return HTTP.Response(200, ["Access-Control-Allow-Origin" => "*", "Content-Type" => "application/json"],
        JSON3.write(Dict(
            "timestamp" => Dates.now(),
            "activations" => hidden_vec
        )))
end

@post "/api/chat" function (req::HTTP.Request)
    try
        global core_state
        data = JSON3.read(String(req.body))
        user_text = data.text

=======
@get "/api/telemetry" function (req::HTTP.Request)
    # Extract hidden state from the LSTM layer (layer 2 in the Chain)
    # The state structure depends on the Flux version, but typically for LSTM
    # it's a tuple (h, c). We want 'h' (the hidden output).

    # Safe access to recurrent state
    # LSTM state in Flux.state(chain) is usually named similarly to the layer
    # We will grab the raw hidden state vector.

    # Note: In recent Flux, state is a NamedTuple. 
    # We need to extract the hidden matrix.
    # For a simple demo, we will re-run the model on 'noise' if idle to get a live look,
    # OR better, just expose the current 'h' from the global state if accessible.

    # Accessing the specific LSTM state (layer 2)
    # core_state[2] should be the LSTM state -> (h, c)
    lstm_state = core_state[2]

    # Depending on Flux version, this might be a tuple or NamedTuple.
    # We'll convert to vector safely.
    hidden_vec = Float32[]

    try
        # Common structure: (h, c)
        if isa(lstm_state, Tuple)
            hidden_vec = vec(lstm_state[1]) # h
        elseif hasproperty(lstm_state, :h) # NamedTuple or struct
            hidden_vec = vec(lstm_state.h)
        else
            # Fallback: just return zeros if structure is unexpected
            hidden_vec = zeros(Float32, HIDDEN_DIM)
        end
    catch e
        println("Telemetry Error: $e")
        hidden_vec = zeros(Float32, HIDDEN_DIM)
    end

    return HTTP.Response(200, ["Access-Control-Allow-Origin" => "*", "Content-Type" => "application/json"],
        JSON3.write(Dict(
            "timestamp" => Dates.now(),
            "activations" => hidden_vec
        )))
end

@post "/api/chat" function (req::HTTP.Request)
    try
        global core_state
        data = JSON3.read(String(req.body))
        user_text = data.text

>>>>>>> Stashed changes
        # 1. Embed Input
        x = text_to_embedding(user_text, INPUT_DIM)

        # 2. Run Model (State Update)
        # Flux.stateful call if we wrapped it, or clean function call:
        (y, new_state) = Flux.run(model, x, core_state)

        # Update global state
        core_state = new_state

        # 3. Interpret Output
        # y is [3] probability vector
        sentiment_idx = argmax(y)
        sentiments = ["negative", "neutral", "positive"]
        detected_sentiment = sentiments[sentiment_idx]

        # 4. Generate Response
        response_text = get_heuristic_response_text(user_text, sentiment_idx)

        return HTTP.Response(200, ["Access-Control-Allow-Origin" => "*", "Content-Type" => "application/json"],
            JSON3.write(Dict(
                "response" => response_text,
                "sentiment" => detected_sentiment,
                "model_output" => y
            )))

    catch e
        println("Chat Error: $e")
        return HTTP.Response(500, ["Access-Control-Allow-Origin" => "*"], JSON3.write(Dict("error" => string(e))))
    end
end

println("StudentIO Single Core (Flux) Server running on port 8080...")
<<<<<<< Updated upstream
serve(port=8080)
>>>>>>> Stashed changes
=======
serve(port=8080)
>>>>>>> Stashed changes
