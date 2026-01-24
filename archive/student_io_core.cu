#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include <stdio.h>
#include <time.h>


// Configuration
#define STATE_DIM 50
#define OBS_DIM 20
#define HIDDEN_DIM 128
#define ACTION_DIM 10

// Structures (Byte-aligned for Julia FFI)
typedef struct {
  float knowledge[STATE_DIM];
  float belief_mean[STATE_DIM];
  float belief_var[STATE_DIM];
  float hidden[HIDDEN_DIM];
  int timestep;
  float learning_rate;
  float forgetting_rate;
} StudentState;

typedef struct {
  float correctness;
  float response_time;
  float confidence;
  float help_requests;
  float features[OBS_DIM];
} Observation;

typedef struct {
  int concept_id;
  float difficulty;
  int problem_type;
  float features[ACTION_DIM];
} Action;

// Constant Memory for Weights
__device__ __constant__ float d_W_ih[HIDDEN_DIM * OBS_DIM];
__device__ __constant__ float d_W_hh[HIDDEN_DIM * HIDDEN_DIM];
__device__ __constant__ float d_W_out[STATE_DIM * HIDDEN_DIM];
__device__ __constant__ float d_policy_W[ACTION_DIM * HIDDEN_DIM];

// --- Utilities ---
__device__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// --- Kernels ---
__global__ void beliefUpdateKernel(StudentState *states,
                                   Observation *observations, int batch_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size)
    return;
  StudentState *s = &states[idx];
  Observation *obs = &observations[idx];

  float new_hidden[HIDDEN_DIM];
  for (int i = 0; i < HIDDEN_DIM; i++) {
    float sum = 0.0f;
    for (int j = 0; j < OBS_DIM; j++)
      sum += d_W_ih[i * OBS_DIM + j] * obs->features[j];
    for (int j = 0; j < HIDDEN_DIM; j++)
      sum += d_W_hh[i * HIDDEN_DIM + j] * s->hidden[j];
    new_hidden[i] = tanhf(sum);
  }

  for (int i = 0; i < STATE_DIM; i++) {
    float sum = 0.0f;
    for (int j = 0; j < HIDDEN_DIM; j++)
      sum += d_W_out[i * HIDDEN_DIM + j] * new_hidden[j];
    s->belief_mean[i] = sigmoid(sum);

    float pred_var = s->belief_var[i] + 0.05f;
    float K = pred_var / (pred_var + 0.1f);
    s->belief_var[i] = (1.0f - K) * pred_var;
    s->belief_mean[i] += K * (obs->correctness - s->belief_mean[i]) * 0.1f;
  }
  for (int i = 0; i < HIDDEN_DIM; i++)
    s->hidden[i] = new_hidden[i];
  s->timestep++;
}

__global__ void knowledgeEvolutionKernel(StudentState *states, Action *actions,
                                         int batch_size,
                                         unsigned long long seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size)
    return;
  curandState_t state;
  curand_init(seed, idx, 0, &state);

  int concept = actions[idx].concept_id;
  states[idx].knowledge[concept] +=
      states[idx].learning_rate * (1.0f - states[idx].knowledge[concept]);
  for (int i = 0; i < STATE_DIM; i++)
    states[idx].knowledge[i] *= (1.0f - states[idx].forgetting_rate);
}

__global__ void generateObservationKernel(StudentState *states,
                                          Observation *observations,
                                          int batch_size,
                                          unsigned long long seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size)
    return;
  curandState_t state;
  curand_init(seed, idx, 0, &state);
  observations[idx].correctness =
      states[idx].knowledge[0] + curand_normal(&state) * 0.1f;
}

__global__ void policyKernel(StudentState *states, Action *actions,
                             int batch_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size)
    return;
  actions[idx].concept_id = 0; // Simple greedy policy for demo
  actions[idx].difficulty = 5.0f;
}

// --- Exported C API ---
extern "C" {
void updateWeights(float *h_W_ih, float *h_W_hh, float *h_W_out,
                   float *h_policy_W) {
  cudaMemcpyToSymbol(d_W_ih, h_W_ih, HIDDEN_DIM * OBS_DIM * sizeof(float));
  cudaMemcpyToSymbol(d_W_hh, h_W_hh, HIDDEN_DIM * HIDDEN_DIM * sizeof(float));
  cudaMemcpyToSymbol(d_W_out, h_W_out, STATE_DIM * HIDDEN_DIM * sizeof(float));
  cudaMemcpyToSymbol(d_policy_W, h_policy_W,
                     ACTION_DIM * HIDDEN_DIM * sizeof(float));
}

void runTimestep(StudentState *d_states, Observation *d_observations,
                 Action *d_actions, int batch_size) {
  int threads = 256;
  int blocks = (batch_size + threads - 1) / threads;
  unsigned long long seed = (unsigned long long)time(NULL);

  generateObservationKernel<<<blocks, threads>>>(d_states, d_observations,
                                                 batch_size, seed);
  beliefUpdateKernel<<<blocks, threads>>>(d_states, d_observations, batch_size);
  policyKernel<<<blocks, threads>>>(d_states, d_actions, batch_size);
  knowledgeEvolutionKernel<<<blocks, threads>>>(d_states, d_actions, batch_size,
                                                seed + 1);
  cudaDeviceSynchronize();
}
}