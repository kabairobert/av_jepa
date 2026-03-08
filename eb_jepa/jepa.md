

### The Mechanics of `parallel` Mode

The predictor doesn't hardcode a subtraction. It is an intricate 3D Convolution (`ResUNet`) mapping a sliding window. In this network, temporal convolutions are typically **"valid" (no padding in the time dimension)**, meaning they naturally reduce the time dimension. 

Assume Ground Truth `state` length $T = 4$ frames: `[S0, S1, S2, S3]`.
Assume `context_length = 2` (temporal receptive field of the CNN).

Here is the exact trace of what PyTorch is doing:

**1. The Forward Pass (`self.predictor(predicted_states)`)**
The CNN takes the whole length-4 sequence. Because it has a temporal window of 2, it slides over the time dimension, computing outputs based *only* on the pairs it sees:
* Window `[S0, S1]` $\rightarrow$ Outputs prediction for step 2: `P2`
* Window `[S1, S2]` $\rightarrow$ Outputs prediction for step 3: `P3`
* Window `[S2, S3]` $\rightarrow$ Outputs prediction for step 4: `P4`

The predictor returns a tensor of length 3: `[P2, P3, P4]`. Notice that **strict causality is maintained**: to predict `P2`, it only looked at `S0` and `S1`. It never saw `S2`.

**2. The Slicing (`[:, :, :-1]`)**
```python
predicted_states = self.predictor... [:, :, :-1]
```
This drops the last element (`P4`). 
Why? Because `P4` corresponds to a time step outside our current ground-truth boundary, so we have nothing to compute loss against.
Tensor becomes length 2: `[P2, P3]`.

**3. The Concatenation (`torch.cat(...)`)**
```python
predicted_states = torch.cat((state[:, :, :context_length], predicted_states), dim=2)
```
We take the true context `state[:, :, :2]`, which is `[S0, S1]`.
We concatenate it with our sliced predictions `[P2, P3]`.
Resulting tensor is length 4: `[S0, S1, P2, P3]`.

**4. The Loss**
```python
self.predcost(state, predicted_states)
```
PyTorch compares the original sequence `[S0, S1, S2, S3]` against `[S0, S1, P2, P3]`.
Because `S0` matches `S0` and `S1` matches `S1` exactly, the loss is mathematically isolated to penalizing how poorly `P2` matched `S2` and `P3` matched `S3`.

**5. The real implementation**
How parallel mode works (The Tensor Flow)
In parallel mode, we feed the full state sequence into the predictor at once.

Initial context: [S0, S1, S2, S3] (T=4). Note that S2,S3 are ground truth here.
Step 1:
    Predictor Input: [S0, S1, S2, S3].
    StateOnlyPredictor creates pairs: [(S0,S1), (S1,S2), (S2,S3)].
    ResUNet outputs: [P2, P3, P4] (T=3).
    jepa.py slices [:-1]: We drop P4 because we don't have S4 to compare it to for loss.
    Tensor becomes: [P2, P3].
    jepa.py concatenates context: cat( [S0, S1], [P2, P3] )                 # <- GT anchor prefix
    Tensor becomes: [S0, S1, P2, P3].

Step 2 (if nsteps=2):
    Predictor Input: [S0, S1, P2, P3]
    StateOnlyPredictor creates pairs: [(S0,S1), (S1,P2), (P2,P3)]
    ResUNet outputs: [P'2, P'3, P'4] (T=3).
    jepa.py slices [:-1]: Drops P'4. Tensor becomes [P'2, P'3].
    jepa.py concatenates context: cat( [S0, S1], [P'2, P'3] )               # <- GT anchor prefix
    Tensor becomes [S0, S1, P'2, P'3].
    Loss: is computed at the end against the ground truth. S0, S1 match perfectly (zero loss). Loss is purely calculated on P'2 vs S2 and P'3 vs S3.

**Warning**
    ! The subtle problem is that StateOnlyPredictor always creates pairs from adjacent frames, regardless of what

---

### What happens on the next `nsteps`?
This is where **Teacher Forcing** truly happens.
If `nsteps=2`, the loop runs again. But what goes into the predictor now?
The tensor `[S0, S1, P2, P3]`.

The CNN slides over *this* mixed tensor:
* Window `[S0, S1]` $\rightarrow$ Outputs `P'2` 
* Window `[S1, P2]` $\rightarrow$ Outputs `P'3` *(Notice it's now using its own past prediction!)*
* Window `[P2, P3]` $\rightarrow$ Outputs `P'4`

### Summary
* My previous diagram of "velocity mapping `S1-S0` to `S1`" was logically flawed. 
* The predictor natively learns feature transformations from blocks of `[S_t, S_{t+1}]` to output `P_{t+2}`.
* The `parallel` mode simply parallelizes this sliding window across the entire time sequence simultaneously, perfectly constrained by tensor indexing to prevent it from "looking into the future."

---


### ! Deprecated ### The Mechanics of `autoregressive` Mode 
In autoregressive mode, we do not start with the full sequence. We only start with the designated context window (e.g., T=2). The sequence grows continuously.

Initial context: [S0, S1] (T=2).

Step 1:
    Predictor Input: [S0, S1].
    StateOnlyPredictor creates pairs: [(S0,S1)] (Only 1 pair!).
    ResUNet outputs: [P2] (T=1).
    jepa.py extracts new frame: It explicitly takes the last frame of the prediction pred_step = predicted_states[:, :, -1:] (which is P2).
    jepa.py appends to context: pred_context = cat([S0, S1], [P2]).
    Tensor becomes: [S0, S1, P2].

Step 2:
    Predictor Input: [S0, S1, P2] (T=3).
    StateOnlyPredictor creates pairs: [(S0,S1), (S1,P2)].
    ResUNet outputs: [P2_remade, P3] (T=2). Note that because ResUNet processes these independently, it redundantly re-predicts step 2!
    jepa.py extracts new frame: pred_step = predicted_states[:, :, -1:]. It throws away P2_remade and only keeps P3.
    jepa.py appends to context: pred_context = cat([S0, S1, P2], [P3]).
    Tensor becomes [S0, S1, P2, P3].

**Warning**:
    ! Autoregressive mode does not work well with convolutional StateOnlyPredictors! very inefficient, wasteful!