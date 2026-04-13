import torch
import torch.nn as nn

from eb_jepa.logging import get_logger
from eb_jepa.utils import flatten_spatio_temporal, unflatten_spatio_temporal

logging = get_logger(__name__)


class JEPAbase(nn.Module):
    """Base JEPA class for planning and inference only. Use JEPA subclass for training."""

    def __init__(self, encoder, aencoder, predictor):
        """Initialize JEPAbase with encoder, action encoder, and predictor."""
        super().__init__()
        # Observation Encoder
        self.encoder = encoder
        # Action Encoder
        self.action_encoder = aencoder
        # Predictor
        self.predictor = predictor
        self.single_unroll = getattr(self.predictor, "is_rnn", False)

    def save(self, file):
        torch.save(self.state_dict(), file)

    def load(self, file):
        self.load_state_dict(torch.load(file), weights_only=False)

    @torch.no_grad()
    def encode(self, observations):
        """Encode a sequence of observations and return the encoder output."""
        return self.encoder(observations)


class JEPA(JEPAbase):
    """Trainable JEPA with prediction loss and anti-collapse regularizer."""

    def __init__(
        self,
        encoder,
        aencoder,
        predictor,
        regularizer,
        predcost,
        predictor_space="encoder",
        predictor_proj=None,
    ):
        """Initialize JEPA with regularizer and prediction cost in addition to base components."""
        super().__init__(encoder, aencoder, predictor)
        self.regularizer = regularizer
        self.predcost = predcost
        self.predictor_space = predictor_space
        self.predictor_proj = nn.Identity() if predictor_proj is None else predictor_proj
        self.ploss = 0
        self.rloss = 0

    def _project_state_for_predictor(self, state):
        """Project [B, C, T, H, W] state into predictor/projector feature space."""
        x_flat, (b, c, t, h, w) = flatten_spatio_temporal(state)
        state_proj = self.predictor_proj(x_flat)
        return unflatten_spatio_temporal(state_proj, b, t, h, w)

    def get_features(self, observations):
        """Return a feature dictionary with encoder and projector-space states."""
        encoder_state = self.encoder(observations)
        return {
            "encoder": encoder_state,
            "projector": self._project_state_for_predictor(encoder_state),
        }

    def route_state(self, state, source):
        """Route an encoder-space state tensor to the requested source space."""
        if source == "encoder":
            return state
        if source == "projector":
            return self._project_state_for_predictor(state)
        raise ValueError(f"Unknown feature source '{source}'. Expected: encoder, projector.")

    @torch.no_grad()
    def infer(self, observations, actions):
        """Produce single-step predictions over all sequence elements in parallel."""
        preds, _ = self.unroll(
            observations,
            actions,
            nsteps=1,
            unroll_mode="parallel",
            compute_loss=False,
            return_all_steps=True,
        )
        return preds[0]

    def unroll(
        self,
        observations,
        actions,
        nsteps=1,
        unroll_mode="parallel",
        ctxt_window_time=1,
        compute_loss=True,
        return_all_steps=False,
    ):
        """Unified multi-step prediction with optional loss computation.

        This function supports both training (with loss computation) and planning/inference
        (without loss, just state prediction).

        Usage examples:
        - Training video_jepa: unroll(x, None, nsteps, unroll_mode="parallel", compute_loss=True)
        - Training ac_video_jepa with RNN: unroll(x, a, nsteps, unroll_mode="autoregressive",
          ctxt_window_time=1, compute_loss=True)
        - Planning with ac_video_jepa: unroll(x, a, nsteps, unroll_mode="autoregressive",
          ctxt_window_time=k, compute_loss=False)
        - Inference like infern(): unroll(x, a, nsteps, unroll_mode="parallel",
          compute_loss=False, return_all_steps=True)

        Predictor behavior:
        - unroll_mode="parallel" (Conv predictor, is_rnn=False):
          Processes all timesteps in parallel. Uses predictor.context_length to
          determine how many ground truth frames to re-feed at each iteration.
          Output: [B, D, T, H', W'] (same length as input, predictions replace non-context).
          Best for training with full ground truth trajectory available.

        - unroll_mode="autoregressive":
          Step-by-step prediction with sliding window of ctxt_window_time states.
          Each step: takes last ctxt_window_time states, predicts next, appends to sequence.
          Output: [B, D, T_context + nsteps, H', W'] (context + predictions appended).
          Best for planning/inference where future ground truth is not available.
          Note: RNN predictors (is_rnn=True) are a special case with ctxt_window_time=1.

        Args:
            observations: [B, C, T, H, W] - observation sequence
                For training (compute_loss=True): full trajectory with ground truth
                For planning (compute_loss=False): context frames only
            actions: [B, A, T_actions] - action sequence, or None for state-only prediction
                T_actions >= nsteps required for autoregressive mode
            nsteps: number of prediction steps
            unroll_mode: "parallel" or "autoregressive"
                - "parallel": Process all timesteps, refeed GT context on left
                - "autoregressive": Step-by-step, append predictions on right
            ctxt_window_time: Context window size for autoregressive mode.
                For RNN predictors (is_rnn=True), this is effectively 1.
            compute_loss: Whether to compute losses (requires ground truth observations)
            return_all_steps: If True, return list of predictions at each step (like infern).
                If False, return only the final predicted states.

        Returns:
            Tuple of (predicted_states, losses) where:
            - If return_all_steps=False:
              predicted_states: [B, D, T_out, H', W'] - final predicted state sequence
            - If return_all_steps=True:
              predicted_states: List[Tensor] of length nsteps, each [B, D, T_out, H', W']
            - losses: None if compute_loss=False, otherwise tuple of 5 elements:
              (total_loss, reg_loss, reg_loss_unweighted, reg_loss_dict, pred_loss)
        """
        state = self.encoder(observations)
        state_for_predictor = self.route_state(state, self.predictor_space)
        context_length = getattr(self.predictor, "context_length", 0)

        # Compute regularization loss if needed
        if compute_loss:
            rloss, rloss_unweight, rloss_dict = self.regularizer(state, actions)
            ploss = 0.0
        else:
            rloss = rloss_unweight = rloss_dict = ploss = None

        # Encode actions
        if actions is not None:
            actions_encoded = self.action_encoder(actions)
        else:
            actions_encoded = None

        # Collect all steps if requested
        all_steps = [] if return_all_steps else None

        # Parallel mode: process all timesteps at once, refeed GT context
        if unroll_mode == "parallel":
            predicted_states = state_for_predictor                                              # my: dimensions [B, D, T, H', W']
            for _ in range(nsteps):                                                             # my: loop over prediction steps. at each step, we predict all timesteps in parallel, but only keep up to T-1 since we predict t+1 from t. we refeed GT context on the left at each step.
                # Predict all timesteps, discard last (no target for it)
                predicted_states = self.predictor(predicted_states, actions_encoded)[           # my: new prediction for all timesteps, but we only keep up to T-1 since we predict t+1 from t
                    :, :, :-1                                                                   # my: removing last timestep since we don't have a target for it (we predict t+1 from t, so last prediction has no t+1 target)
                ]   
                # Collect step if requested
                if return_all_steps:                                                            # my: ← store predicted states at this step. structure of all_steps: list of length nsteps, each element is [B, D, T-1, H', W'] (predicted states for all timesteps except last). 
                    all_steps.append(predicted_states)                                          # my: all_steps[0] corresponds to step 1 predictions, all_steps[1] to step 2 predictions, etc.
                # Refeed ground truth context on the left
                predicted_states = torch.cat(                                                   # my: re-anchor: GT frames 0,1 on left, predictions on right → back to [B, D, T, H, W]
                    (state_for_predictor[:, :, :context_length], predicted_states), dim=2       # my: context_length frames from GT, then predictions for the rest
                )
                if compute_loss:
                    ploss += self.predcost(state_for_predictor, predicted_states) / nsteps

        # Autoregressive mode: step-by-step with sliding window
        # Note: RNN predictors (is_rnn=True) are a special case with ctxt_window_time=1
        elif unroll_mode == "autoregressive":
            if actions is not None and nsteps > actions.size(2):
                raise ValueError(
                    f"nsteps ({nsteps}) larger than action sequence length ({actions.size(2)})"
                )
            # For RNN predictors, force ctxt_window_time=1
            effective_ctxt_window = 1 if self.single_unroll else ctxt_window_time

            predicted_states = state_for_predictor[:, :, :effective_ctxt_window]
            for i in range(nsteps):
                # Take last ctxt_window_time states
                context_states = predicted_states[:, :, -effective_ctxt_window:]
                # Take corresponding actions
                if actions_encoded is not None:
                    context_actions = actions_encoded[
                        :, :, max(0, i + 1 - effective_ctxt_window) : i + 1
                    ]
                else:
                    context_actions = None
                # Predict and take only last timestep
                pred_step = self.predictor(context_states, context_actions)[:, :, -1:]
                # Append prediction to sequence
                predicted_states = torch.cat([predicted_states, pred_step], dim=2)
                # Collect step if requested
                if return_all_steps:
                    all_steps.append(predicted_states.clone())
                if compute_loss:
                    ploss += (
                        self.predcost(pred_step, state_for_predictor[:, :, i + 1 : i + 2])
                        / nsteps
                    )
        else:
            raise ValueError(f"Unknown unroll_mode: {unroll_mode}")

        # Compute total loss and return
        if compute_loss:
            loss = rloss + ploss
            losses = (loss, rloss, rloss_unweight, rloss_dict, ploss)
        else:
            losses = None

        # Return all steps or just final state
        if return_all_steps:
            return all_steps, losses
        else:
            return predicted_states, losses


class JEPAProbe(nn.Module):
    """JEPA with a trainable prediction head. The JEPA encoder is kept fixed."""

    def __init__(self, jepa, head, hcost, feature_source="encoder"):
        """Initialize with a frozen JEPA, prediction head, and head loss function."""
        super().__init__()
        self.jepa = jepa
        self.head = head
        self.hcost = hcost
        self.feature_source = feature_source

    def head_parameters(self):
        """Return parameters of the trainable probe head only."""
        return self.head.parameters()

    def _align_embeddings_to_probe_source(self, embeddings, embedding_source):
        """Align externally provided embeddings to this probe's source space."""
        source = embedding_source or self.feature_source
        if source == self.feature_source:
            return embeddings
        if source == "encoder" and self.feature_source == "projector":
            return self.jepa.route_state(embeddings, "projector")
        raise ValueError(
            f"Cannot map embeddings from source '{source}' to probe source "
            f"'{self.feature_source}'."
        )

    @torch.no_grad()
    def infer(self, observations):
        """Encode observations through JEPA and apply the prediction head."""
        state = self.jepa.get_features(observations)[self.feature_source]
        return self.head(state)

    @torch.no_grad()
    def apply_head(self, embeddings, embedding_source=None):
        """
        Decode embeddings using the head.
        This is useful for generating predictions from an unrolling of the predictor, for example.
        """
        aligned = self._align_embeddings_to_probe_source(embeddings, embedding_source)
        return self.head(aligned)

    @torch.no_grad()
    def score(self, preds, targets, pred_source=None):
        """Score predicted latent trajectories against targets using the detection head."""
        aligned_preds = [
            self._align_embeddings_to_probe_source(pred, pred_source)
            for pred in preds
        ]
        return self.head.score(aligned_preds, targets)

    def forward(self, observations, targets):
        """Forward pass for training the head (JEPA encoder gradients are detached)."""
        with torch.no_grad():
            state = self.jepa.get_features(observations)[self.feature_source]
        output = self.head(state.detach())
        return self.hcost(output, targets)


class MultiSourceJEPAProbe(nn.Module):
    """Container that trains multiple JEPAProbe sources and exposes one active source."""

    def __init__(self, probes, active_source):
        super().__init__()
        if not probes:
            raise ValueError("MultiSourceJEPAProbe requires at least one probe.")
        self.probes = nn.ModuleDict(probes)
        self.set_active_source(active_source)

    def set_active_source(self, source):
        if source not in self.probes:
            raise ValueError(
                f"Unknown active probe source '{source}'. Available: {list(self.probes.keys())}"
            )
        self.active_source = source

    @property
    def head(self):
        return self.probes[self.active_source].head

    def source_names(self):
        return tuple(self.probes.keys())

    def head_parameters(self):
        for probe in self.probes.values():
            yield from probe.head_parameters()

    def infer(self, observations):
        return self.probes[self.active_source].infer(observations)

    def apply_head(self, embeddings, embedding_source=None):
        return self.probes[self.active_source].apply_head(
            embeddings, embedding_source=embedding_source
        )

    @torch.no_grad()
    def score(self, preds, targets, pred_source=None):
        return self.probes[self.active_source].score(
            preds, targets, pred_source=pred_source
        )

    @torch.no_grad()
    def score_by_source(self, preds, targets, pred_source=None):
        return {
            source: probe.score(preds, targets, pred_source=pred_source)
            for source, probe in self.probes.items()
        }

    def forward_with_source_losses(self, observations, targets):
        losses = {
            source: probe(observations, targets)
            for source, probe in self.probes.items()
        }
        mean_loss = sum(losses.values()) / len(losses)
        return mean_loss, losses

    def forward(self, observations, targets):
        mean_loss, _ = self.forward_with_source_losses(observations, targets)
        return mean_loss
