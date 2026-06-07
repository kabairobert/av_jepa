#!/usr/bin/env python3
import re
import os
import shutil
import yaml
import json
import wandb
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent.parent.parent.parent
checkpoint_root = ROOT_DIR / "checkpoints" / "sslda"
cfg_dir = SCRIPT_DIR.parent.parent.parent / "cfgs" / "B12_data_scaling"
status_file = SCRIPT_DIR / "STATUS.md"
output_file = SCRIPT_DIR / "VISUALIZER.html"
assets_dir = SCRIPT_DIR / "VISUALIZER_htmls"
registry_file = SCRIPT_DIR.parent.parent / "metrics_registry.yaml"

# Ensure assets dir exists
assets_dir.mkdir(exist_ok=True)

# Load metrics registry
if not registry_file.exists():
    print(f"Error: {registry_file} not found")
    exit(1)
with open(registry_file) as f:
    registry = yaml.safe_load(f).get("metrics", {})

# Definition dictionary mapping alias -> (Display Name, Description, Formula)
metric_definitions = {
    # --- Subspace alignment / CCA ---
    "diagonality_ratio": (
        "Diagonality Ratio (CCA)",
        "Measures the proportion of cross-modal correlation concentrated strictly along corresponding coordinates. Crucial for axis-aligned representations.",
        "sum(diag(C)) / sum(C)"
    ),
    "cca_diag_score": (
        "CCA Diag Score",
        "Assesses dimension-wise correlation strength on the main diagonal of the CCA covariance matrix.",
        "sum(diag(corr(Z_A, Z_B)))"
    ),
    "cca_rank": (
        "CCA Effective Rank",
        "Measures the effective dimensionality of the shared cross-modal subspace. High values imply richer multi-dimensional representations.",
        "exp(-sum(p_i log p_i)) where p_i = s_i / sum(s)"
    ),
    "cca_dim0": (
        "CCA Dim 0 Correlation",
        "Correlation coefficient of the first canonical dimension between modalities.",
        "ρ_0 ∈ [0, 1]"
    ),
    "cca_dim1": (
        "CCA Dim 1 Correlation",
        "Correlation coefficient of the second canonical dimension between modalities.",
        "ρ_1 ∈ [0, 1]"
    ),
    "cca_dim2": (
        "CCA Dim 2 Correlation",
        "Correlation coefficient of the third canonical dimension between modalities.",
        "ρ_2 ∈ [0, 1]"
    ),
    "retrieval_cos@1": (
        "Retrieval Cos@1",
        "1-nearest neighbor cross-modal retrieval accuracy using cosine similarity. Indicates tight, synchronized latent coordinate matching.",
        "Acc@1 (Cosine)"
    ),
    "retrieval_cos@5": (
        "Retrieval Cos@5",
        "Top-5 nearest neighbor cross-modal retrieval accuracy using cosine similarity.",
        "Acc@5 (Cosine)"
    ),
    "retrieval_l2@1": (
        "Retrieval L2@1",
        "1-nearest neighbor cross-modal retrieval accuracy using L2 distance. Reflects absolute spatial synchronization.",
        "Acc@1 (L2)"
    ),
    "retrieval_l2@5": (
        "Retrieval L2@5",
        "Top-5 nearest neighbor cross-modal retrieval accuracy using L2 distance.",
        "Acc@5 (L2)"
    ),

    # --- Manifold flatness / unrolling ---
    "clean_flatness_ratio_a": (
        "Clean Flatness A",
        "Fraction of clean manifold A variance captured in its top-2 PCA dimensions. Measures how well JEPA unrolls the latent space into a flat plane.",
        "(λ_1 + λ_2) / sum(λ_i)"
    ),
    "clean_flatness_ratio_b": (
        "Clean Flatness B",
        "Fraction of clean manifold B variance captured in its top-2 PCA dimensions.",
        "(λ_1 + λ_2) / sum(λ_i)"
    ),
    "clean_orth_residual_a": (
        "Clean Orth Residual A",
        "Mean orthogonal distance of clean manifold A points from the top-2 PCA plane. Measures unrolling/curvature error.",
        "Mean distance to PCA plane"
    ),
    "clean_orth_residual_b": (
        "Clean Orth Residual B",
        "Mean orthogonal distance of clean manifold B points from the top-2 PCA plane.",
        "Mean distance to PCA plane"
    ),
    "flatness_ratio_a": (
        "Overall Flatness A",
        "Flatness ratio of modality A computed across all points (including noise). Serves as a pessimistic lower bound.",
        "(λ_1 + λ_2) / sum(λ_i)"
    ),
    "flatness_ratio_b": (
        "Overall Flatness B",
        "Flatness ratio of modality B computed across all points.",
        "(λ_1 + λ_2) / sum(λ_i)"
    ),
    "orth_residual_a": (
        "Overall Orth Residual A",
        "Mean orthogonal residual distance of modality A across all points.",
        "Mean distance to PCA plane"
    ),
    "orth_residual_b": (
        "Overall Orth Residual B",
        "Mean orthogonal residual distance of modality B across all points.",
        "Mean distance to PCA plane"
    ),
    "pca_axis_align_a": (
        "PCA Axis-Alignment A",
        "Maximum cosine similarity between top-2 PCA eigenvectors of A and standard coordinate axes. High value = axis-aligned.",
        "max_j |v_i^T e_j|"
    ),
    "pca_axis_align_b": (
        "PCA Axis-Alignment B",
        "Maximum cosine similarity between top-2 PCA eigenvectors of B and standard coordinate axes.",
        "max_j |v_i^T e_j|"
    ),

    # --- Information separation / disentanglement ---
    "r2_dim2_noise": (
        "Noise Leakage (r2_dim2_noise)",
        "Variance of the unique modality dimension explained by external noise. Low value proves clean isolation of private modality info.",
        "R² on noise dimension"
    ),
    "r2_dim0_u1": (
        "Disentanglement Dim 0 (u1)",
        "R² coefficient when regressing latent dimension 0 onto clean factor u1. Measures clean dimension-factor matching.",
        "R²(z_0, u_1)"
    ),
    "r2_dim1_u2": (
        "Disentanglement Dim 1 (u2)",
        "R² coefficient when regressing latent dimension 1 onto clean factor u2.",
        "R²(z_1, u_2)"
    ),
    "r2_joint": (
        "Joint Space R²",
        "Average variance explained when regressing both clean factors jointly from representations. Reflects total factor recovery.",
        "Mean R²(Z, U)"
    ),
    "r2_a": (
        "Modality A R²",
        "Average variance of clean factors explained by Modality A representations alone.",
        "Mean R²(Z_A, U)"
    ),
    "r2_b": (
        "Modality B R²",
        "Average variance of clean factors explained by Modality B representations alone.",
        "Mean R²(Z_B, U)"
    ),
    "r2_joint_u0": (
        "Joint R² (Factor 0)",
        "Variance of clean factor 0 explained jointly by both modalities.",
        "R²(Z, u_0)"
    ),
    "r2_joint_u1": (
        "Joint R² (Factor 1)",
        "Variance of clean factor 1 explained jointly by both modalities.",
        "R²(Z, u_1)"
    ),
    "r2_a_u0": (
        "Modality A R² (Factor 0)",
        "Variance of clean factor 0 explained by Modality A representations.",
        "R²(Z_A, u_0)"
    ),
    "r2_a_u1": (
        "Modality A R² (Factor 1)",
        "Variance of clean factor 1 explained by Modality A representations.",
        "R²(Z_A, u_1)"
    ),
    "r2_b_u0": (
        "Modality B R² (Factor 0)",
        "Variance of clean factor 0 explained by Modality B representations.",
        "R²(Z_B, u_0)"
    ),
    "r2_b_u1": (
        "Modality B R² (Factor 1)",
        "Variance of clean factor 1 explained by Modality B representations.",
        "R²(Z_B, u_1)"
    ),

    # --- Norms / Scales ---
    "z_a_norm": (
        "Z_A Norm Mean",
        "Mean L2 norm of modality A adapter representations. Used to monitor scale preservation vs collapse.",
        "E[||z_a||_2]"
    ),
    "z_b_norm": (
        "Z_B Norm Mean",
        "Mean L2 norm of modality B adapter representations.",
        "E[||z_b||_2]"
    ),
    "z_a_norm_manifold": (
        "Z_A Norm (Manifold)",
        "Mean L2 norm of modality A representations restricted to the clean manifold.",
        "E[||z_a||_2 | clean]"
    ),
    "z_b_norm_manifold": (
        "Z_B Norm (Manifold)",
        "Mean L2 norm of modality B representations restricted to the clean manifold.",
        "E[||z_b||_2 | clean]"
    ),
    "z_a_norm_asym_corrupt": (
        "Z_A Norm (Asym Corrupted)",
        "Mean L2 norm of modality A representations under asymmetric corruption.",
        "E[||z_a||_2 | asym"
    ),
    "z_b_norm_asym_corrupt": (
        "Z_B Norm (Asym Corrupted)",
        "Mean L2 norm of modality B representations under asymmetric corruption.",
        "E[||z_b||_2 | asym"
    ),
    "z_a_norm_external": (
        "Z_A Norm (External)",
        "Mean L2 norm of modality A representations under external noise injection.",
        "E[||z_a||_2 | external]"
    ),
    "z_b_norm_external": (
        "Z_B Norm (External)",
        "Mean L2 norm of modality B representations under external noise injection.",
        "E[||z_b||_2 | external]"
    ),

    # --- Losses and MSEs ---
    "train_loss": (
        "Train Loss",
        "Total aggregated training loss of the multimodal JEPA/EBM system.",
        "Lower is better"
    ),
    "val_loss": (
        "Val Loss",
        "Total validation loss evaluated under the same objective constraints.",
        "Lower is better"
    ),
    "val_align_a2b": (
        "Val Align A→B MSE",
        "Validation Mean Squared Error of the predictive model mapping modality A to modality B.",
        "E[||g_{a2b}(z_a) - z_b||^2]"
    ),
    "val_align_b2a": (
        "Val Align B→A MSE",
        "Validation Mean Squared Error of the predictive model mapping modality B to modality A.",
        "E[||g_{b2a}(z_b) - z_a||^2]"
    ),
    "train_align_a2b": (
        "Train Align A→B MSE",
        "Training Mean Squared Error of the predictive model mapping modality A to modality B.",
        "E[||g_{a2b}(z_a) - z_b||^2]"
    ),
    "train_align_b2a": (
        "Train Align B→A MSE",
        "Training Mean Squared Error of the predictive model mapping modality B to modality A.",
        "E[||g_{b2a}(z_b) - z_a||^2]"
    ),
    "align_mse_a2b_manifold": (
        "Manifold MSE A→B",
        "Predictive alignment MSE computed exclusively over clean manifold points.",
        "MSE | clean"
    ),
    "align_mse_b2a_manifold": (
        "Manifold MSE B→A",
        "Predictive alignment MSE computed exclusively over clean manifold points.",
        "MSE | clean"
    ),
    "align_mse_a2b_asym_corrupt": (
        "Asym MSE A→B",
        "Predictive alignment MSE computed over asymmetric corruption points.",
        "MSE | asym"
    ),
    "align_mse_b2a_asym_corrupt": (
        "Asym MSE B→A",
        "Predictive alignment MSE computed over asymmetric corruption points.",
        "MSE | asym"
    ),
    "align_mse_a2b_external": (
        "External Noise MSE A→B",
        "Predictive alignment MSE computed over external noise points.",
        "MSE | external"
    )
}

# Ordered by importance for Audio-Video Multimodal JEPA alignment analysis
ordered_metrics = [
    # 1. Subspace alignment & CCA
    "diagonality_ratio",
    "cca_diag_score",
    "cca_rank",
    "cca_dim0",
    "cca_dim1",
    "cca_dim2",
    "retrieval_cos@1",
    "retrieval_cos@5",
    "retrieval_l2@1",
    "retrieval_l2@5",

    # 2. Manifold flatness & unrolling
    "clean_flatness_ratio_a",
    "clean_flatness_ratio_b",
    "clean_orth_residual_a",
    "clean_orth_residual_b",
    "flatness_ratio_a",
    "flatness_ratio_b",
    "orth_residual_a",
    "orth_residual_b",
    "pca_axis_align_a",
    "pca_axis_align_b",

    # 3. Disentanglement & isolation
    "r2_dim2_noise",
    "r2_dim0_u1",
    "r2_dim1_u2",
    "r2_joint",
    "r2_a",
    "r2_b",
    "r2_joint_u0",
    "r2_joint_u1",
    "r2_a_u0",
    "r2_a_u1",
    "r2_b_u0",
    "r2_b_u1",

    # 4. Latent representation norms
    "z_a_norm",
    "z_b_norm",
    "z_a_norm_manifold",
    "z_b_norm_manifold",
    "z_a_norm_asym_corrupt",
    "z_b_norm_asym_corrupt",
    "z_a_norm_external",
    "z_b_norm_external",

    # 5. EBM / alignment losses
    "train_loss",
    "val_loss",
    "val_align_a2b",
    "val_align_b2a",
    "train_align_a2b",
    "train_align_b2a",
    "align_mse_a2b_manifold",
    "align_mse_b2a_manifold",
    "align_mse_a2b_asym_corrupt",
    "align_mse_b2a_asym_corrupt",
    "align_mse_a2b_external"
]

ordered_metrics_list = []
for m in ordered_metrics:
    if m in registry:
        ordered_metrics_list.append(m)
for m in registry.keys():
    if m not in ordered_metrics_list:
        ordered_metrics_list.append(m)

# Build dynamic Metric definitions dictionary HTML
dict_cards_html = []
for m in ordered_metrics_list:
    display_name, desc, formula = metric_definitions.get(m, (m.replace("_", " ").title(), f"Evaluation metric: {m}", ""))
    card_html = f"""            <div class="dict-card" id="dict-{m}">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 8px;">
                    <h4 style="margin: 0;">{display_name}</h4>
                    <span class="more-link" onclick="openModal('{m}')" style="color: #0969da; cursor: pointer; font-size: 11px; font-weight: 600; text-decoration: underline; white-space: nowrap;">(more)</span>
                </div>
                <p style="margin-top: 6px;">{desc}</p>
                {f'<div class="math-formula">{formula}</div>' if formula else ''}
            </div>"""
    dict_cards_html.append(card_html)
dict_cards_str = "\n".join(dict_cards_html)

# Build comprehensive LaTeX details for each metric (variable mapping, formulas, steps, diagnostics)
metric_details = {}
for m in ordered_metrics_list:
    display_name, desc, formula = metric_definitions.get(m, (m.replace("_", " ").title(), f"Evaluation metric: {m}", ""))
    
    # Base spaces definitions shown in all modals
    base_spaces = """
    <h5>Given Spaces and Variables:</h5>
    <ul>
        <li>$U = [u_0, u_1]^T \\in \\mathbb{R}^2$: True clean generative factors on 2D manifold</li>
        <li>$X \\in \\mathbb{R}^3, Y \\in \\mathbb{R}^3$: Modality observed input spaces (e.g. Audio, Video)</li>
        <li>$Z_A = f_A(X) \\in \\mathbb{R}^3, Z_B = f_B(Y) \\in \\mathbb{R}^3$: Transformed latent spaces</li>
        <li>$g_{a2b}: Z_A \\to Z_B, g_{b2a}: Z_B \\to Z_A$: Cross-modal predictive alignment maps</li>
    </ul>
    """
    
    if m == "diagonality_ratio":
        details = f"""
        <p>{desc}</p>
        {base_spaces}
        <h5>Mathematical Formulation:</h5>
        <div class="latex-block">
        $$D = \\frac{{\\sum_{{i=1}}^2 C_{{i,i}}}}{{\\sum_{{i=1}}^3 \\sum_{{j=1}}^3 |C_{{i,j}}|}}$$
        </div>
        <p>where $C = \\text{{corr}}(Z_A W_A, Z_B W_B)$ is the canonical correlation matrix.</p>
        <h5>Explanation:</h5>
        <p>Measures strict dimension-wise alignment of latent representations. A high diagonality ratio indicates that corresponding dimensions are strictly coupled, establishing coordinate-wise cross-modal synchronization.</p>
        """
    elif m == "cca_diag_score":
        details = f"""
        <p>{desc}</p>
        {base_spaces}
        <h5>Mathematical Formulation:</h5>
        <div class="latex-block">
        $$\\text{{Score}} = \\sum_{{i=1}}^3 \\rho_i \\cdot \\mathbb{{I}}\\left(i = \\text{{argmax}}_j \\ |\\text{{corr}}(Z_{{A,i}}, Z_{{B,j}})|\\right)$$
        </div>
        <h5>Explanation:</h5>
        <p>Evaluates on-diagonal cross-modal correlation versus off-diagonal coordinate mixing. High values show clean 1-to-1 dimensional matching.</p>
        """
    elif m == "cca_rank":
        details = f"""
        <p>{desc}</p>
        {base_spaces}
        <h5>Mathematical Formulation:</h5>
        <div class="latex-block">
        $$R_{{eff}} = \\exp\\left(-\\sum_{{i=1}}^3 p_i \\ln p_i\\right) \\quad \\text{{where}} \\quad p_i = \\frac{{\\rho_i^2}}{{\\sum_{{j=1}}^3 \\rho_j^2}}$$
        </div>
        <p>where $\\rho_i$ represents the correlation coefficient along canonical dimension $i$.</p>
        <h5>Explanation:</h5>
        <p>The effective rank of linear correlation pathways between adapters. Ideally exactly 2.0, indicating stable recovery of the shared 2D factor manifold.</p>
        """
    elif m in ["cca_dim0", "cca_dim1", "cca_dim2"]:
        dim_idx = m[-1]
        details = f"""
        <p>{desc}</p>
        {base_spaces}
        <h5>Mathematical Formulation:</h5>
        <div class="latex-block">
        $$\\rho_{dim_idx} = \\text{{PearsonCorrelation}}(Z_{{A, {dim_idx}}}^*, Z_{{B, {dim_idx}}}^*)$$
        </div>
        <h5>Explanation:</h5>
        <p>Linear correlation coefficient along canonical dimension {dim_idx}. Monitors active pathways between audio and video representation columns.</p>
        """
    elif m.startswith("retrieval_"):
        sim_type = "Cosine Similarity" if "cos" in m else "L2 Distance"
        k_val = m.split("@")[1]
        details = f"""
        <p>{desc}</p>
        {base_spaces}
        <h5>Mathematical Formulation:</h5>
        <div class="latex-block">
        $$\\text{{Accuracy}} = \\frac{{1}}{{N}} \\sum_{{k=1}}^N \\mathbb{{I}}\\left(Z_B^{{(k)}} \\in \\text{{Top-{k_val} Nearest Neighbors}}(g_{{a2b}}(Z_A^{{(k)}}))\\right)$$
        </div>
        <h5>Explanation:</h5>
        <p>Cross-modal nearest-neighbor query accuracy evaluated using {sim_type}. Higher accuracy proves synchronized global coordinate grids between video and audio latent adapters.</p>
        """
    elif m.endswith("_flatness_ratio_a") or m.endswith("_flatness_ratio_b"):
        mod = "A" if m.endswith("_a") else "B"
        ptype = "clean manifold factors" if "clean" in m else "all data points (including noise)"
        details = f"""
        <p>{desc}</p>
        {base_spaces}
        <h5>Mathematical Formulation:</h5>
        <div class="latex-block">
        $$F = \\frac{{\\lambda_1 + \\lambda_2}}{{\\sum_{{i=1}}^3 \\lambda_i}}$$
        </div>
        <p>where $\\lambda_1 \\ge \\lambda_2 \\ge \\lambda_3$ are eigenvalues of the sample covariance matrix $\\Sigma = \\frac{{1}}{{N}} Z_{mod.lower()}^T Z_{mod.lower()}$ computed over {ptype}.</p>
        <h5>Explanation:</h5>
        <p>Tracks manifold unrolling. A value of 1.0 indicates perfectly flat unrolling (no folding or curvature).</p>
        """
    elif m.endswith("_orth_residual_a") or m.endswith("_orth_residual_b"):
        mod = "A" if m.endswith("_a") else "B"
        ptype = "clean manifold factors" if "clean" in m else "all data points (including noise)"
        details = f"""
        <p>{desc}</p>
        {base_spaces}
        <h5>Mathematical Formulation:</h5>
        <div class="latex-block">
        $$\\delta_{{orth}} = \\frac{{1}}{{N}} \\sum_{{k=1}}^N \\|z_{{mod.lower()}}^{{(k)}} - P_{{top2}}(z_{{mod.lower()}}^{{(k)}})\\|_2$$
        </div>
        <p>where $P_{{top2}} = V_2 V_2^T$ is the orthogonal projection matrix onto the top-2 PCA unrolling plane.</p>
        <h5>Explanation:</h5>
        <p>Quantifies structural folding and unrolling error. Ideally 0.0, indicating zero curvature residual.</p>
        """
    elif m == "r2_dim2_noise":
        details = f"""
        <p>{desc}</p>
        {base_spaces}
        <h5>Mathematical Formulation:</h5>
        <div class="latex-block">
        $$R^2 = 1 - \\frac{{\\sum_{{k=1}}^N (z_{{A,2}}^{{(k)}} - \\hat{{z}}_{{A,2}}^{{(k)}})^2}}{{\\sum_{{k=1}}^N (z_{{A,2}}^{{(k)}} - \\bar{{z}}_{{A,2}})^2}}$$
        </div>
        <p>where $\\hat{{z}}_{{A,2}}$ represents prediction from linear regression models mapped from external noise coordinates.</p>
        <h5>Explanation:</h5>
        <p>Quantifies subspace noise leakage. A low R² is highly desirable, proving complete and clean isolation of unique private modalities from noise.</p>
        """
    elif m.startswith("r2_dim") or m.startswith("r2_joint") or m.startswith("r2_a") or m.startswith("r2_b"):
        details = f"""
        <p>{desc}</p>
        {base_spaces}
        <h5>Mathematical Formulation:</h5>
        <div class="latex-block">
        $$R^2 = 1 - \\frac{{\\sum_{{k=1}}^N (u_j^{{(k)}} - \\hat{{u}}_j^{{(k)}})^2}}{{\\sum_{{k=1}}^N (u_j^{{(k)}} - \\bar{{u}}_j)^2}}$$
        </div>
        <h5>Explanation:</h5>
        <p>Evaluates generative clean factor recovery and disentanglement. High value indicates latent dimensions capture the true coordinates cleanly.</p>
        """
    elif m.endswith("_norm") or "_norm_" in m:
        details = f"""
        <p>{desc}</p>
        {base_spaces}
        <h5>Mathematical Formulation:</h5>
        <div class="latex-block">
        $$\\text{{E}}[\\|z\\|_2] = \\frac{{1}}{{N}} \\sum_{{k=1}}^N \\|z^{{(k)}}\\|_2$$
        </div>
        <h5>Explanation:</h5>
        <p>Tracks average coordinate scaling to safeguard against collapse or exploding bounds.</p>
        """
    elif m.endswith("_loss"):
        details = f"""
        <p>{desc}</p>
        {base_spaces}
        <h5>Mathematical Formulation:</h5>
        <div class="latex-block">
        $$\\mathcal{{L}} = \\mathcal{{L}}_{{align}} + \\lambda_{{pri}} \\mathcal{{L}}_{{prior}} + \\lambda_{{vol}} \\mathcal{{L}}_{{volume}}$$
        </div>
        <h5>Explanation:</h5>
        <p>Overall objective cost minimized by optimization adapters.</p>
        """
    elif "align_a2b" in m or "align_b2a" in m:
        src, tgt = ("A", "B") if "a2b" in m else ("B", "A")
        details = f"""
        <p>{desc}</p>
        {base_spaces}
        <h5>Mathematical Formulation:</h5>
        <div class="latex-block">
        $$\\text{{MSE}} = \\frac{{1}}{{N}} \\sum_{{k=1}}^N \\|g_{{{src.lower()}2{tgt.lower()}}}(z_{src.lower()}^{{(k)}}) - z_{tgt.lower()}^{{(k)}}\\|_2^2$$
        </div>
        <h5>Explanation:</h5>
        <p>Quantifies cross-modal predictability and spatial alignment mapped from Modality {src} to Modality {tgt}.</p>
        """
    else:
        details = f"""
        <p>{desc}</p>
        {base_spaces}
        <h5>Calculation & Math:</h5>
        <div class="latex-block">
        $$\\text{{{display_name}}} \\quad \\text{{evaluates}} \\quad Z_{{A}} \\quad \\text{{and}} \\quad Z_{{B}}$$
        </div>
        <h5>Explanation:</h5>
        <p>Detailed performance index tracking representation properties under sweep configs.</p>
        """
        
    metric_details[m] = details

# Display maps (Fixed for B12 context)
noise_display_map = {
    "1": "Ext 10%",
}
prior_display_map = {
    "1": "L1 Prior",
    "2": "L2 Prior",
}
pred_display_map = {
    "1": "L1 Predictor",
    "2": "L2 Predictor",
}

print("Scanning for local Plotly HTML files...")
run_id_to_latest_html = {}
if checkpoint_root.exists():
    for html_file in checkpoint_root.glob("**/interactive_3d_4way_html_*.html"):
        parts = html_file.parts
        run_id = None
        for p in parts:
            if p.startswith("run-") and "-" in p:
                run_id = p.split("-")[-1]
                break
        
        if run_id:
            if run_id not in run_id_to_latest_html:
                run_id_to_latest_html[run_id] = html_file
            else:
                if html_file.stat().st_mtime > run_id_to_latest_html[run_id].stat().st_mtime:
                    run_id_to_latest_html[run_id] = html_file

print(f"Loaded {len(run_id_to_latest_html)} local plotly files.")

# Query WandB API
api = wandb.Api()
entity = "robertkabai-um"
project = "eb_jepa"
batch_id = "B12_data_scaling"

print("Querying WandB for B12 runs...")
runs = api.runs(f"{entity}/{project}", filters={"tags": {"$in": [batch_id]}})

# Group runs by configuration name
cfg_to_runs = {}
for r in runs:
    cfg_tag = next((tag for tag in r.tags if tag.startswith("B12_")), None)
    if cfg_tag:
        if cfg_tag not in cfg_to_runs:
            cfg_to_runs[cfg_tag] = []
        cfg_to_runs[cfg_tag].append(r)

print(f"Grouped runs for {len(cfg_to_runs)} configurations.")

# Read configs and build data rows
data = []

def get_scale_val(cfg_path):
    name = cfg_path.stem
    match = re.search(r"B12_(\d+)x_", name)
    if match:
        return int(match.group(1))
    return 0

all_cfg_files = sorted(list(cfg_dir.glob("B12_*.yaml")), key=get_scale_val)

for cfg_file in all_cfg_files:
    cfg_name = cfg_file.stem
    
    # Parser regex optimized for B12 structure
    npp_match = re.search(r"B12_(\d+x)_([RM])(\d+)_N1P21", cfg_name)
    if not npp_match:
        continue
    scale, embed_type, dim = npp_match.groups()
    n_idx, p1_idx, p2_idx = "1", "2", "1"

    # Read parameters from yaml configuration
    with open(cfg_file, "r") as yf:
        c = yaml.safe_load(yf)
        noise_rate_a = c['data']['asymmetric_noise_rate_a']
        ext_noise_ratio = c['data']['external_noise_ratio']
        noise_str = f"Asy:{noise_rate_a}/Ext:{ext_noise_ratio}"
        
        prior_str = f"Pri:{c['loss']['prior_type']}" if c['loss']['lambda_prior'] > 0 else "Pri:None"
        pred_str = f"Pre:{c['loss']['pred_loss']}" if c['loss']['lambda_pred'] > 0 else "Pre:None"
        
        noise_desc = noise_display_map.get(n_idx, f"Noise Regime {n_idx}")
        prior_desc = prior_display_map.get(p1_idx, f"Prior {p1_idx}")
        pred_desc = pred_display_map.get(p2_idx, f"Predictor {p2_idx}")

    # Merge train and eval summaries
    merged_summary = {}
    train_run = None
    eval_run = None
    
    cfg_runs = cfg_to_runs.get(cfg_name, [])
    if cfg_runs:
        for r in cfg_runs:
            if "eval" in r.tags:
                if eval_run is None or r.created_at > eval_run.created_at:
                    eval_run = r
            else:
                if train_run is None or r.created_at > train_run.created_at:
                    train_run = r
        
        # Merge metrics: train first, eval overwrites
        if train_run:
            merged_summary.update(dict(train_run.summary))
        if eval_run:
            merged_summary.update(dict(eval_run.summary))
            
    # Resolve metric values using aliases from registry, ordered by our importance list
    metrics_data = {}
    for alias in ordered_metrics_list:
        wandb_key = registry.get(alias)
        if not wandb_key:
            continue
        val = merged_summary.get(wandb_key)
        if isinstance(val, (int, float)):
            metrics_data[alias] = round(float(val), 4)
        else:
            metrics_data[alias] = None

    # Handle Plotly HTML Copy
    local_rel_path = ""
    run_url = ""
    state = "TODO"
    if train_run:
        state = train_run.state.upper()
        run_url = train_run.url
        html_path = run_id_to_latest_html.get(train_run.id)
        if html_path and html_path.exists():
            target_filename = f"{cfg_name}.html"
            target_path = assets_dir / target_filename
            try:
                shutil.copy2(str(html_path), str(target_path))
                local_rel_path = f"VISUALIZER_htmls/{target_filename}"
            except Exception as e:
                print(f"Error copying plot for {cfg_name}: {e}")
    
    row = {
        "config": cfg_name,
        "scale": scale,
        "embed_type": embed_type,
        "dim": dim,
        "n_idx": n_idx,
        "p1_idx": p1_idx,
        "p2_idx": p2_idx,
        "noise": noise_desc,
        "prior": prior_desc,
        "pred": pred_desc,
        "noise_str": noise_str,
        "prior_str": prior_str,
        "pred_str": pred_str,
        "state": state,
        "wandb_url": run_url,
        "local_path": local_rel_path,
        "metrics": metrics_data
    }
    data.append(row)

# Core metrics to show by default
default_metrics = [
    "diagonality_ratio",
    "cca_diag_score",
    "cca_rank",
    "clean_flatness_ratio_a",
    "clean_flatness_ratio_b"
]

# Map alias -> display label for checkboxes and headers
metric_names_display = {
    k: v[0] for k, v in metric_definitions.items()
}

# Add default display name for remaining registry keys
for k in ordered_metrics_list:
    if k not in metric_names_display:
        metric_names_display[k] = k.replace("_", " ").title()

# HTML generation with beautiful Git-style visualizer aesthetics
html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>B12 Data Scaling Dashboard</title>
    <!-- MathJax Setup for beautiful LaTeX equations -->
    <script>
        window.MathJax = {{
            tex: {{
                inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
            }}
        }};
    </script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            margin: 20px;
            background: #fafafa;
            color: #24292f;
        }}
        h1 {{
            font-size: 24px;
            border-bottom: 1px solid #d0d7de;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        h1 span {{
            color: #0969da;
        }}
        
        /* Clean Filters Block */
        .filters {{
            margin-bottom: 20px;
            padding: 15px;
            background: #fff;
            border: 1px solid #d0d7de;
            border-radius: 6px;
            position: sticky;
            top: 10px;
            z-index: 100;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}
        .filter-group {{
            display: inline-block;
            margin-right: 24px;
        }}
        .filters label {{
            font-weight: 600;
            margin-right: 8px;
            font-size: 14px;
            color: #57606a;
        }}
        .filters select {{
            padding: 6px 12px;
            border-radius: 6px;
            border: 1px solid #d0d7de;
            background-color: #f6f8fa;
            cursor: pointer;
            font-size: 13px;
        }}
        #stats {{
            font-weight: 600;
            color: #0969da;
            margin-left: 12px;
            font-size: 14px;
        }}

        /* Column Selector */
        .column-select-box {{
            margin-top: 12px;
            border-top: 1px solid #d0d7de;
            padding-top: 12px;
            font-size: 13px;
        }}
        .col-select-header {{
            font-weight: 600;
            font-size: 13px;
            cursor: pointer;
            color: #0969da;
            user-select: none;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .checkbox-grid {{
            display: none;
            margin-top: 10px;
            flex-wrap: wrap;
            gap: 12px 18px;
            border-top: 1px solid #e1e4e8;
            padding-top: 10px;
        }}
        .checkbox-label {{
            display: flex;
            align-items: center;
            gap: 6px;
            cursor: pointer;
            user-select: none;
            color: #24292f;
            font-weight: 500;
        }}
        .checkbox-label input {{
            cursor: pointer;
        }}

        /* Dictionary panel */
        .dictionary-box {{
            background: #f6f8fa;
            border: 1px solid #d0d7de;
            border-radius: 6px;
            padding: 12px;
            margin-bottom: 20px;
        }}
        .dict-header {{
            font-weight: 600;
            font-size: 14px;
            cursor: pointer;
            color: #0969da;
            user-select: none;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .dict-content {{
            display: none;
            margin-top: 10px;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 12px;
            border-top: 1px solid #d0d7de;
            padding-top: 10px;
        }}
        .dict-card {{
            background: #fff;
            border: 1px solid #d0d7de;
            border-radius: 4px;
            padding: 10px;
        }}
        .dict-card h4 {{
            margin: 0 0 4px 0;
            font-size: 13px;
            color: #24292f;
        }}
        .dict-card p {{
            font-size: 12px;
            color: #57606a;
            line-height: 1.4;
            margin: 0;
        }}
        .dict-card .math-formula {{
            font-family: monospace;
            font-size: 11px;
            background: #f6f8fa;
            padding: 2px 4px;
            border-radius: 3px;
            display: inline-block;
            margin-top: 6px;
            color: #24292f;
        }}

        /* Modal Popup Styling */
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            overflow: auto;
            background-color: rgba(0,0,0,0.5);
            backdrop-filter: blur(4px);
        }}
        .modal-content {{
            background-color: #fff;
            margin: 8% auto;
            padding: 24px;
            border: 1px solid #d0d7de;
            width: 70%;
            max-width: 800px;
            border-radius: 8px;
            box-shadow: 0 4px 24px rgba(0,0,0,0.15);
            position: relative;
            animation: modalFadeIn 0.25s ease-out;
        }}
        @keyframes modalFadeIn {{
            from {{ opacity: 0; transform: translateY(-20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .close-btn {{
            color: #57606a;
            position: absolute;
            right: 20px;
            top: 16px;
            font-size: 24px;
            font-weight: bold;
            cursor: pointer;
            user-select: none;
        }}
        .close-btn:hover {{
            color: #24292f;
        }}
        .modal-title {{
            font-size: 18px;
            font-weight: 600;
            color: #0969da;
            margin-bottom: 16px;
            border-bottom: 1px solid #d0d7de;
            padding-bottom: 8px;
        }}
        .modal-body {{
            font-size: 13px;
            line-height: 1.6;
            color: #24292f;
        }}
        .modal-body h5 {{
            margin: 16px 0 8px 0;
            font-size: 13px;
            font-weight: 600;
            color: #57606a;
        }}
        .modal-body ul {{
            padding-left: 20px;
            margin: 8px 0;
        }}
        .modal-body li {{
            margin-bottom: 6px;
        }}
        .modal-body .latex-block {{
            background: #f6f8fa;
            border: 1px solid #d0d7de;
            border-radius: 6px;
            padding: 12px;
            margin: 12px 0;
            overflow-x: auto;
            text-align: center;
        }}

        /* Table Style */
        table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            background: #fff;
            border: 1px solid #d0d7de;
            border-radius: 6px;
            overflow: hidden;
            margin-bottom: 20px;
        }}
        th, td {{
            border-bottom: 1px solid #d0d7de;
            padding: 12px;
            text-align: left;
            vertical-align: middle;
        }}
        th {{
            background: #f6f8fa;
            font-weight: 600;
            border-bottom: 2px solid #d0d7de;
            font-size: 13px;
            color: #57606a;
            cursor: pointer;
        }}
        tr:hover td {{
            background-color: #fafafa;
        }}
        
        th.col-config, td.col-config {{
            width: 1%;
            white-space: nowrap;
            font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
            font-size: 12px;
            font-weight: 600;
            background: #fdfdfd;
        }}
        .config-label {{
            font-size: 11px;
            color: #888;
            display: block;
            margin-top: 2px;
        }}
        .col-param {{
            width: 1%;
            white-space: nowrap;
            font-size: 13px;
            color: #24292f;
        }}
        th.col-metric, td.metric-cell {{
            width: 1%;
            white-space: nowrap;
            font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
            font-size: 13px;
            text-align: right;
            font-weight: 500;
        }}
        th.col-state, td.col-state, th.col-wandb, td.col-wandb {{
            width: 1%;
            white-space: nowrap;
        }}

        /* State Badges */
        .state-badge {{
            display: inline-block;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .state-badge.finished {{ background: #dafbe1; color: #1a7f37; border: 1px solid #ceead6; }}
        .state-badge.running {{ background: #ddf4ff; color: #0969da; border: 1px solid #c2e7ff; }}
        .state-badge.failed {{ background: #ffebe9; color: #cf222e; border: 1px solid #fec8c4; }}
        .state-badge.todo {{ background: #f6f8fa; color: #57606a; border: 1px solid #d0d7de; }}

        /* Link BADGE */
        .badge-link {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            background: #f6f8fa;
            border: 1px solid #d0d7de;
            color: #0969da;
            text-decoration: none;
            font-size: 11px;
            font-weight: 600;
        }}
        .badge-link:hover {{
            background: #0969da;
            color: #fff;
            border-color: #0969da;
        }}

        /* Visualization Embed Column */
        .embed-cell {{
            padding: 0;
            width: 80%;
        }}
        .embed-container {{
            width: 100%;
            height: 500px;
            resize: vertical;
            overflow: hidden;
            position: relative;
            border-left: 1px solid #d0d7de;
            background: #f6f8fa;
        }}
        iframe {{
            width: 100%;
            height: 100%;
            border: none;
        }}
        .placeholder {{
            color: #57606a;
            font-style: italic;
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            background: #f6f8fa;
            font-size: 13px;
        }}

        .footer-action {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            font-size: 13px;
            color: #57606a;
        }}
        .csv-btn {{
            background: #2da44e;
            border: 1px solid rgba(27,31,36,0.15);
            color: #fff;
            padding: 5px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
            font-size: 12px;
        }}
        .csv-btn:hover {{
            background: #2c974b;
        }}
    </style>
</head>
<body>
    <h1>B12 Data Scaling Alignment Study <span>Dashboard</span></h1>

    <!-- Dictionary Box (Closed by default) -->
    <div class="dictionary-box">
        <div class="dict-header" onclick="toggleDict()">
            <span>📖 Metric Definitions Dictionary (Click to collapse/expand)</span>
            <span id="dict-icon">▼</span>
        </div>
        <div class="dict-content" id="dict-content">
{dict_cards_str}
        </div>
    </div>

    <!-- Filters and Columns Box -->
    <div class="filters">
        <div>
            <div class="filter-group">
                <label for="filter-scale">Scale [S]:</label>
                <select id="filter-scale" onchange="applyFilters()">
                    <option value="all" selected>All Scales (Side-by-Side)</option>
                    <option value="1x">1x Scale (4k pts)</option>
                    <option value="2x">2x Scale (8k pts)</option>
                    <option value="4x">4x Scale (16k pts)</option>
                    <option value="8x">8x Scale (32k pts)</option>
                    <option value="16x">16x Scale (65k pts)</option>
                    <option value="32x">32x Scale (131k pts)</option>
                    <option value="64x">64x Scale (262k pts)</option>
                    <option value="128x">128x Scale (524k pts)</option>
                    <option value="256x">256x Scale (1048k pts)</option>
                </select>
            </div>
            <span id="stats">Showing 0 of 0 runs</span>
        </div>

        <!-- Visibility selectors box -->
        <div class="column-select-box">
            <div class="col-select-header" onclick="toggleColumns()">
                <span>🛠️ Columns Visibility Selector (Click to collapse/expand)</span>
                <span id="col-icon">▼</span>
            </div>
            <div class="checkbox-grid" id="checkbox-grid"></div>
        </div>
    </div>

    <!-- Table -->
    <table>
        <thead>
            <tr id="table-headers">
                <!-- Headers added dynamically -->
            </tr>
        </thead>
        <tbody id="table-body">
            <!-- Rows added dynamically -->
        </tbody>
    </table>

    <div class="footer-action">
        <span>B12 Sweep • Volumetric Dual-Alignment Study</span>
        <button class="csv-btn" onclick="exportCSV()">📥 Download Filtered CSV</button>
    </div>

    <script>
        const runData = {json.dumps(data)};
        const metricNamesDisplay = {json.dumps(metric_names_display)};
        const defaultCoreMetrics = {json.dumps(default_metrics)};
        const allMetrics = {json.dumps(ordered_metrics_list)};

        const METRIC_DIRECTIONS = {{
            // CCA & Retrieval
            diagonality_ratio: "higher",
            cca_diag_score: "higher",
            cca_rank: "higher",
            cca_dim0: "higher",
            cca_dim1: "higher",
            cca_dim2: "higher",
            "retrieval_cos@1": "higher",
            "retrieval_cos@5": "higher",
            "retrieval_l2@1": "higher",
            "retrieval_l2@5": "higher",

            // Flatness
            clean_flatness_ratio_a: "higher",
            clean_flatness_ratio_b: "higher",
            clean_orth_residual_a: "lower",
            clean_orth_residual_b: "lower",
            flatness_ratio_a: "higher",
            flatness_ratio_b: "higher",
            orth_residual_a: "lower",
            orth_residual_b: "lower",
            pca_axis_align_a: "higher",
            pca_axis_align_b: "higher",

            // R2 / Disentanglement
            r2_dim2_noise: "lower",
            r2_dim0_u1: "higher",
            r2_dim1_u2: "higher",
            r2_joint: "higher",
            r2_a: "higher",
            r2_b: "higher",
            r2_joint_u0: "higher",
            r2_joint_u1: "higher",
            r2_a_u0: "higher",
            r2_a_u1: "higher",
            r2_b_u0: "higher",
            r2_b_u1: "higher",

            // Losses & MSEs
            train_loss: "lower",
            val_loss: "lower",
            val_align_a2b: "lower",
            val_align_b2a: "lower",
            train_align_a2b: "lower",
            train_align_b2a: "lower",
            align_mse_a2b_manifold: "lower",
            align_mse_b2a_manifold: "lower",
            align_mse_a2b_asym_corrupt: "lower",
            align_mse_b2a_asym_corrupt: "lower",
            align_mse_a2b_external: "lower"
        }};

        // Sorting State Variables
        let currentSortCol = null;
        let currentSortDir = 'none';

        function handleSort(colId) {{
            if (currentSortCol === colId) {{
                if (currentSortDir === 'none') {{
                    currentSortDir = 'asc';
                }} else if (currentSortDir === 'asc') {{
                    currentSortDir = 'desc';
                }} else {{
                    currentSortDir = 'none';
                    currentSortCol = null;
                }}
            }} else {{
                currentSortCol = colId;
                currentSortDir = 'asc';
            }}
            renderTable();
        }}

        function getHeaderIcon(colId) {{
            if (currentSortCol === colId) {{
                if (currentSortDir === 'asc') return ' ▲';
                if (currentSortDir === 'desc') return ' ▼';
            }}
            return '';
        }}

        // Visibility Toggles
        const columnState = {{
            config: true,
            visualization: true,
            scale_col: false,
            embed_col: false,
            dim_col: false,
            noise_col: false,
            prior_col: false,
            pred_col: false,
            state: false,
            wandb: false
        }};

        // Set default metric columns as active
        allMetrics.forEach(m => {{
            columnState[m] = defaultCoreMetrics.includes(m);
        }});

        // Calculate global min/max for metrics
        const globalMinMax = {{}};
        allMetrics.forEach(m => {{
            let vals = runData.map(r => r.metrics[m]).filter(v => typeof v === 'number' && v !== null);
            if (vals.length > 0) {{
                globalMinMax[m] = {{
                    min: Math.min(...vals),
                    max: Math.max(...vals)
                }};
            }} else {{
                globalMinMax[m] = {{ min: 0, max: 1 }};
            }}
        }});

        // Render visibility options
        function renderCheckboxes() {{
            const grid = document.getElementById("checkbox-grid");
            grid.innerHTML = "";

            const helperColumns = [
                {{ id: "visualization", label: "Visualization [Plot]" }},
                {{ id: "scale_col", label: "Scale Column" }},
                {{ id: "embed_col", label: "Embed Type Column" }},
                {{ id: "dim_col", label: "Dimension Column" }},
                {{ id: "noise_col", label: "Noise Column" }},
                {{ id: "prior_col", label: "Prior Column" }},
                {{ id: "pred_col", label: "Predictor Column" }},
                {{ id: "state", label: "State" }},
                {{ id: "wandb", label: "WandB Link" }}
            ];

            helperColumns.forEach(c => {{
                grid.appendChild(createCheckbox(c.id, c.label));
            }});

            allMetrics.forEach(m => {{
                const label = metricNamesDisplay[m] || m;
                grid.appendChild(createCheckbox(m, label));
            }});
        }}

        function createCheckbox(id, labelText) {{
            const label = document.createElement("label");
            label.className = "checkbox-label";

            const cb = document.createElement("input");
            cb.type = "checkbox";
            cb.checked = columnState[id];
            cb.onchange = () => {{
                columnState[id] = cb.checked;
                renderTable();
            }};

            label.appendChild(cb);
            label.appendChild(document.createTextNode(labelText));
            return label;
        }}

        // Render dynamic table structure
        function renderTable() {{
            const headers = document.getElementById("table-headers");
            headers.innerHTML = "";

            // Config cell (always first)
            const thConfig = document.createElement("th");
            thConfig.textContent = "Config (N P1 P2)" + getHeaderIcon('config');
            thConfig.className = "col-config";
            thConfig.onclick = () => handleSort('config');
            headers.appendChild(thConfig);

            // Separate parameters cells
            if (columnState.scale_col) {{
                const th = document.createElement("th");
                th.textContent = "Scale" + getHeaderIcon('scale_col');
                th.className = "col-param";
                th.onclick = () => handleSort('scale_col');
                headers.appendChild(th);
            }}
            if (columnState.embed_col) {{
                const th = document.createElement("th");
                th.textContent = "Embed Type" + getHeaderIcon('embed_col');
                th.className = "col-param";
                th.onclick = () => handleSort('embed_col');
                headers.appendChild(th);
            }}
            if (columnState.dim_col) {{
                const th = document.createElement("th");
                th.textContent = "Dimension" + getHeaderIcon('dim_col');
                th.className = "col-param";
                th.onclick = () => handleSort('dim_col');
                headers.appendChild(th);
            }}
            if (columnState.noise_col) {{
                const th = document.createElement("th");
                th.textContent = "Noise" + getHeaderIcon('noise_col');
                th.className = "col-param";
                th.onclick = () => handleSort('noise_col');
                headers.appendChild(th);
            }}
            if (columnState.prior_col) {{
                const th = document.createElement("th");
                th.textContent = "Prior" + getHeaderIcon('prior_col');
                th.className = "col-param";
                th.onclick = () => handleSort('prior_col');
                headers.appendChild(th);
            }}
            if (columnState.pred_col) {{
                const th = document.createElement("th");
                th.textContent = "Pred" + getHeaderIcon('pred_col');
                th.className = "col-param";
                th.onclick = () => handleSort('pred_col');
                headers.appendChild(th);
            }}

            // Metrics cells
            allMetrics.forEach(m => {{
                if (columnState[m]) {{
                    const th = document.createElement("th");
                    th.className = "col-metric";
                    th.textContent = (metricNamesDisplay[m] || m) + getHeaderIcon(m);
                    th.style.textAlign = "right";
                    th.onclick = () => handleSort(m);
                    headers.appendChild(th);
                }}
            }});

            // State & WandB
            if (columnState.state) {{
                const th = document.createElement("th");
                th.textContent = "State" + getHeaderIcon('state');
                th.className = "col-state";
                th.onclick = () => handleSort('state');
                headers.appendChild(th);
            }}
            if (columnState.wandb) {{
                const th = document.createElement("th");
                th.textContent = "WandB" + getHeaderIcon('wandb');
                th.className = "col-wandb";
                th.onclick = () => handleSort('wandb');
                headers.appendChild(th);
            }}

            // Inline Plot Cell
            if (columnState.visualization) {{
                const th = document.createElement("th");
                th.textContent = "Visualization";
                th.className = "embed-cell";
                headers.appendChild(th);
            }}

            applyFilters();
        }}

        function applyFilters() {{
            const fScale = document.getElementById("filter-scale").value;

            const tbody = document.getElementById("table-body");
            tbody.innerHTML = "";

            // Filter rows
            let filteredRows = runData.filter(row => {{
                const matchScale = (fScale === 'all' || row.scale === fScale);
                return matchScale;
            }});

            // Sort filtered rows if active
            if (currentSortCol && currentSortDir !== 'none') {{
                filteredRows.sort((a, b) => {{
                    let valA, valB;
                    if (currentSortCol === 'config') {{
                        valA = a.config;
                        valB = b.config;
                    }} else if (currentSortCol === 'scale_col') {{
                        valA = parseFloat(a.scale.replace('x', ''));
                        valB = parseFloat(b.scale.replace('x', ''));
                    }} else if (currentSortCol === 'embed_col') {{
                        valA = a.embed_type;
                        valB = b.embed_type;
                    }} else if (currentSortCol === 'dim_col') {{
                        valA = parseInt(a.dim);
                        valB = parseInt(b.dim);
                    }} else if (currentSortCol === 'noise_col') {{
                        valA = a.noise;
                        valB = b.noise;
                    }} else if (currentSortCol === 'prior_col') {{
                        valA = a.prior;
                        valB = b.prior;
                    }} else if (currentSortCol === 'pred_col') {{
                        valA = a.pred;
                        valB = b.pred;
                    }} else if (currentSortCol === 'state') {{
                        valA = a.state;
                        valB = b.state;
                    }} else if (currentSortCol === 'wandb') {{
                        valA = a.wandb_url || '';
                        valB = b.wandb_url || '';
                    }} else {{
                        valA = a.metrics[currentSortCol];
                        valB = b.metrics[currentSortCol];
                    }}

                    if (valA === null || valA === undefined) return 1;
                    if (valB === null || valB === undefined) return -1;

                    if (valA < valB) return currentSortDir === 'asc' ? -1 : 1;
                    if (valA > valB) return currentSortDir === 'asc' ? 1 : -1;
                    return 0;
                }});
            }}

            filteredRows.forEach(row => {{
                tbody.appendChild(createRow(row));
            }});

            document.getElementById("stats").textContent = `Showing ${{filteredRows.length}} of ${{runData.length}} runs`;
        }}

        function createRow(row) {{
            const tr = document.createElement("tr");

            // Config Cell
            const tdConfig = document.createElement("td");
            tdConfig.className = "col-config";
            tdConfig.innerHTML = `${{row.config}}<span class="config-label">N:${{row.n_idx}} P1:${{row.p1_idx}} P2:${{row.p2_idx}}</span>
            <span style="font-size: 11px; color: #57606a; display: block; margin-top: 5px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; font-weight: normal; line-height: 1.45; white-space: nowrap;">
                <strong>Noise:</strong> ${{row.noise_str}}<br>
                <strong>Pri:</strong> ${{row.prior_str.split(':')[1]}}<br>
                <strong>Pred:</strong> ${{row.pred_str.split(':')[1]}}
            </span>`;
            tr.appendChild(tdConfig);

            // Separate parameters cells
            if (columnState.scale_col) {{
                const td = document.createElement("td");
                td.className = "col-param";
                td.textContent = row.scale;
                tr.appendChild(td);
            }}
            if (columnState.embed_col) {{
                const td = document.createElement("td");
                td.className = "col-param";
                td.textContent = row.embed_type === 'R' ? 'Orthogonal Rotation' : 'Random MLP';
                tr.appendChild(td);
            }}
            if (columnState.dim_col) {{
                const td = document.createElement("td");
                td.className = "col-param";
                td.textContent = row.dim + ' Dimensions';
                tr.appendChild(td);
            }}
            if (columnState.noise_col) {{
                const td = document.createElement("td");
                td.className = "col-param";
                td.textContent = row.noise;
                tr.appendChild(td);
            }}
            if (columnState.prior_col) {{
                const td = document.createElement("td");
                td.className = "col-param";
                td.textContent = row.prior;
                tr.appendChild(td);
            }}
            if (columnState.pred_col) {{
                const td = document.createElement("td");
                td.className = "col-param";
                td.textContent = row.pred;
                tr.appendChild(td);
            }}

            // Metrics Cells
            allMetrics.forEach(m => {{
                if (columnState[m]) {{
                    const td = document.createElement("td");
                    td.className = "metric-cell";
                    const val = row.metrics[m];

                    if (val === null || val === undefined) {{
                        td.textContent = "N/A";
                    }} else {{
                        td.textContent = val;

                        // Background mapping HSL
                        const stats = globalMinMax[m];
                        let ratio = (val - stats.min) / (stats.max - stats.min || 1);
                        
                        const isHigherBetter = METRIC_DIRECTIONS[m] !== "lower";
                        if (!isHigherBetter) {{
                            ratio = 1.0 - ratio;
                        }}

                        if (ratio >= 0.5) {{
                            const intensity = (ratio - 0.5) * 2;
                            td.style.backgroundColor = `rgba(46, 160, 67, ${{intensity * 0.16}})`;
                        }} else {{
                            const intensity = (0.5 - ratio) * 2;
                            td.style.backgroundColor = `rgba(248, 81, 73, ${{intensity * 0.16}})`;
                        }}
                    }}
                    tr.appendChild(td);
                }}
            }});

            // State Cell
            if (columnState.state) {{
                const td = document.createElement("td");
                td.className = "col-state";
                const badge = document.createElement("span");
                badge.className = `state-badge ${{row.state.toLowerCase()}}`;
                badge.textContent = row.state;
                td.appendChild(badge);
                tr.appendChild(td);
            }}

            // WandB Cell
            if (columnState.wandb) {{
                const td = document.createElement("td");
                td.className = "col-wandb";
                if (row.wandb_url) {{
                    const a = document.createElement("a");
                    a.className = "badge-link";
                    a.href = row.wandb_url;
                    a.target = "_blank";
                    a.textContent = "W&B 🚀";
                    td.appendChild(a);
                }} else {{
                    td.textContent = "-";
                }}
                tr.appendChild(td);
            }}

            // Visualization Inline Cell
            if (columnState.visualization) {{
                const td = document.createElement("td");
                td.className = "embed-cell";

                const container = document.createElement("div");
                container.className = "embed-container";

                if (row.local_path) {{
                    const iframe = document.createElement("iframe");
                    iframe.src = row.local_path;
                    iframe.loading = "lazy";
                    container.appendChild(iframe);
                }} else {{
                    const placeholder = document.createElement("div");
                    placeholder.className = "placeholder";
                    placeholder.textContent = "Interactive Plot HTML not available";
                    container.appendChild(placeholder);
                }}

                td.appendChild(container);
                tr.appendChild(td);
            }}

            return tr;
        }}

        // Collapse Dictionary toggle
        function toggleDict() {{
            const content = document.getElementById("dict-content");
            const icon = document.getElementById("dict-icon");
            if (content.style.display === "grid") {{
                content.style.display = "none";
                icon.textContent = "▼";
            }} else {{
                content.style.display = "grid";
                icon.textContent = "▲";
            }}
        }}

        // Collapse Column Selection toggle
        function toggleColumns() {{
            const content = document.getElementById("checkbox-grid");
            const icon = document.getElementById("col-icon");
            if (content.style.display === "flex") {{
                content.style.display = "none";
                icon.textContent = "▼";
            }} else {{
                content.style.display = "flex";
                icon.textContent = "▲";
            }}
        }}

        // Filtered CSV Exporter
        function exportCSV() {{
            const fScale = document.getElementById("filter-scale").value;

            const csvHeaders = ["Config", "Scale", "Embed Type", "Dimension", "State"];
            allMetrics.forEach(m => csvHeaders.push(m));

            const rows = [];
            runData.forEach(row => {{
                const matchScale = (fScale === 'all' || row.scale === fScale);

                if (matchScale) {{
                    const rowData = [
                        row.config,
                        row.scale,
                        row.embed_type === 'R' ? 'Orthogonal Rotation' : 'Random MLP',
                        row.dim + ' Dimensions',
                        row.state
                    ];
                    allMetrics.forEach(m => {{
                        const v = row.metrics[m];
                        rowData.push(v === null ? "N/A" : v);
                    }});
                    rows.push(rowData.join(","));
                }}
            }});

            const csvContent = "data:text/csv;charset=utf-8," 
                + [csvHeaders.join(",")].concat(rows).join("\\n");
            
            const encodedUri = encodeURI(csvContent);
            const link = document.createElement("a");
            link.setAttribute("href", encodedUri);
            link.setAttribute("download", `B12_Sweep_metrics_${{fScale.replace(/\\s+/g, '_')}}.csv`);
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }}

        const metricDetails = {json.dumps(metric_details)};

        function openModal(metricId) {{
            const modal = document.getElementById("metric-modal");
            const title = document.getElementById("modal-title");
            const body = document.getElementById("modal-body");

            const details = metricDetails[metricId];
            if (details) {{
                title.textContent = metricNamesDisplay[metricId] || metricId;
                body.innerHTML = details;
                modal.style.display = "block";
                
                if (window.MathJax && window.MathJax.typesetPromise) {{
                    window.MathJax.typesetPromise([body]).catch(err => console.log(err));
                }}
            }}
        }}

        function closeModal() {{
            document.getElementById("metric-modal").style.display = "none";
        }}

        window.onclick = function(event) {{
            const modal = document.getElementById("metric-modal");
            if (event.target === modal) {{
                modal.style.display = "none";
            }}
        }}

        document.addEventListener("DOMContentLoaded", () => {{
            renderCheckboxes();
            renderTable();
        }});
    </script>
    <!-- Modal for detailed LaTeX formulas -->
    <div id="metric-modal" class="modal">
        <div class="modal-content">
            <span class="close-btn" onclick="closeModal()">&close;</span>
            <div class="modal-title" id="modal-title">Metric Details</div>
            <div class="modal-body" id="modal-body"></div>
        </div>
    </div>
</body>
</html>
"""

with open(output_file, "w") as f:
    f.write(html_content)

print(f"Successfully generated visualizer dashboard at: {output_file}")
