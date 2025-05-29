from typing import Callable, Union, Optional, Literal
from functools import partial

import torch
from torch.utils.data import DataLoader
from torch import Tensor
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from tqdm import tqdm
from einops import einsum

from eap.graph import Graph
from eap.utils import tokenize_plus, compute_mean_activations, load_ablations
from eap.evaluate import evaluate_baseline, evaluate_graph
from eap.attribute_node import make_hooks_and_matrices


def get_scores_ig_activations_directional(
    model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], 
    intervention: Literal['patching', 'zero', 'mean','mean-positional', 'optimal']='patching', steps=30, 
    intervention_dataloader: Optional[DataLoader]=None, optimal_ablation_path: Optional[str] = None,
    quiet:bool=False, neuron:bool=False, 
    patch_direction: Literal['patch-in-corrupt', 'patch-in-clean']='patch-in-corrupt'):
    # Patch direction 'patch-in-corrupt' means that we patch the clean run with the corrupted activations,
    # while 'patch-in-clean' means that we patch the corrupted run with the clean activations.

    if 'mean' in intervention:
        assert intervention_dataloader is not None, "Intervention dataloader must be provided for mean interventions"
        per_position = 'positional' in intervention
        means = compute_mean_activations(model, graph, intervention_dataloader, per_position=per_position)
        means = means.unsqueeze(0)
        if not per_position:
            means = means.unsqueeze(0)

    elif intervention == 'optimal':
        assert optimal_ablation_path is not None, "Path to pre-computed activations must be provided for optimal ablations"
        optimal_ablations = load_ablations(model, graph, optimal_ablation_path)
        optimal_ablations = optimal_ablations.unsqueeze(0).unsqueeze(0)

    if neuron:
        scores = torch.zeros((graph.n_forward, graph.cfg.d_model), device='cuda', dtype=model.cfg.dtype)    
    else:
        scores = torch.zeros((graph.n_forward), device='cuda', dtype=model.cfg.dtype)    
    
    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size

        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, _, _, _ = tokenize_plus(model, corrupted)

        (_, _, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores, neuron=neuron)
        (fwd_hooks_corrupted, _, _), activations_corrupted = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores, neuron=neuron)
        (fwd_hooks_clean, _, _), activations_clean = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores, neuron=neuron)

        if intervention == 'patching':
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                # _ = model(corrupted_tokens, attention_mask=attention_mask)
                # Note that these hooks already multiply the scores by the activation difference
                if patch_direction == 'patch-in-corrupt':
                    _ = model(corrupted_tokens, attention_mask=attention_mask)
                else:
                    _ = model(clean_tokens, attention_mask=attention_mask)

        elif 'mean' in intervention:
            activation_difference += means

        elif intervention == 'optimal':
                activation_difference += optimal_ablations

        with model.hooks(fwd_hooks=fwd_hooks_clean):
            # clean_logits = model(clean_tokens, attention_mask=attention_mask)
            if patch_direction == 'patch-in-corrupt':
                clean_logits = model(clean_tokens, attention_mask=attention_mask)
            else:
                clean_logits = model(corrupted_tokens, attention_mask=attention_mask)
            
        activation_difference += activations_corrupted.clone().detach() - activations_clean.clone().detach()

        def output_interpolation_hook(k: int, clean: torch.Tensor, corrupted: torch.Tensor):
            def hook_fn(activations: torch.Tensor, hook):
                alpha = k/steps
                new_output = alpha * clean + (1 - alpha) * corrupted + activations * 0
                return new_output
            return hook_fn

        total_steps = 0

        nodeslist = [graph.nodes['input']]
        for layer in range(graph.cfg['n_layers']):
            nodeslist.append(graph.nodes[f'a{layer}.h0'])
            nodeslist.append(graph.nodes[f'm{layer}'])

        for node in nodeslist:
            for step in range(1, steps+1):
                total_steps += 1
                
                # if patch_direction == 'patch-in-corrupt':
                #     clean_acts = activations_clean[:, :, graph.forward_index(node)]
                #     corrupted_acts = activations_corrupted[:, :, graph.forward_index(node)]
                # else:
                #     clean_acts = activations_corrupted[:, :, graph.forward_index(node)]
                #     corrupted_acts = activations_clean[:, :, graph.forward_index(node)]

                clean_acts = activations_clean[:, :, graph.forward_index(node)]
                corrupted_acts = activations_corrupted[:, :, graph.forward_index(node)]
                
                fwd_hooks = [(node.out_hook, output_interpolation_hook(step, clean_acts, corrupted_acts))]

                with model.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks):
                    if patch_direction == 'patch-in-corrupt':
                        logits = model(clean_tokens, attention_mask=attention_mask)
                    else:
                        logits = model(corrupted_tokens, attention_mask=attention_mask)

                    metric_value = metric(logits, clean_logits, input_lengths, label)

                    metric_value.backward(retain_graph=True)

    scores /= total_items
    scores /= total_steps

    return scores


def asymmetry_score(corrupt_to_clean: Tensor, clean_to_corrupt: Tensor):
    assert corrupt_to_clean.shape == clean_to_corrupt.shape, \
        f"Cannot calculate asymmetry between matrices of different shapes, {corrupt_to_clean.shape} and {clean_to_corrupt.shape}"

    # Expect attribution scores in opposite directions to cancel out
    rem_dims = tuple(range(1, len(corrupt_to_clean.shape)))
    max_scores = torch.amax((corrupt_to_clean + clean_to_corrupt), dim=rem_dims, keepdim=True)
    return torch.div((corrupt_to_clean + clean_to_corrupt), max_scores)


allowed_aggregations = {'sum', 'mean'}      

def custom_attribute_node(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], 
                   intervention: Literal['patching', 'zero', 'mean','mean-positional', 'optimal']='patching', 
                   aggregation='sum', ig_steps: Optional[int]=None, intervention_dataloader: Optional[DataLoader]=None, 
                   optimal_ablation_path: Optional[str] = None, quiet:bool=False, neuron:bool=False):
    assert model.cfg.use_attn_result, "Model must be configured to use attention result (model.cfg.use_attn_result)"
    assert model.cfg.use_split_qkv_input, "Model must be configured to use split qkv inputs (model.cfg.use_split_qkv_input)"
    assert model.cfg.use_hook_mlp_in, "Model must be configured to use hook MLP in (model.cfg.use_hook_mlp_in)"
    if model.cfg.n_key_value_heads is not None:
        assert model.cfg.ungroup_grouped_query_attention, "Model must be configured to ungroup grouped attention (model.cfg.ungroup_grouped_attention)"
    
    if aggregation not in allowed_aggregations:
        raise ValueError(f'aggregation must be in {allowed_aggregations}, but got {aggregation}')
        
    # Scores are by default summed across the d_model dimension
    # This means that scores are a [n_src_nodes, n_dst_nodes] tensor

    # Run Integrated Gradients in both directions
    corrupt_to_clean_scores = get_scores_ig_activations_directional(
        model, graph, dataloader, metric, intervention=intervention, 
        intervention_dataloader=intervention_dataloader, 
        optimal_ablation_path=optimal_ablation_path, quiet=quiet,
        neuron=neuron, patch_direction='patch-in-corrupt')

    clean_to_corrupt_scores = get_scores_ig_activations_directional(
        model, graph, dataloader, metric, intervention=intervention, 
        intervention_dataloader=intervention_dataloader, 
        optimal_ablation_path=optimal_ablation_path, quiet=quiet,
        neuron=neuron, patch_direction='patch-in-clean')

    # Identify top 10% of components in which scores differ between the two attribution directions
    scores_asymmetry = asymmetry_score(corrupt_to_clean_scores, clean_to_corrupt_scores)
    abs_scores_asymmetry = scores_asymmetry.abs()
    threshold = torch.quantile(abs_scores_asymmetry.flatten(), 0.9)

    latent_components = abs_scores_asymmetry >= threshold
    latent_components_indices = latent_components.nonzero()

    # Corrupt -> clean IG scores are negatively correlated with corrupt->clean AP scores.
    # Note that changing the signage is different from the first run!
    scores = -corrupt_to_clean_scores.clone()

    ### COMPLETE CIRCUIT IMPLEMENTATION
    # Use the attribution score with the greatest magnitude for each latent component

    # magnitude_mask = clean_to_corrupt_scores.abs() >= scores.abs()
    # combined_mask = latent_components.bool() & magnitude_mask
    # scores[combined_mask] = clean_to_corrupt_scores[combined_mask]

    ### AVERAGE IMPLEMENTATION

    # # Use the attribution scores for patching from clean to corrupt for latent components
    # # Use the average between the two directions, to account for latent components being not as important
    # # Note that scores are in opposite directions, so we subtract instead of adding

    scores[latent_components.bool()] += clean_to_corrupt_scores[latent_components.bool()]
    scores[latent_components.bool()] /= 2

    # if neuron:
    #     for n, d in latent_components_indices:
    #         scores[n, d] += clean_to_corrupt_scores[n, d]
    #         scores[n, d] /= 2
    # else:
    #     for n in latent_components_indices:
    #         scores[n] += clean_to_corrupt_scores[n]
    #         scores[n] /= 2

    if aggregation == 'mean':
        scores /= model.cfg.d_model
        
    if neuron:
        graph.neurons_scores[:] = scores.to(graph.scores.device) # (n_forward, d_model)
    else:
        graph.nodes_scores[:] = scores.to(graph.scores.device) # (n_forward,)


        
