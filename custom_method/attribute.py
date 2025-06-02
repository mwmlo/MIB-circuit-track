from typing import Callable, List, Union, Optional, Literal, Tuple
from functools import partial

import torch
from torch.utils.data import DataLoader
from torch import Tensor
from transformer_lens import HookedTransformer

from tqdm import tqdm

from eap.graph import Graph
from eap.utils import tokenize_plus, compute_mean_activations, load_ablations, make_hooks_and_matrices
from eap.evaluate import evaluate_baseline, evaluate_graph

from .utils import asymmetry_score

def get_scores_eap_directional_edge(
    model: HookedTransformer, graph: Graph, dataloader:DataLoader, metric: Callable[[Tensor], Tensor], 
    intervention: Literal['patching', 'zero', 'mean','mean-positional', 'optimal']='patching', 
    intervention_dataloader: Optional[DataLoader]=None, optimal_ablation_path: Optional[str] = None, quiet=False,
    patch_direction: Literal['patch-in-corrupt', 'patch-in-clean']='patch-in-corrupt'
):
    """Gets edge attribution scores using EAP.

    Args:
        model (HookedTransformer): The model to attribute
        graph (Graph): Graph to attribute
        dataloader (DataLoader): The data over which to attribute
        metric (Callable[[Tensor], Tensor]): metric to attribute with respect to
        quiet (bool, optional): suppress tqdm output. Defaults to False.

    Returns:
        Tensor: a [src_nodes, dst_nodes] tensor of scores for each edge
    """
    scores = torch.zeros((graph.n_forward, graph.n_backward), device='cuda', dtype=model.cfg.dtype)    

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
    
    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:

        if patch_direction == 'patch-in-clean':
            # In this case, we will patch the corrupted run with the clean activations
            clean, corrupted = corrupted, clean
        
        batch_size = len(clean)
        total_items += batch_size
        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, _, _, _ = tokenize_plus(model, corrupted)

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)

        with torch.inference_mode():
            if intervention == 'patching':
                # We intervene by subtracting out clean and adding in corrupted activations
                with model.hooks(fwd_hooks_corrupted):
                    _ = model(corrupted_tokens, attention_mask=attention_mask)
            elif 'mean' in intervention:
                # In the case of zero or mean ablation, we skip the adding in corrupted activations
                # but in mean ablations, we need to add the mean in
                activation_difference += means

            elif intervention == 'optimal':
                activation_difference += optimal_ablations

            # For some metrics (e.g. accuracy or KL), we need the clean logits
            clean_logits = model(clean_tokens, attention_mask=attention_mask)

        with model.hooks(fwd_hooks=fwd_hooks_clean, bwd_hooks=bwd_hooks):
            logits = model(clean_tokens, attention_mask=attention_mask)
            metric_value = metric(logits, clean_logits, input_lengths, label)
            metric_value.backward()

    scores /= total_items

    return scores


def get_scores_inputs_ig_directional_edge(
    model: HookedTransformer, graph: Graph, dataloader: DataLoader, 
    metric: Callable[[Tensor], Tensor], steps=30, quiet=False,
    patch_direction: Literal['patch-in-corrupt', 'patch-in-clean']='patch-in-corrupt'
):
    """Gets edge attribution scores using EAP with integrated gradients.

    Args:
        model (HookedTransformer): The model to attribute
        graph (Graph): Graph to attribute
        dataloader (DataLoader): The data over which to attribute
        metric (Callable[[Tensor], Tensor]): metric to attribute with respect to
        steps (int, optional): number of IG steps. Defaults to 30.
        quiet (bool, optional): suppress tqdm output. Defaults to False.

    Returns:
        Tensor: a [src_nodes, dst_nodes] tensor of scores for each edge
    """
    scores = torch.zeros((graph.n_forward, graph.n_backward), device='cuda', dtype=model.cfg.dtype)    
    
    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:

        if patch_direction == 'patch-in-clean':
            # In this case, we will patch the corrupted run with the clean activations
            clean, corrupted = corrupted, clean
        
        batch_size = len(clean)
        total_items += batch_size
        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, _, _, n_pos_corrupted = tokenize_plus(model, corrupted)

        if n_pos != n_pos_corrupted:
            print(f"Number of positions must match, but do not: {n_pos} (clean) != {n_pos_corrupted} (corrupted)")
            print(clean)
            print(corrupted)
            raise ValueError("Number of positions must match")

        # Here, we get our fwd / bwd hooks and the activation difference matrix
        # The forward corrupted hooks add the corrupted activations to the activation difference matrix
        # The forward clean hooks subtract the clean activations 
        # The backward hooks get the gradient, and use that, plus the activation difference, for the scores
        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)

        with torch.inference_mode():
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                _ = model(corrupted_tokens, attention_mask=attention_mask)

            input_activations_corrupted = activation_difference[:, :, graph.forward_index(graph.nodes['input'])].clone()

            with model.hooks(fwd_hooks=fwd_hooks_clean):
                clean_logits = model(clean_tokens, attention_mask=attention_mask)

            input_activations_clean = input_activations_corrupted - activation_difference[:, :, graph.forward_index(graph.nodes['input'])]

        def input_interpolation_hook(k: int):
            def hook_fn(activations, hook):
                new_input = input_activations_corrupted + (k / steps) * (input_activations_clean - input_activations_corrupted) 
                new_input.requires_grad = True 
                return new_input
            return hook_fn

        total_steps = 0
        for step in range(0, steps):
            total_steps += 1
            with model.hooks(fwd_hooks=[(graph.nodes['input'].out_hook, input_interpolation_hook(step))], bwd_hooks=bwd_hooks):
                logits = model(clean_tokens, attention_mask=attention_mask)
                metric_value = metric(logits, clean_logits, input_lengths, label)
                if torch.isnan(metric_value).any().item():
                    print("Metric value is NaN")
                    print(f"Clean: {clean}")
                    print(f"Corrupted: {corrupted}")
                    print(f"Label: {label}")
                    print(f"Metric: {metric}")
                    raise ValueError("Metric value is NaN")
                metric_value.backward()
            
            if torch.isnan(scores).any().item():
                print("Metric value is NaN")
                print(f"Clean: {clean}")
                print(f"Corrupted: {corrupted}")
                print(f"Label: {label}")
                print(f"Metric: {metric}")
                print(f'Step: {step}')
                raise ValueError("Metric value is NaN")

    scores /= total_items
    scores /= total_steps

    return scores


def get_scores_ig_activations_directional_edge(
    model: HookedTransformer, graph: Graph, dataloader: DataLoader, 
    metric: Callable[[Tensor], Tensor], intervention: Literal['patching', 'zero', 'mean','mean-positional', 'optimal']='patching', 
    steps=30, intervention_dataloader: Optional[DataLoader]=None, optimal_ablation_path: Optional[str] = None, quiet=False,
    patch_direction: Literal['patch-in-corrupt', 'patch-in-clean']='patch-in-corrupt',
):
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

    scores = torch.zeros((graph.n_forward, graph.n_backward), device='cuda', dtype=model.cfg.dtype)    
    
    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size

        if patch_direction == 'patch-in-clean':
            # In this case, we will patch the corrupted activations with the clean ones
            clean, corrupted = corrupted, clean

        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, _, _, _ = tokenize_plus(model, corrupted)

        (_, _, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)
        (fwd_hooks_corrupted, _, _), activations_corrupted = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)
        (fwd_hooks_clean, _, _), activations_clean = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)

        if intervention == 'patching':
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                _ = model(corrupted_tokens, attention_mask=attention_mask)

        elif 'mean' in intervention:
            activation_difference += means

        elif intervention == 'optimal':
            activation_difference += optimal_ablations

        with model.hooks(fwd_hooks=fwd_hooks_clean):
            clean_logits = model(clean_tokens, attention_mask=attention_mask)
            activation_difference += activations_corrupted.clone().detach() - activations_clean.clone().detach()

        def output_interpolation_hook(k: int, clean: torch.Tensor, corrupted: torch.Tensor):
            def hook_fn(activations: torch.Tensor, hook):
                alpha = k/steps
                new_output = alpha * clean + (1 - alpha) * corrupted
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
                
                clean_acts = activations_clean[:, :, graph.forward_index(node)]
                corrupted_acts = activations_corrupted[:, :, graph.forward_index(node)]
                fwd_hooks = [(node.out_hook, output_interpolation_hook(step, clean_acts, corrupted_acts))]

                with model.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks):
                    logits = model(clean_tokens, attention_mask=attention_mask)
                    metric_value = metric(logits, clean_logits, input_lengths, label)

                    metric_value.backward(retain_graph=True)

    scores /= total_items
    scores /= total_steps

    return scores


allowed_aggregations = {'sum', 'mean'}    
def custom_attribute_edge(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor],
              intervention: Literal['patching', 'zero', 'mean','mean-positional', 'optimal']='patching', aggregation='sum', 
              ig_steps: Optional[int]=None, intervention_dataloader: Optional[DataLoader]=None, 
              optimal_ablation_path: Optional[str]=None, quiet=False):
    assert model.cfg.use_attn_result, "Model must be configured to use attention result (model.cfg.use_attn_result)"
    assert model.cfg.use_split_qkv_input, "Model must be configured to use split qkv inputs (model.cfg.use_split_qkv_input)"
    assert model.cfg.use_hook_mlp_in, "Model must be configured to use hook MLP in (model.cfg.use_hook_mlp_in)"
    if model.cfg.n_key_value_heads is not None:
        assert model.cfg.ungroup_grouped_query_attention, "Model must be configured to ungroup grouped attention (model.cfg.ungroup_grouped_attention)"
    
    if aggregation not in allowed_aggregations:
        raise ValueError(f'aggregation must be in {allowed_aggregations}, but got {aggregation}')
        
    # Scores are by default summed across the d_model dimension
    # This means that scores are a [n_src_nodes, n_dst_nodes] tensor

    corrupt_to_clean_scores = get_scores_ig_activations_directional_edge(
        model, graph, dataloader, metric, intervention=intervention, 
        intervention_dataloader=intervention_dataloader, 
        optimal_ablation_path=optimal_ablation_path, quiet=quiet,
        patch_direction='patch-in-corrupt')

    clean_to_corrupt_scores = get_scores_ig_activations_directional_edge(
        model, graph, dataloader, metric, intervention=intervention, 
        intervention_dataloader=intervention_dataloader, 
        optimal_ablation_path=optimal_ablation_path, quiet=quiet,
        patch_direction='patch-in-clean')

    # Identify top 10% of components in which scores differ between the two attribution directions
    scores_asymmetry = asymmetry_score(corrupt_to_clean_scores, clean_to_corrupt_scores)
    abs_scores_asymmetry = scores_asymmetry.abs()
    threshold = torch.quantile(abs_scores_asymmetry.flatten(), 0.9)

    latent_components = abs_scores_asymmetry >= threshold
    latent_components_indices = latent_components.nonzero()

    scores = corrupt_to_clean_scores.clone()

    scores[latent_components.bool()] -= clean_to_corrupt_scores[latent_components.bool()]
    scores[latent_components.bool()] /= 2

    if aggregation == 'mean':
        scores /= model.cfg.d_model
        
    graph.scores[:] =  scores.to(graph.scores.device)


def custom_attribute_inputs_edge(
    model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor],
    intervention: Literal['patching', 'zero', 'mean','mean-positional', 'optimal']='patching', aggregation='sum', 
    ig_steps: Optional[int]=None, intervention_dataloader: Optional[DataLoader]=None, 
    optimal_ablation_path: Optional[str]=None, quiet=False
):
    assert model.cfg.use_attn_result, "Model must be configured to use attention result (model.cfg.use_attn_result)"
    assert model.cfg.use_split_qkv_input, "Model must be configured to use split qkv inputs (model.cfg.use_split_qkv_input)"
    assert model.cfg.use_hook_mlp_in, "Model must be configured to use hook MLP in (model.cfg.use_hook_mlp_in)"
    if model.cfg.n_key_value_heads is not None:
        assert model.cfg.ungroup_grouped_query_attention, "Model must be configured to ungroup grouped attention (model.cfg.ungroup_grouped_attention)"
    
    if aggregation not in allowed_aggregations:
        raise ValueError(f'aggregation must be in {allowed_aggregations}, but got {aggregation}')
        
    # Scores are by default summed across the d_model dimension
    # This means that scores are a [n_src_nodes, n_dst_nodes] tensor

    corrupt_to_clean_scores = get_scores_inputs_ig_directional_edge(
        model, graph, dataloader, metric, quiet=quiet,
        patch_direction='patch-in-corrupt')

    clean_to_corrupt_scores = get_scores_ig_activations_directional_edge(
        model, graph, dataloader, metric, quiet=quiet,
        patch_direction='patch-in-clean')

    # Identify top 10% of components in which scores differ between the two attribution directions
    scores_asymmetry = asymmetry_score(corrupt_to_clean_scores, clean_to_corrupt_scores)
    abs_scores_asymmetry = scores_asymmetry.abs()
    threshold = torch.quantile(abs_scores_asymmetry.flatten(), 0.9)

    latent_components = abs_scores_asymmetry >= threshold
    latent_components_indices = latent_components.nonzero()

    scores = corrupt_to_clean_scores.clone()

    scores[latent_components.bool()] -= clean_to_corrupt_scores[latent_components.bool()]
    scores[latent_components.bool()] /= 2

    if aggregation == 'mean':
        scores /= model.cfg.d_model
        
    graph.scores[:] =  scores.to(graph.scores.device)


def custom_attribute_eap_edge(
    model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor],
    intervention: Literal['patching', 'zero', 'mean','mean-positional', 'optimal']='patching', aggregation='sum', 
    ig_steps: Optional[int]=None, intervention_dataloader: Optional[DataLoader]=None, 
    optimal_ablation_path: Optional[str]=None, quiet=False
):
    assert model.cfg.use_attn_result, "Model must be configured to use attention result (model.cfg.use_attn_result)"
    assert model.cfg.use_split_qkv_input, "Model must be configured to use split qkv inputs (model.cfg.use_split_qkv_input)"
    assert model.cfg.use_hook_mlp_in, "Model must be configured to use hook MLP in (model.cfg.use_hook_mlp_in)"
    if model.cfg.n_key_value_heads is not None:
        assert model.cfg.ungroup_grouped_query_attention, "Model must be configured to ungroup grouped attention (model.cfg.ungroup_grouped_attention)"
    
    if aggregation not in allowed_aggregations:
        raise ValueError(f'aggregation must be in {allowed_aggregations}, but got {aggregation}')
        
    # Scores are by default summed across the d_model dimension
    # This means that scores are a [n_src_nodes, n_dst_nodes] tensor

    corrupt_to_clean_scores = get_scores_eap_directional_edge(
        model, graph, dataloader, metric, intervention=intervention, 
        intervention_dataloader=intervention_dataloader, 
        optimal_ablation_path=optimal_ablation_path, quiet=quiet,
        patch_direction='patch-in-corrupt')

    clean_to_corrupt_scores = get_scores_eap_directional_edge(
        model, graph, dataloader, metric, intervention=intervention, 
        intervention_dataloader=intervention_dataloader, 
        optimal_ablation_path=optimal_ablation_path, quiet=quiet,
        patch_direction='patch-in-clean')

    # Identify top 10% of components in which scores differ between the two attribution directions
    scores_asymmetry = asymmetry_score(corrupt_to_clean_scores, clean_to_corrupt_scores)
    abs_scores_asymmetry = scores_asymmetry.abs()
    threshold = torch.quantile(abs_scores_asymmetry.flatten(), 0.9)

    latent_components = abs_scores_asymmetry >= threshold
    latent_components_indices = latent_components.nonzero()

    scores = corrupt_to_clean_scores.clone()

    scores[latent_components.bool()] -= clean_to_corrupt_scores[latent_components.bool()]
    scores[latent_components.bool()] /= 2

    if aggregation == 'mean':
        scores /= model.cfg.d_model
        
    graph.scores[:] =  scores.to(graph.scores.device)