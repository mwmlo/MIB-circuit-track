import argparse
import os
import torch
import pickle
from functools import partial

from transformer_lens import HookedTransformer, HookedTransformerConfig
from huggingface_hub import hf_hub_download

from MIB_circuit_track.dataset import HFEAPDataset
from eap.graph import Graph
from eap.attribute import attribute
from eap.attribute_node import attribute_node
from MIB_circuit_track.metrics import get_metric
from MIB_circuit_track.utils import MODEL_NAME_TO_FULLNAME, TASKS_TO_HF_NAMES, COL_MAPPING

from custom_method import custom_attribute_node, custom_attribute_inputs_node, custom_attribute_edge, custom_attribute_inputs_edge, custom_attribute_eap_node, custom_attribute_eap_edge


def load_interpbench_model():
    hf_cfg = hf_hub_download("mib-bench/interpbench", filename="ll_model_cfg.pkl")
    hf_model = hf_hub_download("mib-bench/interpbench", subfolder="ioi_all_splits", filename="ll_model_100_100_80.pth")

    cfg_dict = pickle.load(open(hf_cfg, "rb"))
    if isinstance(cfg_dict, dict):
        cfg = HookedTransformerConfig.from_dict(cfg_dict)
    else:
        # Some cases in InterpBench have the config as a HookedTransformerConfig object instead of a dict
        assert isinstance(cfg_dict, HookedTransformerConfig)
        cfg = cfg_dict
    cfg.device = "cuda"

    # Enable evaluation mode in the IOI model; has a different config during training
    cfg.use_hook_mlp_in = True
    cfg.use_attn_result = True
    cfg.use_split_qkv_input = True

    model = HookedTransformer(cfg)
    model.load_state_dict(torch.load(hf_model, map_location="cuda"))
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs='+', required=True)
    parser.add_argument("--tasks", type=str, nargs='+', required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--ig-steps", type=int, default=5)
    parser.add_argument("--ablation", type=str, choices=['patching', 'zero', 'mean', 'mean-positional', 'optimal'], default='patching')
    parser.add_argument("--optimal-ablation-path", type=str, default=None)
    parser.add_argument("--level", type=str, choices=['node', 'neuron', 'edge'], default='edge')
    parser.add_argument("--split", type=str, choices=['train', 'validation', 'test'], default='train')
    parser.add_argument("--head", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--circuit-dir", type=str, default='circuits')
    args = parser.parse_args()

    for model_name in args.models:
        if model_name in ("qwen2.5", "gemma2", "llama3"):
            model = HookedTransformer.from_pretrained(MODEL_NAME_TO_FULLNAME[model_name],
                                                    attn_implementation="eager", torch_dtype=torch.bfloat16)
        elif model_name == "interpbench":
            model = load_interpbench_model()
        else:
            model = HookedTransformer.from_pretrained(MODEL_NAME_TO_FULLNAME[model_name])
        model.cfg.use_split_qkv_input = True
        model.cfg.use_attn_result = True
        model.cfg.use_hook_mlp_in = True
        model.cfg.ungroup_grouped_query_attention = True
        neuron_level = args.level == 'neuron'
        node_scores = args.level == 'node'

        for task in args.tasks:
            if f"{task.replace('_', '-')}_{model_name}" not in COL_MAPPING:
                continue
            graph = Graph.from_model(model, neuron_level=neuron_level, node_scores=node_scores)
            hf_task_name = f'mib-bench/{TASKS_TO_HF_NAMES[task]}'
            dataset = HFEAPDataset(hf_task_name, model.tokenizer, split=args.split, task=task, model_name=model_name, num_examples=args.num_examples)
            if args.head is not None:
                head = args.head
                if len(dataset) < head:
                    print(f"Warning: dataset has only {len(dataset)} examples, but head is set to {head}; using all examples.")
                    head = len(dataset)
                dataset.head(head)
            dataloader = dataset.to_dataloader(batch_size=args.batch_size)
            metric = get_metric('logit_diff', task, model.tokenizer, model)
            attribution_metric = partial(metric, mean=True, loss=True)
            if args.level == 'edge':
                if args.method == 'custom':
                    custom_attribute_edge(model, graph, dataloader, attribution_metric, args.ablation,
                                          ig_steps=args.ig_steps, intervention_dataloader=dataloader, 
                                          optimal_ablation_path=args.optimal_ablation_path)
                elif args.method == 'custom-inputs':
                    custom_attribute_inputs_edge(model, graph, dataloader, attribution_metric, args.ablation,
                                                 ig_steps=args.ig_steps, intervention_dataloader=dataloader,
                                                 optimal_ablation_path=args.optimal_ablation_path)
                elif args.method == 'custom-eap':
                    custom_attribute_eap_edge(model, graph, dataloader, attribution_metric, args.ablation,
                                              ig_steps=args.ig_steps, intervention_dataloader=dataloader,
                                              optimal_ablation_path=args.optimal_ablation_path)
                else:
                    attribute(model, graph, dataloader, attribution_metric, args.method, args.ablation, 
                                ig_steps=args.ig_steps, optimal_ablation_path=args.optimal_ablation_path,
                                intervention_dataloader=dataloader)
            else:
                if args.method == 'custom':
                    custom_attribute_node(model, graph, dataloader, attribution_metric, args.ablation, 
                                          neuron=args.level == 'neuron', ig_steps=args.ig_steps, 
                                          optimal_ablation_path=args.optimal_ablation_path,
                                          intervention_dataloader=dataloader)
                elif args.method == 'custom-inputs':
                    custom_attribute_inputs_node(model, graph, dataloader, attribution_metric, args.ablation, 
                                                 neuron=args.level == 'neuron', ig_steps=args.ig_steps,
                                                 optimal_ablation_path=args.optimal_ablation_path,
                                                 intervention_dataloader=dataloader)
                elif args.method == 'custom-eap':
                    custom_attribute_eap_node(model, graph, dataloader, attribution_metric, args.ablation, 
                                              neuron=args.level == 'neuron', ig_steps=args.ig_steps,
                                              optimal_ablation_path=args.optimal_ablation_path,
                                              intervention_dataloader=dataloader)
                else:
                    attribute_node(model, graph, dataloader, attribution_metric, args.method, 
                                    args.ablation, neuron=args.level == 'neuron', ig_steps=args.ig_steps,
                                    optimal_ablation_path=args.optimal_ablation_path,
                                    intervention_dataloader=dataloader)

            # Save the graph
            method_name_saveable = f"{args.method}_{args.ablation}_{args.level}"
            circuit_path = os.path.join(args.circuit_dir, method_name_saveable, f"{task.replace('_', '-')}_{model_name}")
            os.makedirs(circuit_path, exist_ok=True)
            
            # graph.to_pt(f'{circuit_path}/importances.pt')
            graph.to_json(f'{circuit_path}/importances.json')
