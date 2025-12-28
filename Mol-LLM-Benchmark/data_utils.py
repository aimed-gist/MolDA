import torch
from transformers import DataCollatorForSeq2Seq
from torch_geometric.data import Data
from torch_geometric.loader.dataloader import Collater as GraphCollater

import numpy as np

from collections import Counter

import selfies as sf
import os
import rdkit.Chem as Chem
import re
import copy

# Import from prompts module
from prompts import (
    # Templates
    CHEMDFM_PROMPTS,
    LLASMOL_PROMPTS,
    DEFAULT_PROMPTS,
    # Constants
    CLASSIFICATION_BENCHMARKS,
    REGRESSION_BENCHMARKS,
    REACTION_BENCHMARKS,
    TEXT2MOL_BENCHMARKS,
    MOL2TEXT_BENCHMARKS,
    NAME_CONVERSION_BENCHMARKS,
    ALL_TASKS,
    get_task_type,
    task2id,
    id2task,
    # Formatter functions
    selfies_to_smiles,
    extract_description_from_prompt,
    format_prompt_for_galactica,
    format_prompt_for_llama,
    format_prompt_for_mistral,
    format_prompt_for_gpt,
    format_prompt_for_llasmol,
    format_prompt_for_chemdfm,
    format_prompt_for_3d_molm,
    format_prompt_for_mol_llm
)


def extract_user_prompt_from_llada(prompt_text: str) -> str:
    """
    LLaDA-8B 형식에서 user prompt 부분만 추출

    LLaDA format:
    <|startoftext|><|start_header_id|>system<|end_header_id|>
    {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
    {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    Returns:
        user_prompt 부분만 추출 (SELFIES, GRAPH 태그 포함)
    """
    # user<|end_header_id|>\n\n 와 <|eot_id|><|start_header_id|>assistant 사이의 내용 추출
    match = re.search(
        r'<\|start_header_id\|>user<\|end_header_id\|>\s*\n\n(.*?)<\|eot_id\|>',
        prompt_text,
        re.DOTALL
    )
    if match:
        return match.group(1).strip()

    # 매칭 실패시 원본 반환
    return prompt_text


def extract_user_prompt_from_author(prompt_text: str) -> str:
    """
    Author(원본) 데이터 형식에서 user prompt 부분만 추출

    Author format:
    <s>[INST] You are a helpful assistant...
    {user_prompt} [/INST]

    Returns:
        user_prompt 부분만 추출 (시스템 프롬프트 제외, SELFIES/GRAPH 태그 포함)
    """
    # <s>[INST] 제거
    prompt = re.sub(r'^<s>\s*\[INST\]\s*', '', prompt_text)
    # [/INST] 제거
    prompt = re.sub(r'\s*\[/INST\]\s*$', '', prompt)
    # 시스템 프롬프트 제거 (첫 번째 빈 줄 이후가 실제 user prompt)
    # "You are a helpful assistant..." 부분 제거
    prompt = re.sub(
        r'^You are a helpful assistant[^\n]*\n\n',
        '',
        prompt
    )
    return prompt.strip()


input_mol_string_pattern = re.compile("<SELFIES>.*?</SELFIES>")
graph_sequence = re.compile("<GRAPH>[<mol>]+?</GRAPH>")


class DataCollator(DataCollatorForSeq2Seq):
    def __init__(
        self,
        tokenizer,
        padding=True,
        max_length=512,
        pad_to_multiple_of=None,
        return_tensors=None,
        train=True,
        args=None,
    ):
        super().__init__(
            tokenizer,
            padding=padding,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        self.train = train
        self.max_length = max_length
        self.tokenizer.padding_side = "left"
        self.mol_representation = args.mol_representation

        self.apply_molpo = args.train_molpo if self.train else args.eval_molpo

        self.projector_type = args.projector_type
        self.current_epoch = args.current_epoch
        self.args = args

        if self.mol_representation in ["string+graph", "graph_only"]:
            self.graph_collator = GraphCollater([], [])

    def select_mol_representation(self, prompt_text, mol_representation="string+graph"):
        if mol_representation == "string+graph":
            return prompt_text
        elif mol_representation == "string_only":
            string_only_prompt_text = [graph_sequence.sub("", p) for p in prompt_text]
            return string_only_prompt_text
        elif mol_representation == "graph_only":
            graph_only_prompt_text = [
                input_mol_string_pattern.sub("", p) for p in prompt_text
            ]
            return graph_only_prompt_text
        else:
            raise ValueError(
                "check /configs/*.yaml / mol_representation should be one of ['string+graph', 'string_only', 'graph_only']"
            )

    def enumerate_selfies(
        self,
        origin_selfies,
    ):
        origin_smiles = sf.decoder(origin_selfies)

        isomericSmiles = bool(self.args.isomericSmiles)
        canonical = bool(self.args.canonical)
        allHsExplicit = bool(self.args.allHsExplicit)

        processed_smiles = Chem.MolToSmiles(
            Chem.MolFromSmiles(origin_smiles),
            isomericSmiles=isomericSmiles,
            canonical=canonical,
            doRandom=not canonical,
            allHsExplicit=allHsExplicit,
            allBondsExplicit=False,
            kekuleSmiles=False,
        )
        processed_selfies = sf.encoder(processed_smiles)
        return processed_selfies

    def __call__(self, batch, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        # tasks = [task2id(sample["task"]) for sample in batch]  # task id
        temp = [sample for sample in batch]
        tasks = [task2id(sample["task"].split("/", 1)[0]) for sample in batch]
        task_names = [id2task(task) for task in tasks]
        raw_prompt_text = [sample["prompt_text"] for sample in batch]  # 원본 프롬프트 저장
        prompt_text = raw_prompt_text.copy()  # 변환용 복사본
        target_text = [sample["target_text"] for sample in batch]
        input_mol_strings = [sample["input_mol_string"] for sample in batch]
        list_selfies = [
            i.replace("<SELFIES> ", "").replace(" </SELFIES>", "")
            for i in input_mol_strings
        ]
        list_graphs = [
            Data(
                x=torch.tensor(sample["x"], dtype=torch.int64),
                edge_index=torch.tensor(sample["edge_index"], dtype=torch.int64),
                edge_attr=torch.tensor(sample["edge_attr"], dtype=torch.int64),
            )
            for sample in batch
        ]
        # for reagent prediction
        list_additional_graphs = [
            Data(
                x=torch.tensor(sample["additional_x"], dtype=torch.int64),
                edge_index=torch.tensor(
                    sample["additional_edge_index"], dtype=torch.int64
                ),
                edge_attr=torch.tensor(
                    sample["additional_edge_attr"], dtype=torch.int64
                ),
            )
            for sample in batch
        ]

        prompt_text = self.select_mol_representation(
            prompt_text, mol_representation=self.mol_representation
        )

        # Apply model-specific prompt formatting for benchmark mode
        if hasattr(self.args, 'benchmark') and self.args.benchmark:
            intrinsic_prompt = getattr(self.args, 'intrinsic_prompt', True)
            wrapper = getattr(self.args, 'wrapper', True)
            # Detect data format based on direct_data_root path
            data_root = getattr(self.args, 'direct_data_root', '') or ''
            is_llada_data = 'GSAI-ML-LLaDA-8B-Instruct' in data_root
            if not is_llada_data:
            # Author (original) format: remove [INST] wrapper
                prompt_text = [
                    re.sub(r'<s>\[INST\]\s*', '', p).replace('[/INST]', '')
                    for p in prompt_text
                ]
            # Apply model-specific formatting
            # Use both llm_model and filename to identify the model
            model_name = self.args.llm_model.lower()
            filename = getattr(self.args, 'filename', '').lower()

            # Galactica, ChemDFM(intrinsic_prompt=False)는 시스템 프롬프트 포함한 원본을 사용하므로 전처리 스킵
            is_galactica = 'galactica' in model_name or 'galactica' in filename
            is_chemdfm = 'chemdfm' in model_name or 'chemdfm' in filename
            chemdfm_intrinsic = getattr(self.args, 'intrinsic_prompt', True)
            # skip_preprocessing = is_galactica or (is_chemdfm and not chemdfm_intrinsic)

            # if not skip_preprocessing:
            #     # Extract user prompt based on data format (Galactica 제외)
            #     if is_llada_data:
            #         # LLaDA-8B format: extract user prompt from special tokens
            #         prompt_text = [extract_user_prompt_from_llada(p) for p in prompt_text]
            #     else:
            #         # Author (original) format: remove [INST] wrapper
            #         prompt_text = [
            #             re.sub(r'<s>\[INST\]\s*', '', p).replace('[/INST]', '')
            #             for p in prompt_text
            #         ]

            new_prompt_text = []

            for i, p in enumerate(prompt_text):
                selfies_str = list_selfies[i]
                task_name = task_names[i] if i < len(task_names) else ""

                # Route to appropriate formatter
                # Check filename first for LoRA-based models (e.g., LlaSMol uses base Mistral)
                if 'llasmol' in filename or 'llasmol' in model_name:
                    formatted_prompt = format_prompt_for_llasmol(p, selfies_str, task_name,intrinsic_prompt=intrinsic_prompt,)
                elif 'molm_3d' in filename or '3d_molm' in filename or '3dmolm' in filename:
                    formatted_prompt = format_prompt_for_3d_molm(p, selfies_str, task_name)
                elif is_galactica:
                    formatted_prompt = format_prompt_for_galactica(p, selfies_str, task_name)
                elif 'chemdfm' in model_name or 'chemdfm' in filename:
                    formatted_prompt = format_prompt_for_chemdfm(p, selfies_str, task_name, intrinsic_prompt=intrinsic_prompt)
                elif 'llama' in model_name:
                    formatted_prompt = format_prompt_for_llama(p, selfies_str, task_name)
                # elif 'mistral' in model_name:
                #     formatted_prompt = format_prompt_for_mistral(p, selfies_str, tas_name)
                elif 'gpt' in model_name:
                    formatted_prompt = format_prompt_for_gpt(p, selfies_str, task_name)
                elif 'HJChoi' in filename:
                    formatted_prompt = format_prompt_for_mol_llm(p,selfies_str,task_name)
                    
                else:
                    # Default: no special formatting
                    formatted_prompt = p

                new_prompt_text.append(formatted_prompt)

            prompt_text = new_prompt_text
        else:
            # Non-benchmark mode: apply HJChoi formatting if filename matches
            filename = getattr(self.args, 'filename', '').lower()
            if 'hjchoi' in filename:
                new_prompt_text = []
                for i, p in enumerate(prompt_text):
                    selfies_str = list_selfies[i]
                    task_name = task_names[i] if i < len(task_names) else ""
                    formatted_prompt = format_prompt_for_mol_llm(p, selfies_str, task_name)
                    new_prompt_text.append(formatted_prompt)
                prompt_text = new_prompt_text

        if not self.train and self.args.eval_modality_util in [
            "string",
            "graph",
        ]:
            shuffled_idx = []
            # shuffle the selfies_idx, guarantee that the selfies_idx is not in order
            for i in range(len(list_selfies)):
                idxs = np.random.choice(
                    range(len(list_selfies)), size=2, replace=False
                ).tolist()
                if i in idxs:
                    idxs.remove(i)
                shuffled_idx.append(idxs[0])

            if self.args.eval_modality_util == "string":
                processed_selfies = [list_selfies[i] for i in shuffled_idx]
                for i in range(len(prompt_text)):
                    assert (
                        list_selfies[i] in prompt_text[i]
                    ), f"{list_selfies[i]} not in {prompt_text[i]}"
                    prompt_text[i] = prompt_text[i].replace(
                        list_selfies[i], processed_selfies[i]
                    )

            if self.args.eval_modality_util == "graph":
                list_graphs = [list_graphs[i] for i in shuffled_idx]
                list_additional_graphs = [
                    list_additional_graphs[i] for i in shuffled_idx
                ]

        if self.args.selfies_enumeration:
            processed_selfies = [
                self.enumerate_selfies(list_selfies[i])
                for i in range(len(list_selfies))
            ]
            for i in range(len(prompt_text)):
                assert (
                    list_selfies[i] in prompt_text[i]
                ), f"{list_selfies[i]} not in {prompt_text[i]}"
                prompt_text[i] = prompt_text[i].replace(
                    list_selfies[i], processed_selfies[i]
                )
                list_selfies = processed_selfies

        if self.apply_molpo:
            if self.train:
                self.reject_cardinal = self.current_epoch
            else:
                self.reject_cardinal = 0

            prompt_text_reject = prompt_text.copy()

            if self.args.apply_preference_system_prompt:
                for i in range(len(prompt_text_reject)):
                    preference_system_prompt = "In the following problems, molecular graph is either accurate or inaccurate. Your predictions should be based primarily on careful understanding of the provided graph."
                    prompt_text_reject[i] = re.sub(
                        r"(?<=\[INST\]).*(?=\n\n)",
                        preference_system_prompt,
                        prompt_text_reject[i],
                    )

            prompt_text = prompt_text + prompt_text_reject * (
                self.args.molpo_batch_division - 1
            )
            if hasattr(self.args, "reject_label_mask") and self.args.reject_label_mask:
                reject_target_text = [sample[f"{self.reject_cardinal}-th_rejected_target_text"] for sample in batch]
                target_text = target_text + reject_target_text
            else:
                target_text = target_text * self.args.molpo_batch_division
            tasks = tasks * self.args.molpo_batch_division
            task_names = task_names * self.args.molpo_batch_division

            if "graph" in self.mol_representation:
                list_rejected_graphs = [
                    Data(
                        x=torch.tensor(
                            sample[f"{self.reject_cardinal}-th_rejected_x"],
                            dtype=torch.int64,
                        ),
                        edge_index=torch.tensor(
                            sample[f"{self.reject_cardinal}-th_rejected_edge_index"],
                            dtype=torch.int64,
                        ),
                        edge_attr=torch.tensor(
                            sample[f"{self.reject_cardinal}-th_rejected_edge_attr"],
                            dtype=torch.int64,
                        ),
                    )
                    for sample in batch
                ]
                # for reagent prediction
                list_rejected_additional_graphs = [
                    Data(
                        x=torch.tensor(
                            sample[f"{self.reject_cardinal}-th_additional_rejected_x"],
                            dtype=torch.int64,
                        ),
                        edge_index=torch.tensor(
                            sample[
                                f"{self.reject_cardinal}-th_additional_rejected_edge_index"
                            ],
                            dtype=torch.int64,
                        ),
                        edge_attr=torch.tensor(
                            sample[
                                f"{self.reject_cardinal}-th_additional_rejected_edge_attr"
                            ],
                            dtype=torch.int64,
                        ),
                    )
                    for sample in batch
                ]

                list_graphs = (
                    list_graphs * (self.args.molpo_batch_division - 1)
                    + list_rejected_graphs
                )
                list_additional_graphs = (
                    list_additional_graphs * (self.args.molpo_batch_division - 1)
                    + list_rejected_additional_graphs
                )

        # address <mol> token in prompt_text, for the case of using graph modality
        if self.projector_type == "mlp" and "graph" in self.mol_representation:
            for i in range(len(prompt_text)):
                if task_names[i] in ["reagent_prediction"]:
                    num_nodes_in_graph = list_graphs[i].x.size(0)
                    num_nodes_mol = "<mol>" * num_nodes_in_graph
                    mol_tokens_pattern = re.compile(r"(<mol>)+(?=</GRAPH>\|>>\|)")
                    assert mol_tokens_pattern.search(
                        prompt_text[i]
                    ), f"{prompt_text[i]}"
                    prompt_text[i] = mol_tokens_pattern.sub(
                        num_nodes_mol, prompt_text[i]
                    )

                    num_additional_nodes_in_graph = list_additional_graphs[i].x.size(0)
                    num_additional_nodes_mol = "<mol>" * num_additional_nodes_in_graph
                    additional_mol_tokens_pattern = re.compile(
                        r"(?<=\|>>\|<GRAPH>)(<mol>)+"
                    )
                    assert additional_mol_tokens_pattern.search(
                        prompt_text[i]
                    ), f"{prompt_text[i]}"
                    prompt_text[i] = additional_mol_tokens_pattern.sub(
                        num_additional_nodes_mol, prompt_text[i]
                    )
                elif task_names[i] in TEXT2MOL_BENCHMARKS:
                    # there is no input <mol> token
                    pass
                else:
                    num_nodes_in_graph = list_graphs[i].x.size(0)
                    num_nodes_mol = "<mol>" * num_nodes_in_graph
                    mol_tokens_pattern = re.compile("(<mol>)+")
                    assert mol_tokens_pattern.search(
                        prompt_text[i]
                    ), f"{prompt_text[i]}"
                    prompt_text[i] = mol_tokens_pattern.sub(
                        num_nodes_mol, prompt_text[i]
                    )

        self.tokenizer.padding_side = "left"
        prompt_tokenized = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
        )
        target_tokenized = self.tokenizer(
            target_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
        )

        full_input_ids = [
            p + t
            for p, t in zip(
                prompt_tokenized["input_ids"], target_tokenized["input_ids"]
            )
        ]
        full_attention_mask = [
            p + t
            for p, t in zip(
                prompt_tokenized["attention_mask"], target_tokenized["attention_mask"]
            )
        ]

        prompt_length = [len(p) for p in prompt_tokenized["input_ids"]]
        full_input_ids = [f_ids[: self.max_length] for f_ids in full_input_ids]
        full_attention_mask = [
            f_ids[: self.max_length] for f_ids in full_attention_mask
        ]

        self.tokenizer.padding_side = "left"
        features = self.tokenizer.pad(
            {"input_ids": full_input_ids, "attention_mask": full_attention_mask},
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        if not self.train:
            prompt_features = self.tokenizer.pad(
                {
                    "input_ids": [p for p in prompt_tokenized["input_ids"]],
                    "attention_mask": [p for p in prompt_tokenized["attention_mask"]],
                },
                padding=self.padding,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=return_tensors,
            )

            features["prompt_input_ids"] = prompt_features.input_ids  # ['input_ids']
            features["prompt_attention_mask"] = (
                prompt_features.attention_mask
            )  # ['attention_mask']

            self.tokenizer.padding_side = "right"
            gen_features = self.tokenizer.pad(
                {
                    "input_ids": [t for t in target_tokenized["input_ids"]],
                },
                padding=self.padding,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=return_tensors,
            )
            gen_features.input_ids = gen_features.input_ids.masked_fill(
                gen_features.input_ids == self.tokenizer.pad_token_id, -100
            )
            features["gen_labels"] = gen_features.input_ids

            input_mol_strings_tokenized = self.tokenizer(
                input_mol_strings,
                truncation=False,
                max_length=self.max_length,
                padding=True,
                return_tensors=return_tensors,
                add_special_tokens=False,
            )

            features["input_mol_strings"] = input_mol_strings_tokenized.input_ids

        labels_ids = torch.full_like(features["input_ids"], self.tokenizer.pad_token_id)
        for i, target in enumerate(target_tokenized["input_ids"]):
            label = target
            if prompt_length[i] >= self.max_length:
                continue
            else:
                len_label = min(len(label), self.max_length - prompt_length[i])
                labels_ids[i, -len_label:] = torch.tensor(
                    label[:len_label], dtype=torch.int64
                )

        labels_ids = labels_ids.masked_fill(
            labels_ids == self.tokenizer.pad_token_id, -100
        )
        features["labels"] = labels_ids
        if self.apply_molpo:
            molpo_labels_ids = labels_ids.clone()
            for molpo_mask_id in self.tokenizer.molpo_mask_ids:
                molpo_labels_ids = molpo_labels_ids.masked_fill(
                    molpo_labels_ids == molpo_mask_id, -100
                )
            if hasattr(self.args, "reject_label_mask") and self.args.reject_label_mask:
                num_chosen = molpo_labels_ids.shape[0] // self.args.molpo_batch_division
                chosen_molpo_labels_ids = molpo_labels_ids.clone()[:num_chosen]
                reject_molpo_labels_ids = molpo_labels_ids.clone()[num_chosen:]

                chosen_molpo_labels_ids = chosen_molpo_labels_ids.masked_fill(
                    chosen_molpo_labels_ids == reject_molpo_labels_ids, -100
                )
                molpo_labels_ids = torch.cat(
                    (chosen_molpo_labels_ids, reject_molpo_labels_ids), dim=0
                )
            features["molpo_labels"] = molpo_labels_ids

        assert (
            features.input_ids.size(1) <= self.max_length
        ), f"features.input_ids.size(1)={features.input_ids.size(1)} > self.max_length={self.max_length}"
        assert (
            features.labels.size(1) <= self.max_length
        ), f"features.labels.size(1)={features.labels.size(1)} > self.max_length={self.max_length}"

        features["tasks"] = torch.tensor(tasks, dtype=torch.int16)
        # 원본 데이터셋의 idx 저장 (오프라인 평가용)
        batch_idx = [sample.get("idx", i) for i, sample in enumerate(batch)]
        features["idx"] = torch.tensor(batch_idx, dtype=torch.int64)
        # 문자열 리스트 저장 (transfer_batch_to_device에서 GPU 이동 스킵)
        features["raw_prompt_text"] = raw_prompt_text  # 원본 프롬프트 텍스트 (변환 전)
        features["prompt_text"] = prompt_text  # 변환된 프롬프트 텍스트
        features["target_text"] = target_text  # 타겟 텍스트

        # 3D-MoLM용 3D graph 생성
        filename = getattr(self.args, 'filename', '').lower()
        if 'molm_3d' in filename or '3d_molm' in filename or '3dmolm' in filename:
            from model.molm_3d.graph_utils import smiles2graph_with_timeout, load_unimol_dictionary

            # 캐시 로드 (한 번만)
            if not hasattr(self, '_molm3d_graph_cache'):
                direct_data_root = getattr(self.args, 'direct_data_root', None)
                if direct_data_root:
                    # 캐시 경로 후보들 (폴더 내부, 폴더 외부)
                    cache_paths = [
                        os.path.join(direct_data_root, 'molm3d_graphs.pt'),  # 폴더 내부
                        direct_data_root.rstrip('/') + '_molm3d_graphs.pt',  # 폴더 외부
                    ]
                    cache_path = None
                    for path in cache_paths:
                        if os.path.exists(path):
                            cache_path = path
                            break

                    if cache_path:
                        print(f"[DataCollator] Loading MoLM3D graph cache from: {cache_path}")
                        self._molm3d_graph_cache = torch.load(cache_path, weights_only=False)
                        print(f"[DataCollator] Loaded {len(self._molm3d_graph_cache)} cached graphs")
                    else:
                        print(f"[DataCollator] No cache found, will generate graphs on-the-fly")
                        self._molm3d_graph_cache = None
                else:
                    self._molm3d_graph_cache = None

            # dictionary 로드 (한 번만, fallback용)
            if not hasattr(self, '_unimol_dictionary'):
                self._unimol_dictionary = load_unimol_dictionary()

            molm_3d_graphs = []
            for i, sample in enumerate(batch):
                idx = sample.get("idx", i)

                # 캐시에서 로드 시도
                if self._molm3d_graph_cache and idx in self._molm3d_graph_cache:
                    cached = self._molm3d_graph_cache[idx]
                    # Convert numpy arrays to torch tensors if needed
                    if isinstance(cached[0], np.ndarray):
                        graph = (
                            torch.from_numpy(cached[0]).long(),
                            torch.from_numpy(cached[1]),
                            torch.from_numpy(cached[2]).long()
                        )
                    else:
                        graph = cached
                else:
                    # Fallback: 캐시에 없으면 실시간 생성 (타임아웃 + 2D fallback 지원)
                    try:
                        selfies_str = list_selfies[i] if i < len(list_selfies) else ""
                        smiles = sf.decoder(selfies_str) if selfies_str else ""
                        graph = smiles2graph_with_timeout(smiles, self._unimol_dictionary, timeout=5) if smiles else None
                    except:
                        graph = None
                molm_3d_graphs.append(graph)
            features["molm_3d_graphs"] = molm_3d_graphs

        if "graph" in self.mol_representation:
            graphs = self.graph_collator(list_graphs)
            additional_graphs = self.graph_collator(list_additional_graphs)
            features["graphs"] = graphs
            features["additional_graphs"] = additional_graphs
            features["is_mol_token"] = (
                features["input_ids"] == self.tokenizer.mol_token_id
            )
            if not self.train:
                features["prompt_is_mol_token"] = (
                    features["prompt_input_ids"].clone().detach() # .clone().detach() 수정됨
                    == self.tokenizer.mol_token_id
                )

        return features


def random_noise_selfies(selfies, tokenizer, sl_noise_ratio=0.3):
    selfies_ids = tokenizer.encode(selfies, add_special_tokens=False)
    total_selfies_token_ids = tokenizer.selfies_token_ids
    num_ids_to_replace = int(sl_noise_ratio * len(selfies_ids))
    replacing_random_ids = np.random.choice(
        total_selfies_token_ids, num_ids_to_replace, replace=True
    )

    # replace selfies_ids with randomly selected total_selfies_token_ids as many as num_ids_to_replace
    position_to_replace = np.random.choice(
        len(selfies_ids), num_ids_to_replace, replace=False
    )
    noised_selfies_ids = copy.deepcopy(selfies_ids)
    for i, replance_idx in enumerate(position_to_replace):
        noised_selfies_ids[replance_idx] = replacing_random_ids[i]

    noised_selfies = tokenizer.decode(
        noised_selfies_ids, skip_special_tokens=True
    ).replace(" ", "")
    return noised_selfies
