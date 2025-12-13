# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os

from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from datasets import load_dataset, load_from_disk, load_dataset_builder

from data_utils import DataCollator


class Stage3DM(LightningDataModule):
    def __init__(
        self,
        tokenizer,
        mode: str = "to_be_removed",
        num_workers: int = 0,
        args=None,
    ):
        super().__init__()
        self.args = args
        self.mode = mode
        self.num_workers = num_workers

        self.max_length = args.max_length
        self.batch_size = args.batch_size
        self.inference_batch_size = args.inference_batch_size

        self.mol_representation = args.mol_representation
        self.tokenizer = tokenizer

        if self.mode in ["test"]:
            self.test_dataset = get_dataset("test", tokenizer, args)
        else:
            if args.debug:
                self.train_dataset = get_dataset("test", tokenizer, args)
            else:
                self.train_dataset = get_dataset("train", tokenizer, args)
            self.test_dataset = get_dataset("test", tokenizer, args)
            self.val_dataset = get_dataset("test", tokenizer, args)

        # InstructGraph.py 경로 결정: direct_data_root 사용 시 상위 디렉토리에서 찾기
        if hasattr(args, 'direct_data_root') and args.direct_data_root is not None:
            # direct_data_root의 부모 디렉토리에서 InstructGraph.py 찾기
            instruct_graph_path = os.path.join(os.path.dirname(args.direct_data_root), "InstructGraph.py")
        else:
            instruct_graph_path = os.path.join(args.raw_data_root, "InstructGraph.py")

        print(instruct_graph_path, "-InstructGraph.py path")
        builder = load_dataset_builder(
            instruct_graph_path,
            trust_remote_code=True,
        )
        builder.config.train_tasks
        self.task_subtask_name_pairs = list(builder.config.test_tasks)

        tokenizer.padding_side = "left"
        self.train_collator = DataCollator(
            tokenizer=tokenizer,
            padding=True,
            max_length=args.max_length,
            return_tensors="pt",
            train=True,
            args=args,
        )
        self.eval_collator = DataCollator(
            tokenizer=tokenizer,
            padding=True,
            max_length=args.max_length,
            return_tensors="pt",
            train=False,
            args=args,
        )

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if self.args.num_workers > 0 else False,
            collate_fn=self.train_collator,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True if self.args.num_workers > 0 else False,
            collate_fn=self.eval_collator,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True if self.args.num_workers > 0 else False,
            collate_fn=self.eval_collator,
        )
        return loader


def get_dataset(split, tokenizer, args):
    if hasattr(args, 'direct_data_root') and args.direct_data_root is not None:
        print(f"Loading dataset from direct_data_root: {args.direct_data_root}")
        dataset = load_from_disk(args.direct_data_root)

        # Apply task filtering if tasks are specified
        if hasattr(args, 'tasks') and args.tasks is not None:
            original_size = len(dataset)
            dataset = dataset.filter(lambda x: x["task"] in args.tasks)
            filtered_size = len(dataset)
            print(f"Filtered dataset: {original_size} -> {filtered_size} samples")
            print(f"Allowed tasks: {args.tasks}")
    else:
        data_path = args.raw_data_root
        # TODO: deprecate mol_representation in preprocessed data, substitute mol representation with <INPUT>
        mol_representation = "string+graph"
        num_query_token = args.num_query_token
        base_model = args.llm_model.replace("/", "-")

        # to avoid re-generating
        tags = [base_model, mol_representation]
        if "graph" in mol_representation:
            tags += [f"q{num_query_token}"]
        tags += [split]
        processed_file_name = "_".join(tags)
        preprocessed_data_path = os.path.join(data_path, processed_file_name)

        if args.data_tag is not None:
            taged_preprocessed_data_path = preprocessed_data_path + f"_{args.data_tag}"
            if os.path.exists(taged_preprocessed_data_path):
                dataset = load_from_disk(taged_preprocessed_data_path)
            elif os.path.exists(preprocessed_data_path):
                assert args.tasks is not None
                print(
                    f"preprocessed data not found: {taged_preprocessed_data_path}, filtering from {preprocessed_data_path}"
                )
                dataset = load_from_disk(preprocessed_data_path)
                dataset = dataset.filter(lambda x: x["task"] in args.tasks)
                dataset.save_to_disk(taged_preprocessed_data_path)
            else:
                raise ValueError(f"preprocessed data not found: {preprocessed_data_path}")
        else:
            dataset = load_from_disk(preprocessed_data_path)

    return dataset
