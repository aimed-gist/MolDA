from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from tqdm import tqdm
import numpy as np
import torch
from data_utils import (
    CLASSIFICATION_BENCHMARKS,
    REGRESSION_BENCHMARKS,
    MOL2TEXT_BENCHMARKS,
    TEXT2MOL_BENCHMARKS,
    REACTION_BENCHMARKS,
)
import model.added_tokens as added_tokens
import re
from Levenshtein import distance as lev


def caption_evaluate(predictions, targets, tokenizer, prompts, input_mol_strings, flexible_parsing=False):
    references = []
    hypotheses = []
    ref_sentences = []
    hyp_sentences = []
    failure_idxs = []

    meteor_scores = []
    rouge_scores = []
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])

    patterns = {
        "DESCRIPTION": {
            "dual_side": re.compile(r"(?<=<DESCRIPTION>).*?(?=</DESCRIPTION>)"),
            "left_side": re.compile(r"(?<=<DESCRIPTION>).*"),
        },
        "IUPAC": {
            "dual_side": re.compile(r"(?<=<IUPAC>).*?(?=</IUPAC>)"),
            "left_side": re.compile(r"(?<=<IUPAC>).*"),
        },
        "MOLFORMULA": {
            "dual_side": re.compile(r"(?<=<MOLFORMULA>).*?(?=</MOLFORMULA>)"),
            "left_side": re.compile(r"(?<=<MOLFORMULA>).*"),
        },
    }

    for i in range(len(targets)):
        target = targets[i]
        prediction = predictions[i]

        pattern = None
        for key, matching_pattern in patterns.items():
            if matching_pattern["left_side"].search(targets[i]):
                pattern = matching_pattern
                break
        if pattern is None:
            print(targets[i])
            continue
        assert pattern is not None

        if pattern["dual_side"].search(target):
            ref = pattern["dual_side"].search(target).group()
        else:
            ref = pattern["left_side"].search(target).group()
        ref_sentences.append(ref)
        ref_tokens = tokenizer.tokenize(ref)

        try:
            if pattern["dual_side"].search(prediction):
                pred = pattern["dual_side"].search(prediction).group()
            elif pattern["left_side"].search(prediction):
                pred = pattern["left_side"].search(prediction).group()
            elif flexible_parsing:
                # Benchmark mode: use plain text prediction as-is
                # Clean up the prediction (remove EOS tokens, extra whitespace)
                pred = parse_flexible_caption(prediction)
                if pred is None:
                    raise ValueError("Failed to parse caption")
            else:
                raise ValueError("No pattern match found")
            hyp_sentences.append(pred)
            pred_tokens = tokenizer.tokenize(pred)

            references.append([ref_tokens])
            hypotheses.append(pred_tokens)

        except:
            failure_idxs.append(i)
            pred = None
            pred_tokens = None

    if hypotheses:
        bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5))
        bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
        bleu2 *= 100
        bleu4 *= 100

        for ref, hyp in tqdm(zip(references, hypotheses)):
            mscore = meteor_score(ref, hyp)
            meteor_scores.append(mscore)

        _meteor_score = np.mean(meteor_scores)
        _meteor_score *= 100

        for ref_sen, hyp_sen in tqdm(zip(ref_sentences, hyp_sentences)):
            lscore = scorer.score(hyp_sen, ref_sen)
            rouge_scores.append(lscore)

        rouge_1 = np.mean([rs["rouge1"].fmeasure for rs in rouge_scores]) * 100
        rouge_2 = np.mean([rs["rouge2"].fmeasure for rs in rouge_scores]) * 100
        rouge_l = np.mean([rs["rougeL"].fmeasure for rs in rouge_scores]) * 100

    else:
        bleu2 = 0
        bleu4 = 0
        _meteor_score = 0
        rouge_1 = 0
        rouge_2 = 0
        rouge_l = 0

    evaluation_results = {
        "bleu2": bleu2,
        "bleu4": bleu4,
        "rouge1": rouge_1,
        "rouge2": rouge_2,
        "rougeL": rouge_l,
        "meteor": _meteor_score,
    }
    failed_cases = {
        "predictions": [predictions[i] for i in failure_idxs],
        "targets": [targets[i] for i in failure_idxs],
        "prompts": [prompts[i] for i in failure_idxs],
        "input_mol_strings": [input_mol_strings[i] for i in failure_idxs],
    }
    return evaluation_results, failed_cases


from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger
import selfies


def molecule_evaluate(
    predictions, targets, tokenizer, prompts, input_mol_strings, morgan_r=2, flexible_parsing=False, include_failed_as_zero=False
):
    MACCS_sims = []
    morgan_sims = []
    RDK_sims = []
    levs = []
    morgan_r = 2

    failure_idxs = []
    exact_matches = []
    ref_selfies_list = []
    ref_smiles_list = []
    pred_selfies_list = []
    pred_smiles_list = []

    # 변환된 prediction을 저장할 리스트 (평가 직전 데이터)
    converted_predictions = []

    for i in tqdm(range(len(targets))):
        target = targets[i].replace(" ", "")
        prediction = predictions[i].replace(" ", "")

        try:
            if re.search(r"(?<=<SELFIES>).*?(?=</SELFIES>)", target):
                target_selfies = re.search(
                    r"(?<=<SELFIES>).*?(?=</SELFIES>)", target
                ).group()
            else:
                target_selfies = re.search(r"(?<=<SELFIES>).*", target).group()
            target_smiles = selfies.decoder(target_selfies)
            target_mol = Chem.MolFromSmiles(target_smiles)
            target_canonical_smiles = Chem.CanonSmiles(target_smiles)
            target_canonical_selfies = selfies.encoder(target_canonical_smiles)

            # Parse prediction based on mode
            if flexible_parsing:
                # Benchmark mode: flexible parsing (supports SMILES, SELFIES, etc.)
                prediction_selfies = parse_flexible_molecule(prediction)
                if prediction_selfies is None:
                    raise ValueError(f"Failed to parse molecule from: {prediction}")
            else:
                # Standard mode: strict SELFIES parsing
                if re.search(r"(?<=<SELFIES>).*?(?=</SELFIES>)", prediction) is not None:
                    prediction_selfies = re.search(
                        r"(?<=<SELFIES>).*?(?=</SELFIES>)", prediction
                    ).group()
                else:
                    prediction_selfies = re.search(r"(?<=<SELFIES>).*", prediction).group()

                prediction_selfies = prediction_selfies.split("<SELFIES>")[-1]
                prediction_selfies = prediction_selfies.split("</SELFIES>")[0]

                assert (
                    "<SELFIES>" not in prediction_selfies
                    and "</SELFIES>" not in prediction_selfies
                )

            prediction_smiles = selfies.decoder(prediction_selfies)
            prediction_mol = Chem.MolFromSmiles(prediction_smiles)
            prediction_canonical_smiles = Chem.CanonSmiles(prediction_smiles)
            prediction_canonical_selfies = selfies.encoder(prediction_canonical_smiles)

            exact_matches.append(
                Chem.MolToInchi(target_mol) == Chem.MolToInchi(prediction_mol)
            )
            # 변환된 SELFIES 저장
            converted_predictions.append(f"<SELFIES>{prediction_canonical_selfies}</SELFIES>")
        except:
            failure_idxs.append(i)
            prediction_mol = None
            # 원본 prediction 유지 (변환 실패)
            converted_predictions.append(predictions[i])

            if include_failed_as_zero:
                # 실패 케이스에 0점 부여 (전체 평균에 포함)
                exact_matches.append(False)
                MACCS_sims.append(0.0)
                RDK_sims.append(0.0)
                morgan_sims.append(0.0)
                levs.append(0.0)
            # else: 원본 방식 - 실패 케이스 제외 (리스트에 추가 안함)

            print(f"\n=== [Conversion Error at index {i}] ===")
            print(f"Target      : {target}")
            print(f"Prediction  : {prediction}")
            if isinstance(input_mol_strings, list) and len(input_mol_strings) > i:
                print(f"Input Mol   : {input_mol_strings[i]}")
            else:
                print(f"Input Mol   : {input_mol_strings}")
            if isinstance(prompts, list) and len(prompts) > i:
                print(f"Prompt      : {prompts[i]}")
            else:
                print(f"Prompt      : {prompts}")
            print("======================================\n")
            continue

        if prediction_mol is not None:

            levs.append(lev(target_canonical_smiles, prediction_canonical_smiles))

            pred_selfies = tokenizer.tokenize(prediction_canonical_selfies)
            pred_smiles = tokenizer.tokenize(prediction_canonical_smiles)
            pred_selfies_list.append(pred_selfies)
            pred_smiles_list.append(pred_smiles)

            ref_selfies = tokenizer.tokenize(target_canonical_selfies)
            ref_smiles = tokenizer.tokenize(target_canonical_smiles)
            ref_selfies_list.append([ref_selfies])
            ref_smiles_list.append([ref_smiles])

            MACCS_sims.append(
                DataStructs.FingerprintSimilarity(
                    MACCSkeys.GenMACCSKeys(target_mol),
                    MACCSkeys.GenMACCSKeys(prediction_mol),
                    metric=DataStructs.TanimotoSimilarity,
                )
            )
            RDK_sims.append(
                DataStructs.FingerprintSimilarity(
                    Chem.RDKFingerprint(target_mol),
                    Chem.RDKFingerprint(prediction_mol),
                    metric=DataStructs.TanimotoSimilarity,
                )
            )
            morgan_sims.append(
                DataStructs.TanimotoSimilarity(
                    AllChem.GetMorganFingerprint(target_mol, morgan_r),
                    AllChem.GetMorganFingerprint(prediction_mol, morgan_r),
                )
            )

    validity_ratio = 1 - len(failure_idxs) / len(predictions)
    MACCS_sim = np.mean(MACCS_sims)
    RDK_sim = np.mean(RDK_sims)
    morgan_sim = np.mean(morgan_sims)
    exact_match_ratio = np.mean(exact_matches)
    levenshtein_score = np.mean(levs)

    if pred_smiles_list:
        bleu_smiles = corpus_bleu(
            ref_smiles_list, pred_smiles_list, weights=(0.25, 0.25, 0.25, 0.25)
        )
        bleu_smiles *= 100
    else:
        bleu_smiles = 0

    if pred_selfies_list:
        bleu_selfies = corpus_bleu(
            ref_selfies_list, pred_selfies_list, weights=(0.25, 0.25, 0.25, 0.25)
        )
        bleu_selfies *= 100
    else:
        bleu_selfies = 0

    results = {
        "validity_ratio": validity_ratio,
        "MACCS_FTS": MACCS_sim,
        "RDK_FTS": RDK_sim,
        "morgan_FTS": morgan_sim,
        "exact_match_ratio": exact_match_ratio,
        "levenshtein_score": levenshtein_score,
        "bleu_smiles": bleu_smiles,
        "bleu_selfies": bleu_selfies,
    }
    failed_cases = {
        "predictions": [predictions[i] for i in failure_idxs],
        "targets": [targets[i] for i in failure_idxs],
        "prompts": [prompts[i] for i in failure_idxs],
        "input_mol_strings": [input_mol_strings[i] for i in failure_idxs],
    }
    return results, failed_cases, converted_predictions


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def pad_and_concat(tensor_list, fill_value=0):
    """
    concat the first dimension and pad the second dimension
    tensor_list: [[B (diff), N_num, *], ...]
    """
    device = tensor_list[0].device
    dtype = tensor_list[0].dtype
    max_dim1 = max(t.shape[1] for t in tensor_list)
    sum_dim0 = sum(t.shape[0] for t in tensor_list)
    if len(tensor_list[0].shape) == 3:
        out = torch.full(
            (sum_dim0, max_dim1, tensor_list[0].shape[-1]),
            fill_value=fill_value,
            device=device,
            dtype=dtype,
        )
        i = 0
        for t in tensor_list:
            out[i : i + t.shape[0], : t.shape[1]] = t
            i += t.shape[0]
        return out
    elif len(tensor_list[0].shape) == 2:
        out = torch.full(
            (sum_dim0, max_dim1), fill_value=fill_value, device=device, dtype=dtype
        )
        i = 0
        for t in tensor_list:
            out[i : i + t.shape[0], : t.shape[1]] = t
            i += t.shape[0]
        return out
    raise NotImplementedError()


def get_task_specific_list(predictions, targets, tasks, prompts, input_mol_strings):
    unique_tasks = list(set(tasks))
    task_specific_predictions = {t: [] for t in unique_tasks}
    task_specific_targets = {t: [] for t in unique_tasks}
    task_specific_prompts = {t: [] for t in unique_tasks}
    task_specific_input_mol_strings = {t: [] for t in unique_tasks}
    for i, t in enumerate(tasks):
        task_specific_predictions[t].append(predictions[i])
        task_specific_targets[t].append(targets[i])
        task_specific_prompts[t].append(prompts[i])
        task_specific_input_mol_strings[t].append(input_mol_strings[i])
    return (
        task_specific_predictions,
        task_specific_targets,
        task_specific_prompts,
        task_specific_input_mol_strings,
    )


# correspond to all the tasks other than classification
def per_device_evaluate(
    predictions,
    targets,
    tasks,
    prompts,
    input_mol_strings,
    tokenizer,
    total_task_subtask_pairs,
    flexible_parsing=False,
    include_failed_as_zero=False,
):
    # get unique items from all_tasks
    unique_tasks = list(set(tasks))
    # remove tasks_to_be_removed
    tasks_to_be_removed = [
        "smol-name_conversion-i2f",
        "smol-name_conversion-s2f",
        "smol-name_conversion-i2s",
        "smol-name_conversion-s2i",
    ] + CLASSIFICATION_BENCHMARKS

    unique_tasks = [
        t for t in unique_tasks if t.split("/")[0] not in tasks_to_be_removed
    ]

    evaluation_results = {task: dict() for task in unique_tasks}

    (
        task_specific_predictions,
        task_specific_targets,
        task_specific_prompts,
        task_specific_input_mol_strings,
    ) = get_task_specific_list(predictions, targets, tasks, prompts, input_mol_strings)
    failed_cases = {
        "predictions": [],
        "targets": [],
        "prompts": [],
        "tasks": [],
        "input_mol_strings": [],
    }

    # initialize evaluation results for all tasks with null values
    # necessary to make uniform shape of evaluation results
    for t in total_task_subtask_pairs:
        if t.split("/")[0] in tasks_to_be_removed:
            continue

        task_name = t.split("/")[0]
        null_value = 0
        if task_name in REGRESSION_BENCHMARKS:
            results = {
                "mae": null_value,
                "mse": null_value,
                "rmse": null_value,
                "failure_rate": null_value,
            }
        elif (
            task_name in TEXT2MOL_BENCHMARKS + REACTION_BENCHMARKS
        ):  # output is a molecule
            results = {
                "validity_ratio": null_value,
                "MACCS_FTS": null_value,
                "RDK_FTS": null_value,
                "morgan_FTS": null_value,
                "exact_match_ratio": null_value,
                "levenshtein_score": null_value,
                "bleu_smiles": null_value,
                "bleu_selfies": null_value,
            }
        elif task_name in MOL2TEXT_BENCHMARKS:
            results = {
                "bleu2": null_value,
                "bleu4": null_value,
                "rouge1": null_value,
                "rouge2": null_value,
                "rougeL": null_value,
                "meteor": null_value,
            }
        else:
            raise NotImplementedError("Task not implemented")
        # update number of instances
        results["num_instances"] = 0
        evaluation_results[t] = results

    for t in task_specific_predictions.keys():
        if t.split("/")[0] in tasks_to_be_removed:
            continue

        task_predictions = task_specific_predictions[t]
        task_targets = task_specific_targets[t]
        task_prompts = task_specific_prompts[t]
        task_input_mol_strings = task_specific_input_mol_strings[t]
        task_name = t.split("/")[0]

        _converted_predictions = None
        if task_name in REGRESSION_BENCHMARKS:
            results, _failed_cases = regression_evaluate(
                predictions=task_predictions,
                targets=task_targets,
                prompts=task_prompts,
                input_mol_strings=task_input_mol_strings,
                flexible_parsing=flexible_parsing,
            )
        elif (
            task_name in TEXT2MOL_BENCHMARKS + REACTION_BENCHMARKS
        ):  # output is a molecule
            results, _failed_cases, _converted_predictions = molecule_evaluate(
                predictions=task_predictions,
                targets=task_targets,
                tokenizer=tokenizer,
                prompts=task_prompts,
                input_mol_strings=task_input_mol_strings,
                flexible_parsing=flexible_parsing,
                include_failed_as_zero=include_failed_as_zero,
            )
            # 변환된 predictions로 업데이트
            task_specific_predictions[t] = _converted_predictions
        elif task_name in MOL2TEXT_BENCHMARKS:
            results, _failed_cases = caption_evaluate(
                predictions=task_predictions,
                targets=task_targets,
                tokenizer=tokenizer,
                prompts=task_prompts,
                input_mol_strings=task_input_mol_strings,
                flexible_parsing=flexible_parsing,
            )
        else:
            raise NotImplementedError("Task not implemented")
        # update number of instances
        results["num_instances"] = len(task_predictions)
        evaluation_results[t] = results

        if task_name not in CLASSIFICATION_BENCHMARKS:
            for k in _failed_cases.keys():
                failed_cases[k].extend(_failed_cases[k])
            failed_cases["tasks"].extend(
                [t.split("/")[0] for _ in range(len(_failed_cases["predictions"]))]
            )

    # 변환된 predictions를 원래 리스트에 반영
    converted_predictions = []
    for i, task in enumerate(tasks):
        task_name = task.split("/")[0]
        if task_name in TEXT2MOL_BENCHMARKS + REACTION_BENCHMARKS:
            if task in task_specific_predictions:
                # task_specific_predictions[task]에서 해당 인덱스의 변환된 prediction 가져오기
                task_preds = task_specific_predictions[task]
                # 원래 predictions 리스트에서 해당 task의 인덱스 찾기
                task_indices = [j for j, t in enumerate(tasks) if t == task]
                local_idx = task_indices.index(i) if i in task_indices else -1
                if local_idx >= 0 and local_idx < len(task_preds):
                    converted_predictions.append(task_preds[local_idx])
                else:
                    converted_predictions.append(predictions[i])
            else:
                converted_predictions.append(predictions[i])
        else:
            converted_predictions.append(predictions[i])

    return evaluation_results, failed_cases, converted_predictions


def classification_evaluate(total_labels, total_probs):
    total_preds = total_probs.argmax(dim=-1)

    # Convert tensors to numpy arrays for use with scikit-learn metrics
    total_preds_np = total_preds.numpy()
    total_labels_np = total_labels.numpy()

    # Calculate metrics
    acc = accuracy_score(y_true=total_labels_np, y_pred=total_preds_np)
    f1 = f1_score(y_true=total_labels_np, y_pred=total_preds_np)
    prec = precision_score(y_true=total_labels_np, y_pred=total_preds_np)
    rec = recall_score(y_true=total_labels_np, y_pred=total_preds_np)
    try:
        roc_auc = roc_auc_score(
            y_true=total_labels_np,
            y_score=total_probs[
                :, 1
            ].numpy(),  # Use y_score here because roc_auc_score expects probability scores
        )
    except:
        roc_auc = float("nan")

    evaluation_results = {
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "roc_auc": roc_auc,
    }
    return evaluation_results


# correspond to classification tasks
def total_device_evaluate(
    total_labels, total_tasks, total_probs, classification_task_subtask_pairs
):
    evaluation_results = {}

    # initialize evaluation results for classification tasks with null values
    for t in classification_task_subtask_pairs:
        task_name = t.split("/")[0]
        null_value = float("nan")
        results = {
            "accuracy": null_value,
            "f1": null_value,
            "precision": null_value,
            "recall": null_value,
            "roc_auc": null_value,
            "num_instances": 0,
        }
        evaluation_results[t] = results

    unique_tasks = list(set(total_tasks))
    task_specific_labels = {t: [] for t in unique_tasks}
    task_specific_probs = {t: [] for t in unique_tasks}

    for i, t in enumerate(total_tasks):
        task_specific_labels[t].append(total_labels[i])
        task_specific_probs[t].append(total_probs[i])

    for t in task_specific_labels.keys():
        task_probs = task_specific_probs[t]
        task_probs = torch.stack(task_probs, dim=0)
        task_labels = task_specific_labels[t]
        task_labels = torch.stack(task_labels, dim=0)
        task_name = t.split("/")[0]
        if task_name in CLASSIFICATION_BENCHMARKS:
            results = classification_evaluate(
                total_probs=task_probs,
                total_labels=task_labels,
            )
        else:
            raise NotImplementedError("Task not implemented")
        # update number of instances
        results["num_instances"] = len(total_labels)
        evaluation_results[t] = results

    return evaluation_results


from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def parse_flexible_classification(text):
    """
    Parse classification result from various output formats.
    Supports:
    1. Mol-LLM format: <BOOLEAN>True</BOOLEAN> or <BOOLEAN>False</BOOLEAN>
    2. Yes/No format: "Yes", "No", "yes", "no", "YES", "NO"
    3. True/False format: "True", "False", "true", "false"
    4. Text with embedded answer: "Yes, the molecule is...", "No, it does not..."

    Returns:
        list: [prob_false, prob_true] where one is 1.0 and other is 0.0
              or None if parsing fails
    """
    if not text:
        return None

    # Clean up text - remove EOS tokens
    for eos_token in ['</s>', '<|end_of_text|>', '<|endoftext|>', '[EOS]', '<eos>']:
        text = text.replace(eos_token, '')

    text_lower = text.lower().strip()

    # 1. Try Mol-LLM format: <BOOLEAN>True/False</BOOLEAN>
    match = re.search(r'<BOOLEAN>\s*(True|False|true|false)\s*</BOOLEAN>', text, re.IGNORECASE)
    if match:
        result = match.group(1).lower()
        if result == 'true':
            return [0.0, 1.0]
        else:
            return [1.0, 0.0]

    # Left-side only match
    match = re.search(r'<BOOLEAN>\s*(True|False|true|false)', text, re.IGNORECASE)
    if match:
        result = match.group(1).lower()
        if result == 'true':
            return [0.0, 1.0]
        else:
            return [1.0, 0.0]

    # 2. Check for Yes/No at the start of the text (most common benchmark output)
    # Handle cases like "Yes", "No", "Yes.", "No.", "Yes, the molecule...", etc.
    if text_lower.startswith('yes'):
        return [0.0, 1.0]  # Yes = True
    if text_lower.startswith('no'):
        return [1.0, 0.0]  # No = False

    # 3. Check for True/False
    if text_lower.startswith('true'):
        return [0.0, 1.0]
    if text_lower.startswith('false'):
        return [1.0, 0.0]

    # 4. Search anywhere in text for yes/no/true/false patterns
    # Use word boundaries to avoid matching "nope", "yesterday", etc.
    if re.search(r'\byes\b', text_lower):
        return [0.0, 1.0]
    if re.search(r'\bno\b', text_lower):
        return [1.0, 0.0]
    if re.search(r'\btrue\b', text_lower):
        return [0.0, 1.0]
    if re.search(r'\bfalse\b', text_lower):
        return [1.0, 0.0]

    # 5. Check for positive/negative patterns
    # Handle cases like "The molecule is likely to be...", "The molecule is unlikely to be..."
    if re.search(r'\b(likely|positive|active|permeable)\b', text_lower):
        return [0.0, 1.0]
    if re.search(r'\b(unlikely|negative|inactive|impermeable)\b', text_lower):
        return [1.0, 0.0]

    return None


def convert_logit2binary_prob(logits, predictions, tokenizer):
    # WARNING: for specific LLM tokenizer, behaviour might be different
    # below code is for mistral7B tokenizer
    # if you want to use this function for other tokenizer, you should check working, and modify if necessary
    True_token_id = tokenizer.encode("True")[-1]
    False_token_id = tokenizer.encode("False")[-1]

    bos_token, eos_token = added_tokens.BOOL

    # Handle different tokenizer types
    # Some tokenizers (like Galactica/OPT) don't accept list input for encode()
    try:
        # Try list input first (Mistral and most tokenizers)
        boolean_bos_id = tokenizer.encode([bos_token])[-1]
    except (TypeError, ValueError):
        # Fallback to string input (Galactica/OPT tokenizers)
        boolean_bos_id = tokenizer.encode(bos_token)[-1]

    prediction_position_ids = torch.zeros(logits.shape[:-1], dtype=torch.bool)
    is_using_prediction_position_ids = torch.zeros(
        (logits.shape[0], 2), dtype=torch.bool
    ).to(logits.device)

    for idx, pred in enumerate(predictions):
        # first, inspect that pred includes boolean tokens
        # second, inspect that there is only one token between boolean tokens
        # third, get position id of the prediction token between the boolean tokens
        pred_token_ids = tokenizer.encode(pred, add_special_tokens=False)
        try:
            assert re.search(
                f"{bos_token}.+{eos_token}", pred
            ).group(), f"pred should be searched by re pattern {bos_token}.+{eos_token}"
            boolean_bos_position = pred_token_ids.index(boolean_bos_id)
            prediction_position_ids[idx, boolean_bos_position + 1] = True
            is_using_prediction_position_ids[idx, :] = True
        except:
            prediction_position_ids[idx, 0] = True
            is_using_prediction_position_ids[idx, :] = False

    true_logits = logits[prediction_position_ids][:, True_token_id]
    false_logits = logits[prediction_position_ids][:, False_token_id]

    total_logits = torch.cat(
        [false_logits.unsqueeze(1), true_logits.unsqueeze(1)], dim=-1
    )
    total_probs = total_logits.softmax(-1)

    total_probs = torch.where(
        is_using_prediction_position_ids,
        total_probs,
        torch.full_like(total_probs, -1),
    )

    total_probs = [p.tolist() for p in total_probs]
    return total_probs


def parse_flexible_molecule(text):
    """
    Parse molecule from various output formats and convert to SELFIES.
    Supports:
    1. Mol-LLM format: <SELFIES>...</SELFIES>
    2. Galactica format: [START_I_SMILES]...[END_I_SMILES]
    3. Plain SMILES: CC(=O)O

    Returns:
        str: SELFIES string, or None if parsing fails
    """
    import selfies
    from rdkit import Chem

    if not text:
        return None

    # Truncate at EOS token if present
    if '</s>' in text:
        text = text.split('</s>')[0]

    # Truncate at newline if molecule appears before it
    if '\n' in text:
        first_line = text.split('\n')[0].strip()
        text = first_line

    # 1. Try Mol-LLM SELFIES format
    match = re.search(r'<SELFIES>\s*(.*?)\s*</SELFIES>', text)
    if match:
        selfies_str = match.group(1).replace(' ', '')
        return selfies_str

    # 2. Try Galactica SMILES format: [START_I_SMILES]...[END_I_SMILES]
    match = re.search(r'\[START_I_SMILES\](.*?)\[END_I_SMILES\]', text)
    if match:
        smiles_str = match.group(1).strip()
        try:
            # Validate SMILES
            mol = Chem.MolFromSmiles(smiles_str)
            if mol is not None:
                # Convert to canonical SMILES first, then to SELFIES
                canonical_smiles = Chem.CanonSmiles(smiles_str)
                selfies_str = selfies.encoder(canonical_smiles)
                return selfies_str
        except:
            pass

    # 2-1. Try Galactica format without START token (model output continues from prompt ending with [START_I_SMILES])
    # e.g., "CC(C)(C)C(=O)C#N[END_I_SMILES]" -> extract "CC(C)(C)C(=O)C#N"
    match = re.search(r'^(.*?)\[END_I_SMILES\]', text)
    if match:
        smiles_str = match.group(1).strip()
        try:
            mol = Chem.MolFromSmiles(smiles_str)
            if mol is not None:
                canonical_smiles = Chem.CanonSmiles(smiles_str)
                selfies_str = selfies.encoder(canonical_smiles)
                return selfies_str
        except:
            pass

    # 3. Try plain SMILES (heuristic: contains typical SMILES characters)
    # Look for patterns like: C, CC, C(=O)O, c1ccccc1, etc.
    match = re.search(r'[A-Z][a-zA-Z0-9@+\-\[\]()=#\.]+', text)
    if match:
        potential_smiles = match.group()
        try:
            mol = Chem.MolFromSmiles(potential_smiles)
            if mol is not None:
                canonical_smiles = Chem.CanonSmiles(potential_smiles)
                selfies_str = selfies.encoder(canonical_smiles)
                return selfies_str
        except:
            pass

    return None


def parse_flexible_caption(text):
    """
    Parse caption/description from various output formats.
    Supports:
    1. Mol-LLM format: <DESCRIPTION>...</DESCRIPTION>
    2. Plain text: "The molecule is a ..."
    3. Text with EOS tokens: "The molecule is a ...</s>"

    Returns:
        str: Cleaned caption text, or None if empty
    """
    if not text:
        return None

    # 1. Try Mol-LLM format first
    match = re.search(r'<DESCRIPTION>\s*(.*?)\s*</DESCRIPTION>', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Left-side only match
    match = re.search(r'<DESCRIPTION>\s*(.*)', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 2. Plain text: clean up EOS tokens and extra whitespace
    # Remove common EOS tokens
    for eos_token in ['</s>', '<|end_of_text|>', '<|endoftext|>', '[EOS]', '<eos>']:
        text = text.replace(eos_token, '')

    # Remove any remaining special tokens like <s>, [PAD], etc.
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\[[A-Z]+\]', '', text)

    # Clean up whitespace
    text = text.strip()

    # Return None if empty after cleanup
    if not text:
        return None

    return text


def parse_flexible_number(text):
    """
    Parse numerical value from various output formats.
    Supports:
    1. Mol-LLM format: <FLOAT> <|-|><|0|><|.|><|2|><|4|> </FLOAT>
    2. Plain number: -0.24, 0.24, 1e-5, etc.
    3. Number with text: "The value is -0.24" or "Answer: 0.24"
    4. Number followed by extra content: "-0.24</s>Title: ..." or "0.24\nExplanation: ..."
    5. Number with trailing garbage: "-3.09 COc1ccc(OC)cc1" → -3.09
    6. Number with brackets: "4.2 [5]_" → 4.2

    Returns:
        float: Parsed number, or None if parsing fails
    """
    if not text:
        return None

    # Truncate at EOS tokens if present (different models use different tokens)
    for eos_token in ['</s>', '<|end_of_text|>', '<|endoftext|>', '[EOS]']:
        if eos_token in text:
            text = text.split(eos_token)[0]

    # Truncate at newline if number appears before it
    if '\n' in text:
        first_line = text.split('\n')[0].strip()
        # Try to parse first line first
        match = re.search(r'^[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', first_line)
        if match:
            text = first_line

    # 1. Try Mol-LLM format: <FLOAT>...</FLOAT>
    match = re.search(r'<FLOAT>\s*(.*?)\s*</FLOAT>', text)
    if match:
        inner = match.group(1).replace(' ', '')
        inner = inner.replace('<|', '').replace('|>', '')
        try:
            return float(inner)
        except:
            pass

    # 2. Try to extract number at the START of the text (most common case for benchmarks)
    # This handles cases like: "-3.09 COc1ccc(OC)cc1", "4.2 [5]_", "-2.4 (comment)"
    text_stripped = text.strip()
    match = re.match(r'^([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', text_stripped)
    if match:
        try:
            return float(match.group(1))
        except:
            pass

    # 3. Try plain number anywhere (including scientific notation)
    # Match: -0.24, +0.24, 0.24, 1e-5, -1.5e+3, etc.
    match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', text)
    if match:
        try:
            return float(match.group())
        except:
            pass

    return None


def regression_evaluate(predictions, targets, prompts, input_mol_strings, flexible_parsing=False):

    total_labels = []
    total_predictions = []
    failure_idxs = []

    for i in range(len(predictions)):
        label = (
            re.search(r"(?<=<FLOAT>).*?(?=</FLOAT>)", targets[i])
            .group()
            .replace(" ", "")
        )
        label = label.replace("<|", "").replace("|>", "")
        label = float(label)

        # Parse prediction based on mode
        if flexible_parsing:
            # Benchmark mode: flexible parsing for various output formats
            prediction = parse_flexible_number(predictions[i])
            if prediction is not None:
                total_labels.append(label)
                total_predictions.append(prediction)
            else:
                failure_idxs.append(i)
        else:
            # Standard mode: strict Mol-LLM format parsing
            try:
                assert (
                    "<|.|>" in predictions[i]
                ), f"Prediction should include <|.|> token for proper magnitude of order, but {predictions[i]}"
                prediction = (
                    re.search(r"(?<=<FLOAT>).*?(?=</FLOAT>)", predictions[i])
                    .group()
                    .replace(" ", "")
                )

                prediction = prediction.replace("<|", "").replace("|>", "")
                prediction = float(prediction)

                total_labels.append(label)
                total_predictions.append(prediction)
            except:
                failure_idxs.append(i)
    failure_rate = len(failure_idxs) / len(predictions)

    # Calculate regression metrics: mae, mse, rmse
    total_labels = np.array(total_labels)
    total_predictions = np.array(total_predictions)

    mae = np.mean(np.abs(total_labels - total_predictions))
    mse = np.mean((total_labels - total_predictions) ** 2)
    rmse = np.mean((total_labels - total_predictions) ** 2) ** 0.5

    evaluation_results = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "failure_rate": failure_rate,
    }
    failed_cases = {
        "predictions": [predictions[i] for i in failure_idxs],
        "targets": [targets[i] for i in failure_idxs],
        "prompts": [prompts[i] for i in failure_idxs],
        "input_mol_strings": [input_mol_strings[i] for i in failure_idxs],
    }
    return evaluation_results, failed_cases
