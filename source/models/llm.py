import contextlib
import numpy as np
import torch
from torch.cuda.amp import autocast as autocast
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from peft import (
    LoraConfig,
    get_peft_model,
)

def replace_trailing_ones_with_counts(mask):
    B, T = mask.shape
    counts = torch.arange(T, device=mask.device).unsqueeze(0) - (T - mask.sum(dim=1)).unsqueeze(1)

    return torch.clamp(counts, min=0)

def position_weighted_loss(logits, labels, gamma):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    B, T = shift_labels.shape

    loss = torch.nn.CrossEntropyLoss(reduction='none')(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    ).view(B, T)

    mask = (shift_labels != -100).float()
    position_mask = replace_trailing_ones_with_counts(mask)

    loss = (loss * mask * (gamma ** position_mask)).sum() / mask.sum()

    return loss

class BaselineLLM(torch.nn.Module):
    def __init__(
        self,
        args,
        **kwargs
    ):
        super().__init__()
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.temperature = args.temperature
        if "gemma-2" in args.llm_path:
            self.BOS = '<bos><start_of_turn>user\n'
            self.EOS_USER = '<end_of_turn>\n<start_of_turn>model\n'
            self.EOS = '<end_of_turn>\n'
            self.IGNORE_INDEX = -100
        if "granite-3.3" in args.llm_path:
            self.BOS = '<|start_of_role|>user<|end_of_role|>'
            self.EOS_USER = '<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>'
            self.EOS = '<|end_of_text|>\n'
            self.IGNORE_INDEX = -100

        kwargs = {
            "max_memory": {i: '80GiB' for i in range(args.n_gpus)},
            "device_map": "auto",
            "revision": "main",
        }
        if "grpo" in args.run_name:
            kwargs.pop("max_memory", None)
            kwargs.pop("device_map", None)
        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_path, revision=kwargs["revision"])
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'
        model = AutoModelForCausalLM.from_pretrained(
            args.llm_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            **kwargs
        )
        print(f'Loaded {args.llm_path}')

        lora_r: int = args.lora_r
        lora_alpha: int = 16
        lora_dropout: float = 0.1
        lora_target_modules = [
            'k_proj', 'v_proj', 'q_proj', 'o_proj', 
            'gate_proj', 'up_proj', 'down_proj'
        ]
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        print(f'{args.llm_path} has been factorized for training with a rank of {args.lora_r}!')
        self.model = model
        self.word_embedding = self.model.model.get_input_embeddings()

        self.gamma = args.gamma

    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param

    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        if self.device != torch.device("cpu"):
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def forward(self, samples):
        # encode prompts and labels
        prompts = self.tokenizer(samples["prompt"], add_special_tokens=False)
        labels = self.tokenizer(samples["label"], add_special_tokens=False)

        # encode special tokens
        eos_tokens = self.tokenizer(self.EOS, add_special_tokens=False)
        eos_user_tokens = self.tokenizer(self.EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(self.BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device))
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.model.device)).unsqueeze(0)

        batch_size = len(samples["id"])
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []
        for i in range(batch_size):
            # Add bos & eos token
            label_input_ids = labels.input_ids[i][:self.max_prompt_length] + eos_tokens.input_ids
            input_ids = prompts.input_ids[i] + eos_user_tokens.input_ids + label_input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=0)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            label_input_ids = [self.IGNORE_INDEX] * (inputs_embeds.shape[0]-len(label_input_ids)) + label_input_ids
            batch_label_input_ids.append(label_input_ids)

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length + batch_attention_mask[i]
            batch_label_input_ids[i] = [self.IGNORE_INDEX]*pad_length + batch_label_input_ids[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.model.device)

        with self.maybe_autocast():
            if self.gamma < 1.0:
                logits = self.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=label_input_ids,
                    return_dict=True,
                ).logits
                loss = position_weighted_loss(logits, label_input_ids, self.gamma)
            else:
                outputs = self.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=label_input_ids,
                    return_dict=True,
                )
                logits, loss = outputs.logits, outputs.loss

        return {
            "loss": loss,
            "logits": logits,
        }

    def inference(self, samples):
        # encode prompts
        prompts = self.tokenizer(samples["prompt"], add_special_tokens=False)

        # encode special tokens
        eos_user_tokens = self.tokenizer(self.EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(self.BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device))
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.model.device)).unsqueeze(0)

        batch_size = len(samples["id"])
        batch_inputs_embeds = []
        batch_attention_mask = []
        for i in range(batch_size):
            # Add bos & eos token
            input_ids = prompts.input_ids[i] + eos_user_tokens.input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=0)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length + batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.max_completion_length,
                do_sample=False, temperature=self.temperature, top_k=50, top_p=1.0,
                output_scores=True, return_dict_in_generate=True,
                use_cache=True  # IMPORTANT!
            )

            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, 
                normalize_logits=True
            )
            batch_token_scores = []
            for i in range(batch_size):
                token_scores = []
                for token_id, transition_score in zip(outputs.sequences[i], transition_scores[i]):
                    if token_id != self.tokenizer.eos_token_id:
                        token_scores.append((self.tokenizer.decode(token_id).strip(), np.exp(transition_score.cpu().numpy())))
                batch_token_scores.append(token_scores)

        preds = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        preds = [p.strip() for p in preds]

        return {"id": samples["id"],
                "label": samples["label"],
                "pred": preds, "token_scores": batch_token_scores,
        }