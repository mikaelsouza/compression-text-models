import math
import os
import time

import psutil
import torch
from src.fastdistillation.lm_seqs_dataset import LmSeqsDataset
from src.fastdistillation.utils import logger
from torch import nn
from torch.optim import AdamW
from torch.utils.data import BatchSampler, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup


class Distiller:
    def __init__(
        self,
        params: dict,
        dataset: LmSeqsDataset,
        token_probs: torch.tensor,
        student: nn.Module,
        teacher: nn.Module,
    ):
        logger.info("Initializing Distiller")
        self.params = params
        self.dump_path = params.dump_path

        self.student = student
        self.teacher = teacher

        self.student_config = student.config
        self.vocab_size = student.config.vocab_size

        sampler = RandomSampler(dataset)
        sampler = BatchSampler(
            sampler=sampler, batch_size=params.batch_size, drop_last=False
        )

        self.dataloader = DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
        )

        self.temperature = params.temperature
        assert self.temperature > 0.0

        self.alpha_ce = params.alpha_ce
        self.alpha_mlm = params.alpha_mlm
        self.alpha_cos = params.alpha_cos

        logger.info("Using MLM loss for LM step.")
        self.mlm_mask_prop = params.mlm_mask_prop
        assert 0.0 <= self.mlm_mask_prop <= 1.0
        assert params.word_mask + params.word_keep + params.word_rand == 1.0
        self.pred_probs = torch.tensor(
            [
                params.word_mask,
                params.word_keep,
                params.word_rand,
            ],
            device=params.device,
        )
        self.token_probs = token_probs

        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.n_sequences_epoch = 0
        self.total_loss_epoch = 0
        self.last_loss = 0
        self.last_loss_ce = 0
        self.last_loss_mlm = 0
        self.last_loss_cos = 0
        self.last_log = 0

        self.ce_loss_fct = nn.KLDivLoss(reduction="batchmean")
        self.lm_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.cosine_loss_fct = nn.CosineEmbeddingLoss(reduction="mean")

        logger.info("--- Initializing model optimizer")
        assert params.gradient_accumulation_steps >= 1
        self.num_steps_epoch = len(self.dataloader)
        num_train_optimization_steps = (
            int(
                self.num_steps_epoch
                / params.gradient_accumulation_steps
                * params.n_epoch
            )
            + 1
        )

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in student.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": params.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in student.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        logger.info(
            "------ Number of trainable parameters (student): %i"
            % sum([p.numel() for p in self.student.parameters() if p.requires_grad])
        )
        logger.info(
            "------ Number of parameters (student): %i"
            % sum([p.numel() for p in self.student.parameters()])
        )
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=params.learning_rate,
            eps=params.adam_epsilon,
            betas=(0.9, 0.98),
        )

        warmup_steps = math.ceil(num_train_optimization_steps * params.warmup_prop)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_train_optimization_steps,
        )

        logger.info("--- Initializing Tensorboard")
        self.tensorboard = SummaryWriter(
            log_dir=os.path.join(self.dump_path, "log", "train")
        )
        self.tensorboard.add_text(
            tag="config/training", text_string=str(self.params), global_step=0
        )
        self.tensorboard.add_text(
            tag="config/student", text_string=str(self.student_config), global_step=0
        )

    def prepare_batch_mlm(self, batch):
        """
        Prepare the batch: from the token_ids and the lengths, compute the attention mask and the masked label for MLM.

        Input:
        ------
            batch: `Tuple`
                token_ids: `torch.tensor(bs, seq_length)` - The token ids for each of the sequence. It is padded.
                lengths: `torch.tensor(bs)` - The lengths of each of the sequences in the batch.

        Output:
        -------
            token_ids: `torch.tensor(bs, seq_length)` - The token ids after the modifications for MLM.
            attn_mask: `torch.tensor(bs, seq_length)` - The attention mask for the self-attention.
            mlm_labels: `torch.tensor(bs, seq_length)` - The masked language modeling labels. There is a -100 where there is nothing to predict.
        """
        token_ids, attn_mask = batch
        lengths = attn_mask.sum(-1)
        # token_ids, lengths = self.round_batch(x=token_ids, lengths=lengths)
        assert token_ids.size(0) == lengths.size(0)

        bs, max_seq_len = token_ids.size()
        mlm_labels = token_ids.clone().detach()

        x_prob = self.token_probs[token_ids.flatten()]
        n_tgt = math.ceil(self.mlm_mask_prop * lengths.sum().item())
        tgt_ids = torch.multinomial(x_prob / x_prob.sum(), n_tgt, replacement=False)
        pred_mask = torch.zeros(
            bs * max_seq_len, dtype=torch.bool, device=token_ids.device
        )  # previously `dtype=torch.uint8`, cf pytorch 1.2.0 compatibility
        pred_mask[tgt_ids] = 1
        pred_mask = pred_mask.view(bs, max_seq_len)

        pred_mask[token_ids == self.params.special_tok_ids["pad_token"]] = 0

        _token_ids_real = token_ids[pred_mask]
        _token_ids_rand = _token_ids_real.clone().random_(self.vocab_size)
        _token_ids_mask = _token_ids_real.clone().fill_(
            self.params.special_tok_ids["mask_token"]
        )
        probs = torch.multinomial(
            self.pred_probs, len(_token_ids_real), replacement=True
        )
        _token_ids = (
            _token_ids_mask * (probs == 0).long()
            + _token_ids_real * (probs == 1).long()
            + _token_ids_rand * (probs == 2).long()
        )
        token_ids = token_ids.masked_scatter(pred_mask, _token_ids)

        mlm_labels[
            ~pred_mask
        ] = (
            -100
        )  # previously `mlm_labels[1-pred_mask] = -1`, cf pytorch 1.2.0 compatibility

        # sanity checks
        assert 0 <= token_ids.min() <= token_ids.max() < self.vocab_size

        return token_ids, attn_mask, mlm_labels

    def train(self):
        """
        The real training loop.
        """

        logger.info("Starting training")
        self.last_log = time.time()
        self.student.train()
        self.teacher.eval()

        for _ in range(self.params.n_epoch):

            logger.info(f"--- Starting epoch {self.epoch}/{self.params.n_epoch-1}")

            iter_bar = tqdm(
                self.dataloader,
                desc="-Iter",
            )
            for batch in iter_bar:
                token_ids, attn_mask, lm_labels = self.prepare_batch_mlm(batch=batch)
                self.step(
                    input_ids=token_ids, attention_mask=attn_mask, lm_labels=lm_labels
                )
                iter_bar.update()
                iter_bar.set_postfix(
                    {
                        "Last_loss": f"{self.last_loss:.2f}",
                        "Avg_cum_loss": f"{self.total_loss_epoch/self.n_iter:.2f}",
                    }
                )
            iter_bar.close()

            logger.info(f"--- Ending epoch {self.epoch}/{self.params.n_epoch-1}")
            self.end_epoch()

        logger.info("Save very last checkpoint as `pytorch_model.bin`.")
        self.save_checkpoint(checkpoint_name="pytorch_model.bin")
        logger.info("Training is finished")

    def step(
        self,
        input_ids: torch.tensor,
        attention_mask: torch.tensor,
        lm_labels: torch.tensor,
    ):
        """
        One optimization step: forward of student AND teacher, backward on the loss (for gradient accumulation),
        and possibly a parameter update (depending on the gradient accumulation).

        Input:
        ------
        input_ids: `torch.tensor(bs, seq_length)` - The token ids.
        attention_mask: `torch.tensor(bs, seq_length)` - The attention mask for self attention.
        lm_labels: `torch.tensor(bs, seq_length)` - The language modeling labels (mlm labels for MLM and clm labels for CLM).
        """
        student_outputs = self.student(
            input_ids=input_ids, attention_mask=attention_mask
        )  # (bs, seq_length, voc_size)
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=input_ids, attention_mask=attention_mask
            )  # (bs, seq_length, voc_size)

        s_logits, s_hidden_states = (
            student_outputs["logits"],
            student_outputs["hidden_states"],
        )
        t_logits, t_hidden_states = (
            teacher_outputs["logits"],
            teacher_outputs["hidden_states"],
        )
        assert s_logits.size() == t_logits.size()

        # https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
        # https://github.com/peterliht/knowledge-distillation-pytorch/issues/2

        mask = (
            (lm_labels > -1).unsqueeze(-1).expand_as(s_logits)
        ).bool()  # (bs, seq_length, voc_size)

        s_logits_slct = torch.masked_select(
            s_logits, mask
        )  # (bs * seq_length * voc_size) modulo the 1s in mask
        s_logits_slct = s_logits_slct.view(
            -1, s_logits.size(-1)
        )  # (bs * seq_length, voc_size) modulo the 1s in mask
        t_logits_slct = torch.masked_select(
            t_logits, mask
        )  # (bs * seq_length * voc_size) modulo the 1s in mask
        t_logits_slct = t_logits_slct.view(
            -1, s_logits.size(-1)
        )  # (bs * seq_length, voc_size) modulo the 1s in mask
        assert t_logits_slct.size() == s_logits_slct.size()

        loss_ce = (
            self.ce_loss_fct(
                nn.functional.log_softmax(s_logits_slct / self.temperature, dim=-1),
                nn.functional.softmax(t_logits_slct / self.temperature, dim=-1),
            )
            * (self.temperature) ** 2
        )
        loss = self.alpha_ce * loss_ce

        if self.alpha_mlm > 0.0:
            loss_mlm = self.lm_loss_fct(
                s_logits.view(-1, s_logits.size(-1)), lm_labels.view(-1)
            )
            loss += self.alpha_mlm * loss_mlm

        if self.alpha_cos > 0.0:
            s_hidden_states = s_hidden_states[-1]  # (bs, seq_length, dim)
            t_hidden_states = t_hidden_states[-1]  # (bs, seq_length, dim)
            mask = (
                attention_mask.unsqueeze(-1).expand_as(s_hidden_states).bool()
            )  # (bs, seq_length, dim)
            assert s_hidden_states.size() == t_hidden_states.size()
            dim = s_hidden_states.size(-1)

            s_hidden_states_slct = torch.masked_select(
                s_hidden_states, mask
            )  # (bs * seq_length * dim)
            s_hidden_states_slct = s_hidden_states_slct.view(
                -1, dim
            )  # (bs * seq_length, dim)
            t_hidden_states_slct = torch.masked_select(
                t_hidden_states, mask
            )  # (bs * seq_length * dim)
            t_hidden_states_slct = t_hidden_states_slct.view(
                -1, dim
            )  # (bs * seq_length, dim)

            target = s_hidden_states_slct.new(s_hidden_states_slct.size(0)).fill_(
                1
            )  # (bs * seq_length,)
            loss_cos = self.cosine_loss_fct(
                s_hidden_states_slct, t_hidden_states_slct, target
            )
            loss += self.alpha_cos * loss_cos

        self.total_loss_epoch += loss.item()
        self.last_loss = loss.item()
        self.last_loss_ce = loss_ce.item()
        if self.alpha_mlm > 0.0:
            self.last_loss_mlm = loss_mlm.item()
        if self.alpha_cos > 0.0:
            self.last_loss_cos = loss_cos.item()

        self.optimize(loss)

        self.n_sequences_epoch += input_ids.size(0)

    def optimize(self, loss):
        """
        Normalization on the loss (gradient accumulation or distributed training), followed by
        backward pass on the loss, possibly followed by a parameter update (depending on the gradient accumulation).
        Also update the metrics for tensorboard.
        """
        # Check for NaN
        if (loss != loss).data.any():
            logger.error("NaN detected")
            exit()

        if self.params.gradient_accumulation_steps > 1:
            loss = loss / self.params.gradient_accumulation_steps

        loss.backward()
        self.iter()
        if self.n_iter % self.params.gradient_accumulation_steps == 0:
            nn.utils.clip_grad_norm_(
                self.student.parameters(), self.params.max_grad_norm
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

    def iter(self):
        """
        Update global counts, write to tensorboard and save checkpoint.
        """
        self.n_iter += 1
        self.n_total_iter += 1

        if self.n_total_iter % self.params.log_interval == 0:
            self.log_tensorboard()
            self.last_log = time.time()
        if self.n_total_iter % self.params.checkpoint_interval == 0:
            self.save_checkpoint()

    def log_tensorboard(self):
        """
        Log into tensorboard. Only by the master process.
        """
        for param_name, param in self.student.named_parameters():
            self.tensorboard.add_scalar(
                tag="parameter_mean/" + param_name,
                scalar_value=param.data.mean(),
                global_step=self.n_total_iter,
            )
            self.tensorboard.add_scalar(
                tag="parameter_std/" + param_name,
                scalar_value=param.data.std(),
                global_step=self.n_total_iter,
            )
            if param.grad is None:
                continue
            self.tensorboard.add_scalar(
                tag="grad_mean/" + param_name,
                scalar_value=param.grad.data.mean(),
                global_step=self.n_total_iter,
            )
            self.tensorboard.add_scalar(
                tag="grad_std/" + param_name,
                scalar_value=param.grad.data.std(),
                global_step=self.n_total_iter,
            )

        self.tensorboard.add_scalar(
            tag="losses/cum_avg_loss_epoch",
            scalar_value=self.total_loss_epoch / self.n_iter,
            global_step=self.n_total_iter,
        )
        self.tensorboard.add_scalar(
            tag="losses/loss",
            scalar_value=self.last_loss,
            global_step=self.n_total_iter,
        )
        self.tensorboard.add_scalar(
            tag="losses/loss_ce",
            scalar_value=self.last_loss_ce,
            global_step=self.n_total_iter,
        )
        if self.alpha_mlm > 0.0:
            self.tensorboard.add_scalar(
                tag="losses/loss_mlm",
                scalar_value=self.last_loss_mlm,
                global_step=self.n_total_iter,
            )
        if self.alpha_cos > 0.0:
            self.tensorboard.add_scalar(
                tag="losses/loss_cos",
                scalar_value=self.last_loss_cos,
                global_step=self.n_total_iter,
            )
        self.tensorboard.add_scalar(
            tag="learning_rate/lr",
            scalar_value=self.scheduler.get_lr()[0],
            global_step=self.n_total_iter,
        )

        self.tensorboard.add_scalar(
            tag="global/memory_usage",
            scalar_value=psutil.virtual_memory()._asdict()["used"] / 1_000_000,
            global_step=self.n_total_iter,
        )
        self.tensorboard.add_scalar(
            tag="global/speed",
            scalar_value=time.time() - self.last_log,
            global_step=self.n_total_iter,
        )

    def end_epoch(self):
        """
        Finally arrived at the end of epoch (full pass on dataset).
        Do some tensorboard logging and checkpoint saving.
        """
        logger.info(
            f"{self.n_sequences_epoch} sequences have been trained during this epoch."
        )

        self.save_checkpoint(checkpoint_name=f"model_epoch_{self.epoch}.pth")
        self.tensorboard.add_scalar(
            tag="epoch/loss",
            scalar_value=self.total_loss_epoch / self.n_iter,
            global_step=self.epoch,
        )

        self.epoch += 1
        self.n_sequences_epoch = 0
        self.n_iter = 0
        self.total_loss_epoch = 0

    def save_checkpoint(self, checkpoint_name: str = "checkpoint.pth"):
        """
        Save the current state. Only by the master process.
        """
        mdl_to_save = (
            self.student.module if hasattr(self.student, "module") else self.student
        )
        mdl_to_save.config.save_pretrained(self.dump_path)
        state_dict = mdl_to_save.state_dict()
        torch.save(state_dict, os.path.join(self.dump_path, checkpoint_name))
