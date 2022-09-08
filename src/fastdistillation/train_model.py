import pytorch_lightning as pl
import torch


class DistillationTrainModel(pl.LightningModule):
    def __init__(
        self,
        student_model: torch.nn.Module,
        teacher_model: torch.nn.Module,
    ):
        super().__init__()
        self.student: torch.nn.Module = student_model.train()
        self.teacher: torch.nn.Module = teacher_model.eval()

        self.learning_rate: float = 5e-4
        self.adam_epsilon: float = 1e-6
        self.weight_decay: float = 0.0
        self.temperature: float = 2.0

    def training_step(self, batch, batch_idx):
        token_ids, attn_mask, lm_labels = batch
        student_outputs = self.student(
            input_ids=token_ids,
            attention_mask=attn_mask,
        )
        teacher_outputs = self.teacher(
            input_ids=token_ids,
            attention_mask=attn_mask,
        )

    def configure_optimizers(self):
        no_decay = no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.student.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.student.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            eps=self.adam_epsilon,
            betas=(0.9, 0.98),
        )
        return optimizer

    @staticmethod
    def calculate_cross_entropy(
        student_logits,
        teacher_logits,
        lm_labels,
        temperature: float,
    ):
        loss_fn = torch.nn.KLDivLoss(reduction="batchmean")
        mask = ((lm_labels > -1).unsqueeze(-1).expand_as(student_logits)).bool()
        s_logits_slct = torch.masked_select(student_logits, mask)
        s_logits_slct = s_logits_slct.view(-1, student_logits.size(-1))
        t_logits_slct = torch.masked_select(teacher_logits, mask)
        t_logits_slct = t_logits_slct.view(-1, teacher_logits.size(-1))
        assert t_logits_slct.size() == s_logits_slct.size()

        loss_ce = (
            loss_fn(
                torch.nn.functional.log_softmax(s_logits_slct / temperature, dim=-1),
                torch.nn.functional.softmax(t_logits_slct / temperature, dim=-1),
            )
            * (temperature) ** 2
        )
        return loss_ce

    @staticmethod
    def calculate_mlm_loss(student_logits, lm_labels):
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss_mlm = loss_fn(
            student_logits.view(-1, student_logits.size(-1)),
            lm_labels.view(-1),
        )
        return loss_mlm

    @staticmethod
    def calculate_cos_loss(
        s_hidden_states,
        t_hidden_states,
        attention_mask,
    ):

        loss_fn = torch.nn.CosineEmbeddingLoss(reduction="mean")
        mask = attention_mask.unsqueeze(-1).expand_as(s_hidden_states)
        dim = s_hidden_states.size(-1)
        s_hidden_states_slct = torch.masked_select(s_hidden_states, mask)
        s_hidden_states_slct = s_hidden_states_slct.view(-1, dim)
        t_hidden_states_slct = torch.masked_select(t_hidden_states, mask)
        t_hidden_states_slct = t_hidden_states_slct.view(-1, dim)
        target = s_hidden_states_slct.new(s_hidden_states_slct.size(0)).fill_(1)
        loss_cos = loss_fn(s_hidden_states_slct, t_hidden_states_slct, target)
        return loss_cos
