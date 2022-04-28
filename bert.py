from transformers import BertForSequenceClassification, BertPreTrainedModel

class BertEnsembleModel(BertPreTrainedModel):
    def __init__(self, model_path, config=None, args=None):
        super().__init__(config)
        self.args = args

        self.model1 = BertForSequenceClassification.from_pretrained(
            model_path,
            config=config,
        )
        if args.ensemble:
            self.model2 = BertForSequenceClassification.from_pretrained(
                model_path,
                config=config,
            )

    def forward(self, *model_args, **kwargs):
        res1 = self.model1(*model_args, **kwargs)

        if self.args.ensemble:
            res2 = self.model2(*model_args, **kwargs)
            return res1, res2

        return res1