from ..model_meta import MetaType, model


class ModelConfig(object):
    def __init__(self):
        self.id_dimension = 8
        self.item_id_vocab = 500
        self.category_id_vocab = 500
        self.mlp_layers = 2
        self.nhead = 4

    @staticmethod
    @model("meta_sasrec_fusion", MetaType.ConfigParser)
    def parse(json_obj):
        conf = ModelConfig()
        conf.id_dimension = json_obj.get("id_dimension")
        conf.item_id_vocab = json_obj.get("item_id_vocab")
        conf.category_id_vocab = json_obj.get("category_id_vocab")
        conf.mlp_layers = json_obj.get("mlp_layers")
        conf.nhead = json_obj.get("nhead")

        return conf
