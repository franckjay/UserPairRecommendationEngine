import torch
import torch.nn as nn


class EmbeddingRankingModel(nn.Module):
    def __init__(
        self,
        n_docs=3,
        layer_size=200,
        emb_size=10,
        n_users=2,
        n_user_feat=1,
        n_item_feat=1,
        n_text_feat=2,
        n_text_size=768,
        max_emb_vocab=100,
        batch_size=5,
    ):

        super(EmbeddingRankingModel, self).__init__()
        """

        :param n_docs: Number of items/docs we are ranking
        :param layer_size: NN layer size
        :param emb_size: How big of an embedding do we want?
        :param n_users: Number of users in the ranking context
        :param n_user_feat: How many user features are there per user?
        :param n_item_feat: Number of float features per item ranked
        :param n_text_feat: Number of text features per item we want to rank
        :param n_text_size: Vector size for each text embedding feature
        :param max_emb_vocab: How big of an embedding table do we need?
        :param batch_size: Training batch size
        """
        self.batch_size = batch_size
        self.user_embedding = nn.Embedding(max_emb_vocab, emb_size)
        self.item_embedding = nn.Embedding(max_emb_vocab, emb_size)
        n_text_embeddings = n_text_feat * n_docs * n_text_size
        n_context_vars = n_users * n_user_feat  # User age

        self.n_tot_features = (
            n_users * emb_size
            + n_docs * emb_size
            + n_docs * n_item_feat
            + n_text_embeddings
            + n_context_vars
        )

        _input = nn.Linear(self.n_tot_features, layer_size)
        _output = nn.Linear(layer_size, n_docs)

        self.layers = nn.Sequential(
            _input,
            nn.BatchNorm1d(layer_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(layer_size, layer_size),
            nn.BatchNorm1d(layer_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            _output,
        )
        self.double

    def forward(self, x, u_cats, i_cats):
        """
        Forward pass
        :param x: Float Tensor
        :param u_cats: User index tensor
        :param i_cats: Item index tensor
        :return: Predictions for this batch
        """

        # Take User and Item embeddings for each value
        u_embs = self.user_embedding(u_cats.long())
        i_embs = self.item_embedding(i_cats.long())
        # Orient along the correct axis
        u_embs = u_embs.view(self.batch_size, -1)
        i_embs = i_embs.view(self.batch_size, -1)
        # Concat float values and embeddings together
        x = torch.cat([u_embs, i_embs, x], 1)
        return self.layers(x)
