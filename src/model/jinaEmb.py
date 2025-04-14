import torch.nn as nn


# Jina Embedding Model + Classifier
class JinaEmbeddingClassifier(nn.Module):
    # Generator
    def __init__(self, embedding_model, dropout, hidden_size, num_classes):
        super(JinaEmbeddingClassifier, self).__init__()
        self.embedding_model = embedding_model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    # Forward
    def forward(self, **input_dicts):
        embedded_outputs = self.embedding_model(**input_dicts)
        x = self.dropout(embedded_outputs.pooler_output).float()
        logits = self.classifier(x)
        return logits

    # Overriding
    def to(self, device):
        super(JinaEmbeddingClassifier, self).to(device)
        self.embedding_model.to(device)
        return self
