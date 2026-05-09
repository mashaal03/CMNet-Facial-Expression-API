import torch
import torch.nn as nn
import torch.nn.functional as F

class AlgorithmicLDLLoss(nn.Module):
    def __init__(self, device='cuda'):
        super(AlgorithmicLDLLoss, self).__init__()
        self.device = device
        
        # ---------------------------------------------------------
        # THE BIOLOGICAL SIMILARITY MATRIX
        # Indices: 0:Surprise, 1:Fear, 2:Disgust, 3:Happy, 4:Sad, 5:Anger, 6:Neutral
        # ---------------------------------------------------------
        # Each row must sum exactly to 1.0
        self.similarity_matrix = torch.tensor([
            [0.80, 0.15, 0.00, 0.05, 0.00, 0.00, 0.00], # 0: True Surprise (Leaks to Fear/Happy)
            [0.15, 0.80, 0.00, 0.00, 0.05, 0.00, 0.00], # 1: True Fear (Leaks to Surprise/Sad)
            [0.00, 0.00, 0.80, 0.00, 0.05, 0.15, 0.00], # 2: True Disgust (Leaks to Anger/Sad)
            [0.10, 0.00, 0.00, 0.85, 0.00, 0.00, 0.05], # 3: True Happy (Leaks to Surprise/Neutral)
            [0.00, 0.10, 0.00, 0.00, 0.80, 0.00, 0.10], # 4: True Sad (Leaks to Fear/Neutral)
            [0.00, 0.05, 0.15, 0.00, 0.00, 0.80, 0.00], # 5: True Anger (Leaks to Disgust/Fear)
            [0.05, 0.00, 0.00, 0.05, 0.05, 0.00, 0.85], # 6: True Neutral (Leaks to Sad/Happy/Surprise)
        ], dtype=torch.float32).to(self.device)

        # KLDivLoss expects log-probabilities for inputs and standard probabilities for targets
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, logits, targets):
        """
        logits: Raw outputs from the neural network (Batch_size, 7)
        targets: 1D tensor of ground truth integers (Batch_size)
        """
        # 1. Convert raw logits to Log-Probabilities
        log_preds = F.log_softmax(logits, dim=1)
        
        # 2. Map the 1D integer targets to our 2D Soft Distribution Matrix
        # E.g., Target '2' becomes [0.00, 0.00, 0.80, 0.00, 0.05, 0.15, 0.00]
        soft_targets = self.similarity_matrix[targets]
        
        # 3. Calculate KL Divergence (How far off is the model from the soft distribution?)
        loss = self.kl_loss(log_preds, soft_targets)
        
        return loss