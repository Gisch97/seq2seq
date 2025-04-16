import torch as tr 

def compute_metrics(x_rec, x_true, mask):
    """
    Calculates the F1 score, accuracy (sequence level and threshold),
    precision, recall, and perplexity.

    Args:
        x_rec (torch.Tensor): Reconstructed sequence [N, C, L].
        x_true (torch.Tensor): True sequence [N, C, L]. 
        mask (torch.Tensor): Mask to filter valid sequences [N, C, L].

    Returns:
        dict: Calculated metrics.
    """
    xt_rec = x_rec.permute(0,2,1)
    xt_true = x_true.permute(0,2,1) 

    # Obtener el índice de la clase predicha y verdadera
    pred_idx = xt_rec.argmax(dim=-1)   # [B, L]
    true_idx = xt_true.argmax(dim=-1)  # [B, L]
 
    pred_flat = pred_idx[mask]    
    true_flat = true_idx[mask]
 
    TP = (pred_flat == true_flat).sum().item()    
    
    total = mask.sum().item()
    accuracy = TP / total if total > 0 else 0.0

    FP = FN = (pred_flat != true_flat).sum().item()
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
 
    seq_accuracy = (x_rec == x_true).all(dim=1).all(dim=1).float().mean().item()

    # Devolver todas las métricas
    return {"F1": f1, "Accuracy": accuracy, "Accuracy_seq": seq_accuracy}