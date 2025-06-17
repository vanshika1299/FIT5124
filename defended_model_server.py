import torch
import torch.nn.functional as F

def defend_output(logits, mode="adaptive", noise_level=0.15, flip_prob=0.1):
    probs = F.softmax(logits, dim=1)
    batch_size, num_classes = probs.shape

    if mode == "adaptive":
        # Step 1: Flip top class with probability `flip_prob`
        top_class = torch.argmax(probs, dim=1)
        flip_mask = torch.rand(batch_size) < flip_prob

        # Random wrong labels
        random_wrong = torch.randint_like(top_class, low=0, high=num_classes)
        # Ensure wrong label is different
        random_wrong = torch.where(random_wrong == top_class, (random_wrong + 1) % 10, random_wrong)

        final_label = torch.where(flip_mask, random_wrong, top_class)
        # Convert to soft vector (one-hot + uniform noise)
        noisy_probs = torch.full((batch_size, num_classes), noise_level / (num_classes - 1))
        for i in range(batch_size):
            noisy_probs[i][final_label[i]] = 1.0 - noise_level
        return noisy_probs

    else:
        return probs  # fallback
