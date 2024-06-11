# Quantum Contrastive Learning<br>for High-Energy Physics Analysis at the LHC
## Learning quantum representations of classical high energy physics data with contrastive learning

Checklist
- Experiment 1: base classical contrastive learning model (mnist)
- Experiment 2: add quantum layers to base (mnist)
- Run both models on lhc data and record results on wandb
- Visualise embeddings and benefits of the quantum layer (swap?)
- Add fidelity in the contrastive loss function
- Try base model for contrastive triplet loss and make hybrid model
- Run on lhc data
- Try other layers resnet, VIT, QVIT etc
- Try extending SimCLR

Presentation
- wandb for experiment tracking
- package the code
- fastapi server