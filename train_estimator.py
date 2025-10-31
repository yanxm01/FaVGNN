import torch.optim as optim
import torch.nn as nn

def train_estimator(model, G, features, sens, idx_sens_train, args, logger):
    optimizer = optim.Adam(model.estimator.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(args.e_epochs):
        optimizer.zero_grad()
        s = model.estimator(G, features)
        loss = criterion(s[idx_sens_train], sens[idx_sens_train].unsqueeze(1).float())
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0 or epoch == args.e_epochs - 1:
            logger.info(f'Epoch {epoch}, Loss: {loss.item():.4f}')







