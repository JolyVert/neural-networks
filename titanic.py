import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import joblib
import numpy as np
import torch.nn.functional as F

BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 10000
PATIENCE = 150

try:
    X_train_loaded = joblib.load('data/X_train_processed.pkl')
    Y_train_loaded = joblib.load('data/y_train_processed.pkl')
    loaded_scaler = joblib.load('data/titanic_scaler.pkl')

    X_train_tensor = torch.tensor(X_train_loaded.values, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train_loaded, dtype=torch.float32).unsqueeze(1)

except FileNotFoundError:
    print("ERROR: Make sure 'X_train_processed.pkl' and 'Y_train.pkl' exist in the directory!")
    raise

train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

INPUT_SIZE = X_train_tensor.shape[1]
print(f"Number of input features (INPUT_SIZE): {INPUT_SIZE}")
print(f"Number of training samples: {len(X_train_tensor)}")
print("-" * 30)


class SimpleMLP(nn.Module):
    def __init__(self, input_size):
        super(SimpleMLP, self).__init__()
        HIDDEN_SIZE_1 = 32
        HIDDEN_SIZE_2 = 16
        self.fc1 = nn.Linear(input_size, HIDDEN_SIZE_1)
        self.fc2 = nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2)
        self.fc3 = nn.Linear(HIDDEN_SIZE_2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


model = SimpleMLP(INPUT_SIZE)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

loss_history = []
accuracy_history = []
min_val_loss = np.inf
patience_counter = 0

print("\n--- Training Started ---")

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    correct_predictions = 0
    total_samples = 0

    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * inputs.size(0)
        predicted = (outputs > 0.5).float()
        correct_predictions += (predicted == targets).sum().item()
        total_samples += targets.size(0)

    avg_loss = epoch_loss / total_samples
    accuracy = correct_predictions / total_samples
    loss_history.append(avg_loss)
    accuracy_history.append(accuracy)

    if avg_loss < min_val_loss:
        min_val_loss = avg_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_titanic_model.pth')
    else:
        patience_counter += 1

    print(f"Epoch {epoch+1}/{NUM_EPOCHS}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")

    if patience_counter >= PATIENCE:
        print(f"\n--- Stopped early at epoch {epoch+1} ---")
        print(f"No loss improvement for {PATIENCE} epochs. Loading best model.")
        model.load_state_dict(torch.load('best_titanic_model.pth'))
        break

print("\nTraining finished.")
print("-" * 30)

model.eval()
with torch.no_grad():
    sample_input = X_train_tensor[0].unsqueeze(0)
    prediction = model(sample_input)
    predicted_class = 1 if prediction.item() >= 0.5 else 0

    print(f"Sample input (tensor): {sample_input}")
    print(f"Predicted survival probability: {prediction.item():.4f}")
    print(f"Predicted class (0/1): {predicted_class}")
    print(f"Actual class: {Y_train_tensor[0].item()}")
