import torch
from torch import nn, optim

class TrainModel:

    def __init__(self, model, ROOT_DIR, trainloader, testloader):

        # Define the optimizer (Stochastic Gradient Descent) with a learning rate of 0.0001
        optimizer = optim.SGD(model.parameters(), lr=0.0001)
        
        # Define the loss function (Cross-Entropy Loss)
        criterion = nn.CrossEntropyLoss()
        
        # Check if GPU is available
        print(torch.cuda.is_available())
        if torch.cuda.is_available():
            model = model.to("cuda")
            criterion = criterion.to("cuda")

        # Train the model
        self.train_data(criterion, model, optimizer, ROOT_DIR, trainloader)

        # Evaluate the model on the test set
        self.evaluate(model, testloader)

    def evaluate(self, model, testloader):
        device = "cuda"

        y_pred_list = []
        y_true_list = []

        with torch.no_grad():
            for x_batch, y_batch in testloader:
                if torch.cuda.is_available():
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_test_pred = model(x_batch)
                _, y_pred_tag = torch.max(y_test_pred, dim=1)
                y_pred_list.extend(y_pred_tag.cpu().numpy())
                y_true_list.extend(y_batch.cpu().numpy())

        y_true_list_max = [m.argmax() for m in y_true_list]

        correct_count, all_count = 0, 0
        for i in range(len(y_pred_list)):
            if (y_pred_list[i] == y_true_list_max[i]):
                correct_count += 1
            all_count += 1
        print("\nModel Accuracy =", round((correct_count / all_count), 4) * 100, '%')

    def train_data(self, criterion, model, optimizer, ROOT_DIR, trainloader):
        # Train the model for 5 epochs
        for i in range(5):

            running_loss = 0
            model.train()  # Set the model in training mode
            for images, labels in trainloader:

                if torch.cuda.is_available():
                    images = images.to("cuda")
                    labels = labels.to("cuda")

                optimizer.zero_grad()  # Zero the gradients

                output = model(images)  # Forward pass

                loss = criterion(output, labels)

                loss.backward()  # Backpropagation

                optimizer.step()  # Update weights

                running_loss += loss.item()
            else:
                print("Epoch {} - Training loss: {}".format(i + 1, running_loss / len(trainloader)))

        # Save the trained model if needed
        # filepath = ROOT_DIR + "model.pt"
        # torch.save(model.state_dict(), filepath)