# Inspired by https://github.com/cezannec/CNN_Text_Classification/blob/master/CNN_Text_Classification.ipynb
import embeddings
import torch
import torch.nn as nn
from run_sentiment import encode_sentiment_data

from datasets import load_dataset


class SentimentLinear(nn.Module):
    def __init__(
        self,
        embedding_dim,
        drop_prob=0.5,
    ):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentLinear, self).__init__()

        # set class vars
        self.embedding_dim = embedding_dim

        # 1. embedding layer <-- for now disable the embedding layer since we don't retrain the embeddings
        # self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # set weights to pre-trained
        # self.embedding.weight = nn.Parameter(torch.from_numpy(embed_model.vectors)) # all vectors
        # (optional) freeze embedding weights
        # if freeze_embeddings:
        #    self.embedding.requires_grad = False

        # 2. convolutional layers
        # 3. final, fully-connected layer for classification
        self.fc1 = nn.Linear(embedding_dim, 32)
        self.fc2 = nn.Linear(32, 1)

        # 4. dropout and sigmoid layers
        self.dropout = nn.Dropout(drop_prob)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()


    # Defines how a batch of inputs, x, passes through the model layers.
    # returns a single, sigmoid-activated class score as output.
    def forward(self, embeds):
        # permute embedding dim to input channels (batch_size x in_channels x seq_length)
        x = embeds.mean(dim=1)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.dropout(x)

        # final logit
        logit = self.fc2(x)

        # sigmoid-activated --> a class score
        return self.sig(logit)


# training loop
def train(
    model, data_train, data_val, learning_rate=0.001, max_epochs=50, batch_size=128
):
    # loss and optimization functions
    criterion = nn.BCELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    best_val_acc = 0
    
    (X_train, y_train) = data_train
    (X_val, y_val) = data_val
    n_training_samples = len(X_train)
    batch_size = min(batch_size, n_training_samples)

    model.train()
    for epoch in range(1, max_epochs + 1):
        # batch loop
        train_correct = 0
        train_loss = 0
        n_batches = 0
        
        for batch_num, example_num in enumerate(
            range(0, n_training_samples, batch_size)
        ):
            y = torch.tensor(y_train[example_num : example_num + batch_size])
            x = torch.tensor(X_train[example_num : example_num + batch_size])
            model.zero_grad()

            # get the output from the model
            output = model(x)

            # calculate the loss and perform backprop
            loss = criterion(output.squeeze(), y.float())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

            for i, y_true in enumerate(y):
                if output[i] > 0.5 and y_true == 1.0 or output[i] < 0.5 and y_true == 0:
                    train_correct += 1

        # Evaluate on validation set after epoch
        model.eval()
        val_losses = []
        model.eval()
        y = torch.tensor(y_val)
        x = torch.tensor(X_val)
        output = model(x)
        val_loss = criterion(output.squeeze(), y.float())
        val_losses.append(val_loss.item())
        val_correct = 0
        for i, y_true in enumerate(y):
            if output[i] > 0.5 and y_true == 1.0 or output[i] < 0.5 and y_true == 0:
                val_correct += 1
        model.train()
        # print(y, output)
        print("Epoch: {}/{}...".format(epoch, max_epochs))
        print(
            "Train loss: {:.6f}...".format(train_loss / n_batches),
            f"Train correct: {train_correct}/{len(X_train)}",
            f"Train accuracy: {train_correct/len(X_train):.2%}",
        )
        print(
            "Val loss: {:.6f}".format(val_loss.item()),
            f"Val correct: {val_correct}/{len(X_val)}",
            f"Val accuracy: {val_correct/len(X_val):.2%}",
        )
        best_val_acc = val_correct/len(X_val) if val_correct/len(X_val) > best_val_acc else best_val_acc
        print(
            f"Best Val accuracy: {best_val_acc:.2%}",
        )
        print()


if __name__ == "__main__":
    EMBEDDING_SIZE = 50
    (X_train, y_train), (X_val, y_val) = encode_sentiment_data(
        load_dataset("glue", "sst2"),
        embeddings.GloveEmbedding(
            "wikipedia_gigaword", d_emb=EMBEDDING_SIZE, show_progress=True
        ),
        450,
        100,
    )

    print("X_train size:", len(X_train))
    print("X_val size:", len(X_val))

    train(
        SentimentLinear(EMBEDDING_SIZE),
        (X_train, y_train),
        (X_val, y_val),
        learning_rate=0.5,
        max_epochs=1000,
    )
