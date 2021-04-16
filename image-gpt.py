from comet_ml import Experiment
import torch
import numpy as np
from preprocess import *
from transformers import GPT2LMHeadModel, GPT2Config, AdamW
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hyperparameters = {
    "batch_size": 10,
    "num_epochs": 5,
    "learning_rate": .0001,
    "num_heads": 4,
    "num_layers": 4,
    "embedding_size": 200
}

def train(model, train_loader, optimizer, experiment):
    model = model.train()
    with experiment.train():
        for i in range(hyperparameters["num_epochs"]):
            for batch in train_loader:
                input = batch["input"].to(device) #trains model to predict next pixel given previous pixels
                optimizer.zero_grad()
                output = model(input,labels=input)
                mean_loss = output[0]
                loss = mean_loss * (len(input[0])-1)
                loss.backward()
                optimizer.step()

def test(model, test_loader, experiment):
    model = model.eval()
    with experiment.test():
        for batch in test_loader:
            input = batch["input"].to(device)
            with torch.no_grad():
                output = model(input,labels=input)
            mean_loss = output[0]
            loss = mean_loss * (len(input[0])-1)
    experiment.end()

def sample(image, pixels_to_predict):
    #repeatedly predict next pixel
    input = list(image)
    for i in range(pixels_to_predict):
        with torch.no_grad():
            output = model(np.array(input).to(device),labels=label)
        logits = output[1]
        prediction = np.argmax(logits, axis=2)
        input.append(prediction)
    #reshape to produce image
    input = np.array(input)
    input = np.reshape((32,32))

if __name__ == "__main__":
    parser = argparse.ArgumentParser() #code based on CS1460 Computational Linguistics argument parsing
    parser.add_argument("file1")
    parser.add_argument("file2")
    parser.add_argument("file3")
    parser.add_argument("file4")
    parser.add_argument("file5")
    parser.add_argument("-l", "--load", action="store_true",
                        help="load model.pt")
    parser.add_argument("-s", "--save", action="store_true",
                        help="save model.pt")
    parser.add_argument("-T", "--train", action="store_true",
                        help="run training loop")
    parser.add_argument("-t", "--test", action="store_true",
                        help="run testing loop")
    parser.add_argument("-S", "--sample", action="store_true",
                        help="sample")
    args = parser.parse_args()

    model = GPT2LMHeadModel(GPT2Config(vocab_size=256,n_embd=hyperparameters["embedding_size"],n_layer=hyperparameters["num_layers"],n_head=hyperparameters["num_heads"])).to(device) #from 0-255

    optimizer = AdamW(model.parameters(),lr=hyperparameters["learning_rate"])

    if args.train or args.test:
        train_loader,test_loader = load_dataset([args.file1, args.file2, args.file3, args.file4, args.file5], hyperparameters["batch_size"])
        experiment = Experiment(api_key="cdVj0ApyXZj7uxTF7EeCgH3cu", project_name="computer-vision", workspace="brynnchernosky", log_code=False)
        experiment.log_parameters(hyperparameters)

    if args.load:
        model.load_state_dict(torch.load('model.pt'))
    if args.train:
        train(model, train_loader, optimizer, experiment)
    if args.save:
        torch.save(model.state_dict(), 'model.pt')
    if args.test:
        test(model, test_loader, experiment)
    if args.sample:
        sample()
