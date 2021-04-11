import torch
import numpy as np
from preprocess import *
from transformers import GPT2Model, AdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hyperparameters = {
    "batch_size": 100,
    "embedding_size": 500,
    "num_heads": 12,
    "num_layers": 12,
    "num_epochs": 3,
    "learning_rate": .001
}

def train(model, train_loader, optimizer, experiment):
    model = model.train()
    with experiment.train():
        for i in range(hyperparameters["num_epochs"]):
            for batch in train_loader:
                input,label = batch["input"],batch["label"]
                optimizer.zero_grad()
                output = model(input,labels=label)
                logits = output[1]
                prediction = torch.argmax(logits,dim=2)
                #do something here!
                loss.backward()
                optimizer.step()

                accuracy = correct/total
                perplexity = torch.exp(loss/total).item()
                experiment.log_metric("accuracy", accuracy)
                experiment.log_metric("perplexity", perplexity)

def test(model, test_loader, experiment):
    model = model.eval()
    with experiment.test():
        for batch in test_loader:
            input,label = batch["input"], batch["label"]
            with torch.no_grad():
                output = model(input,labels=label)

            logits = output[1]
            #do something here!
            experiment.log_metric("accuracy", accuracy)
            experiment.log_metric("perplexity", perplexity)

def sample():
    #do something here!

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file")
    parser.add_argument("test_file")
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

    model = GPT2Model()

    optimizer = AdamW(model.parameters(),lr=hyperparameters["learning_rate"])

    if args.train or args.test:
        train_loader, test_loader = load_dataset(args.train_file, args.test_file)
        experiment = Experiment(log_code=False)
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
