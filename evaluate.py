from LSTMModel import lstm
from dataset import getData
from parser_my import args
import torch

def eval():
    # model = torch.load(args.save_file)
    model = lstm(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size=1)
    checkpoint = torch.load(args.save_file)
    model.load_state_dict(checkpoint['state_dict'])
    preds = []
    labels = []
    close_max, close_min, train_loader, test_loader = getData(args.corpusFile, args.sequence_length, args.batch_size)
    for idx, (x, label) in enumerate(test_loader):
        x = x.squeeze(1)
        pred = model(x)
        list = pred.data.squeeze(1).tolist()
        preds.extend(list[-1])
        labels.extend(label.tolist())
    print(len(preds))
    for i in range(len(preds)):
        print('未来第%.2d周 预测值是%.2f' % (i+1,
        preds[i][0] * (close_max - close_min) + close_min))

eval()