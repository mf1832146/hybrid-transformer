import json

import torch
from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.handlers import ModelCheckpoint, EarlyStopping
from pytorch_pretrained_bert import BertAdam
from tqdm import tqdm

from train.my_data_loader import DataLoaderX
from train.dataset import TreeDataSet, collate_fn
from torch.utils.data import DataLoader

from train.evaluation import batch_evaluate
from model.module import make_model, Train, GreedyEvaluate
from train.train_utils import LabelSmoothing, BLEU4, MyLoss, Rouge, Meteor
from ignite.contrib.handlers.tensorboard_logger import *
import numpy as np


class Solver:
    def __init__(self, args, ast2id, code2id, nl2id, id2nl):
        self.args = args

        self.ast2id = ast2id
        self.code2id = code2id
        self.nl2id = nl2id
        self.id2nl = id2nl

        self.nl_vocab_size = len(self.nl2id)
        self.epoch = 0

        if self.args.model in ['hybrid-transformer']:
            code_len = len(self.ast2id)
        else:
            code_len = len(self.code2id)

        self.model = make_model(code_vocab=code_len,
                                nl_vocab=len(self.nl2id),
                                N=self.args.num_layers,
                                d_model=self.args.model_dim,
                                d_ff=self.args.ffn_dim,
                                k=self.args.k,
                                h=self.args.num_heads,
                                num_features=3,
                                max_simple_name_len=self.args.max_simple_name_len,
                                dropout=self.args.dropout)

    def train(self):
        use_relative = True if self.args.model == 'hybrid-transformer' else False
        train_data_set = TreeDataSet(file_name=self.args.data_dir + '/train.json',
                                     ast_path=self.args.data_dir + '/tree/train/',
                                     ast2id=self.ast2id,
                                     nl2id=self.nl2id,
                                     max_ast_size=self.args.code_max_len,
                                     max_simple_name_size=self.args.max_simple_name_len,
                                     k=self.args.k,
                                     max_comment_size=self.args.comment_max_len,
                                     use_code=use_relative)
        valid_data_set = TreeDataSet(file_name=self.args.data_dir + '/valid.json',
                                     ast_path=self.args.data_dir + '/tree/valid/',
                                     ast2id=self.ast2id,
                                     nl2id=self.nl2id,
                                     max_ast_size=self.args.code_max_len,
                                     max_simple_name_size=self.args.max_simple_name_len,
                                     k=self.args.k,
                                     max_comment_size=self.args.comment_max_len,
                                     use_code=use_relative)

        train_loader = DataLoaderX(dataset=train_data_set,
                                   batch_size=self.args.batch_size,
                                   shuffle=True,
                                   collate_fn=collate_fn,
                                   num_workers=8)
        valid_loader = DataLoaderX(dataset=valid_data_set,
                                   batch_size=self.args.batch_size,
                                   shuffle=False,
                                   collate_fn=collate_fn)

        device = "cpu"

        if torch.cuda.is_available():
            if self.args.g != 0:
                torch.cuda.set_device(self.args.g)
            device = "cuda"
            print('use gpu')

        if self.args.load_epoch != '-1':
            self.load_model(load_epoch=self.args.load_epoch)
            print('load epoch ', self.args.load_epoch)

        model_opt = BertAdam(self.model.parameters(), lr=1e-4)
        criterion = LabelSmoothing(size=self.nl_vocab_size,
                                   padding_idx=0, smoothing=0.1)

        train_model = Train(self.model)
        greedy_evaluator = GreedyEvaluate(self.model, self.args.comment_max_len, self.nl2id['<s>'])

        trainer = create_supervised_trainer(train_model, model_opt, criterion, device)

        # metric_valid = {"bleu": BLEU4(self.id2nl), "rouge": Rouge(self.id2nl), "meteor": Meteor(self.id2nl)}
        metric_valid = {"bleu": BLEU4(self.id2nl)}

        """
        train + generator
        validation + greedy_decode
        """
        validation_evaluator = create_supervised_evaluator(greedy_evaluator, metric_valid, device)

        # save model
        save_handler = ModelCheckpoint('checkpoint/' + self.args.model, n_saved=10,
                                       filename_prefix='',
                                       create_dir=True,
                                       global_step_transform=lambda e, _: e.state.epoch + int(self.args.load_epoch) + 1,
                                       require_empty=False)
        trainer.add_event_handler(Events.EPOCH_COMPLETED(every=2), save_handler, {self.args.model: self.model})

        # early stop
        early_stop_handler = EarlyStopping(patience=20, score_function=self.score_function, trainer=trainer)
        validation_evaluator.add_event_handler(Events.COMPLETED, early_stop_handler)

        @trainer.on(Events.EPOCH_COMPLETED)
        # @trainer.on(Events.ITERATION_COMPLETED)
        def compute_metrics(engine):

            validation_evaluator.run(valid_loader)

            print('Epoch ' + str(self.epoch) + ' end')
            self.epoch += 1

        tb_logger = TensorboardLogger(self.args.log_dir + self.args.model + '/')

        tb_logger.attach(
            validation_evaluator,
            log_handler=OutputHandler(tag="validation", metric_names=["bleu"], another_engine=trainer),
            event_name=Events.EPOCH_COMPLETED,
        )

        log_batch = int(train_data_set.__len__() / self.args.batch_size / 10)
        if log_batch < 1:
            log_batch = 1
        tb_logger.attach(
            trainer,
            log_handler=OutputHandler(
                tag="training", output_transform=lambda loss: {"batchloss": loss}, metric_names="all"
            ),
            event_name=Events.ITERATION_COMPLETED(every=log_batch),
            # event_name=Events.ITERATION_COMPLETED(every=1),
        )

        trainer.run(train_loader, max_epochs=self.args.num_step)
        tb_logger.close()

    @staticmethod
    def score_function(engine):
        bleu = engine.state.metrics['bleu']
        return bleu

    def load_model(self, load_epoch):
        model_path = 'checkpoint/'+ self.args.model + '/' + self.args.model + '_' + load_epoch + '.pth'

        if torch.cuda.is_available():
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint)

    def test(self, load_epoch):
        self.load_model(load_epoch=load_epoch)
        self.model.eval()

        use_relative = True if self.args.model == 'hybrid-transformer' else False
        test_data_set = TreeDataSet(file_name=self.args.data_dir + '/test.json',
                                    ast_path=self.args.data_dir + '/tree/test/',
                                    ast2id=self.ast2id,
                                    nl2id=self.nl2id,
                                    max_ast_size=self.args.code_max_len,
                                    k=self.args.k,
                                    max_comment_size=self.args.comment_max_len,
                                    use_code=use_relative)

        test_loader = DataLoaderX(dataset=test_data_set,
                                  batch_size=self.args.batch_size,
                                  shuffle=False,
                                  collate_fn=collate_fn)

        greedy_evaluator = GreedyEvaluate(self.model, self.args.comment_max_len, self.nl2id['<s>'])

        results = []

        print('starting test...')

        for data_batch in tqdm(test_loader):

            inputs, batch_comments = data_batch
            batch_code = inputs[0]
            node_num = torch.sum(batch_code != 0, dim=1).numpy()

            y_pred = greedy_evaluator(inputs)
            references, hypothesises = batch_evaluate(batch_comments, y_pred, self.id2nl)

            for j in range(len(references)):
                results.append({
                    'node_len': str(node_num[j]),
                    'predict': ' '.join(hypothesises[j]) if len(hypothesises[j]) > 0 else '',
                    'true': ' '.join(references[j])
                })

        with open('predict_'+ self.args.model +'.json', 'w') as f:
            json.dump(results, f)

    def visualize(self, load_epoch):
        self.load_model(load_epoch=load_epoch)

        parent_emb = self.model.state_dict()['encoder.relative_pos_emb.parent_emb.weight']
        brother_emb = self.model.state_dict()['encoder.relative_pos_emb.brother_emb.weight']

        np.savetxt('parent_emb.txt', parent_emb.numpy())
        np.savetxt('brother_emb.txt', brother_emb.numpy())

    def gold_test(self, load_epoch):
        self.load_model(load_epoch=load_epoch)
        self.model.eval()

        if self.args.model in ['ast-transformer']:
            test_data_set = TreeDataSet(file_name=self.args.data_dir + '/test_gold.json',
                                        ast_path=self.args.data_dir + '/tree/test/',
                                        ast2id=self.ast2id,
                                        nl2id=self.nl2id,
                                        max_ast_size=self.args.code_max_len,
                                        k=self.args.k,
                                        max_comment_size=self.args.comment_max_len,
                                        use_code=True)

            test_loader = DataLoader(dataset=test_data_set,
                                     batch_size=1,
                                     shuffle=False,
                                     collate_fn=collate_fn)

            greedy_evaluator = GreedyEvaluate(self.model, self.args.comment_max_len, self.nl2id['<s>'])

            results = []

            print('starting gold test...')

            for data_batch in tqdm(test_loader):

                inputs, batch_comments = data_batch

                batch_code = inputs[0]
                node_num = torch.sum(batch_code != 0, dim=1).numpy()

                y_pred = greedy_evaluator(inputs)
                references, hypothesises = batch_evaluate(batch_comments, y_pred, self.id2nl)

                for j in range(len(references)):
                    results.append({
                        'node_len': str(node_num[j]),
                        'predict': ' '.join(hypothesises[j]) if len(hypothesises[j]) > 0 else '',
                        'true': ' '.join(references[j])
                    })

            with open('gold_predict_' + self.args.model + '.json', 'w') as f:
                json.dump(results, f)





