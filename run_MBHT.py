import argparse
from logging import getLogger
import os
from recbole.config import Config
from recbole.data import create_dataset
from recbole.data.utils import get_dataloader, create_samplers
from recbole.model.sequential_recommender.mbht import MBHT
from recbole.utils import init_logger, init_seed, get_model, get_trainer, set_color


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='MBHT', help='Model for session-based rec.')
    parser.add_argument('--dataset', '-d', type=str, default='tmall_beh', help='Benchmarks for session-based rec.')
    parser.add_argument('--validation', action='store_true', help='Whether evaluating on validation set (split from train set), otherwise on test set.')
    parser.add_argument('--valid_portion', type=float, default=0.1, help='ratio of validation set.')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=2048)
    return parser.parse_known_args()[0]


if __name__ == '__main__':
    args = get_args()

    # configurations initialization
    config_dict = {
        'USER_ID_FIELD': 'session_id',
        'load_col': None,
        # 'neg_sampling': {'uniform':1},
        'neg_sampling': None,
        'benchmark_filename': ['train', 'val', 'test'],
        'alias_of_item_id': ['item_id_list'],
        'topk': [5, 10],
        'metrics': ['Recall', 'NDCG'],
        'valid_metric': 'NDCG@10',
        'eval_args':{
            'mode':'full',
            'order':'TO'
            },
        'gpu_id':args.gpu_id,
        "MAX_ITEM_LIST_LENGTH":199,
        "train_batch_size": 64 if args.dataset == "ijcai_beh" else 128,
        "eval_batch_size":48 if args.dataset == "ijcai_beh" else 256,
        "hyper_len":10 if args.dataset == "ijcai_beh" else 6,
        "scales":[10, 4, 20],
        "enable_hg":1,
        "enable_ms":1,
        "customized_eval":1,
        "abaltion":"",
        'stopping_step': 3,
        'epochs': 8
    }

    if args.dataset == "retail_beh":
        config_dict['scales'] = [5, 4, 20]
        config_dict['hyper_len'] = 6
        
    config = Config(model="MBHT", dataset=f'{args.dataset}', config_dict=config_dict)
    # config['device']="cpu"
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config, log_root="log")
    logger = getLogger()

    logger.info(f"PID: {os.getpid()}")
    logger.info(args)
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_dataset, val_dataset, test_dataset = dataset.build()
    train_sampler, val_sampler, test_sampler = create_samplers(config, dataset, [train_dataset, val_dataset, test_dataset])
    train_data = get_dataloader(config, 'train')(config, train_dataset, None, shuffle=True)
    val_data = get_dataloader(config, 'valid')(config, val_dataset, None, shuffle=False)
    test_data = get_dataloader(config, 'test')(config, test_dataset, None, shuffle=False)

    # model loading and initialization
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model training and evaluation
    best_valid_score, best_valid_result = trainer.fit(
            train_data, val_data, saved=True, show_progress=config['show_progress']
        )
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])
    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')
