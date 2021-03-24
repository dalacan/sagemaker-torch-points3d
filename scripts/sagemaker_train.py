import argparse
import subprocess
import os
import json

def main():
    # Get SageMaker parameters
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_dir', default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--output_data_dir', default=os.environ.get('SM_OUTPUT_DATA_DIR'))    
    parser.add_argument('--train_dir', default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--task', default='segmentation')
    parser.add_argument('--model_type', default='pointnet2')
    parser.add_argument('--model_name', default='pointnet2_charlesssg')
    parser.add_argument('--dataset_name', default='shapenet-fixed') 
    parser.add_argument('--weight_name', default='miou') 
    parser.add_argument('--forward_category', default='Cap') 
    parser.add_argument('--epochs', default='100')
    parser.add_argument('--lr', default='0.001')
    parser.add_argument('--hydra_verbose', default='true')
    parser.add_argument('--hydra_pretty_print', default='true')
    args = parser.parse_args()
    
    # Pass in hydra configuration overrides
    train_cmd = ["python3", "train.py",
        "hydra.run.dir={}".format(args.model_dir), 
        "hydra.job_logging.handlers.file_handler.filename={}/train.log".format(args.output_data_dir),
        "data.dataroot={}".format(args.train_dir),
        "task={}".format(args.task),
        "model_type={}".format(args.model_type),
        "model_name={}".format(args.model_name),
        "dataset={}".format(args.dataset_name), 
        "training.epochs={}".format(args.epochs),
        "training.optim.base_lr={}".format(args.lr),
        "hydra.verbose={}".format(args.hydra_verbose),
        "pretty_print={}".format(args.hydra_pretty_print),
        "wandb.log=false",
    ]

    # Output training inputs
    print(args.train_dir)
    print(os.listdir(args.train_dir))    
    
    # Write config so can load model type for inference
    config = vars(args)
    print('saving config: {}'.format(config))
    with open(os.path.join(args.model_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Call into subprocess and get output
    print('running subprocess: {}'.format(' '.join(train_cmd)))
    p = subprocess.run(train_cmd, stdout=subprocess.PIPE)
    
    # Write the output and error to logs and return error code
    if p.stdout != None:
        print('process output:')
        print(p.stdout.decode('utf-8'))
    if p.stderr != None:
        print('process error:')
        print(p.stderr.decode('utf-8'))
    return p.returncode

if __name__ == "__main__":
    main()