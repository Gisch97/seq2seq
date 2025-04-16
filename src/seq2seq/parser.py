import argparse
from ._version import __version__
def parser():   

    parser = argparse.ArgumentParser(
        prog="seq2seq",
        description="Autoencoder sequence-to-sequence: an end-to-end method for RNA sequence prediction based on deep learning",
        epilog="webserver link | None",
    )
    parser.add_argument("--global_config", type=str, help="Path to the global configuration file")
    parser.add_argument("-d", type=str, default="cpu", help="Device ('cpu' or 'cuda')")
    parser.add_argument("-batch", type=int, default=4, help="Batch size for handling sequences")
    parser.add_argument("-j", type=int, default=2, help="Number of parallel workers")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode (default: False)") 
    parser.add_argument("--max-length", type=int, help="Maximum sequence length to process (default: None") 
    parser.add_argument('--version', '-v', action='version', version='%(prog)s ' + __version__)
    parser.add_argument("--exp", type=str, default=None, help="Experiment name")
    parser.add_argument("--run", type=str, default=None, help="Run name (default: none)")
    parser.add_argument("--swaps", type=int, default=None, help="number of swaps in embedding (default: 0)")
    

    subparsers = parser.add_subparsers(
        title="Actions", dest="command", description="Running commands", required=False
    ) 

    parser_train = subparsers.add_parser("train", help="Train a new model")
    
    parser_train.add_argument(
        "--train_file",
        type=str,
        help="Training dataset (csv file with 'id', 'sequence')",
    )

    parser_train.add_argument(
        "--valid-file",
        type=str,
        help="Validation dataset to stop training. If not provided, validation split is randomly generated from training data. Columns are the same as training",
    )
    parser_train.add_argument(
        "-o",
        type=str,
        dest="out_path",
        help="Output path (if not provided, it is generated with the current date)",
    )

    parser_train.add_argument(
        "-n", "--max-epochs",
        type=int,
        # default=1000,
        help="Maximum number of training epochs",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true", help="Cache of data for training (default: cache is used)",
    )
    
    parser_train.add_argument("--train_config", type=str, help="Path to the train config file")


    # test parser
    parser_test = subparsers.add_parser("test", help="Test a model")
    parser_test.add_argument(
        "--test_file",
        type=str,
        help="Testing dataset (csv file with 'id', 'sequence')",
    )
    parser_test.add_argument(
        "-w", type=str, dest="model_weights", help="Trained model weights"
    )
    parser_test.add_argument(
        "-o",
        type=str, dest="out_path", 
        help="Output test metrics (default: only printed on the console)",
    )
    parser_test.add_argument("--test_config", type=str, help="Path to the test config file")

    # pred parser
    parser_pred = subparsers.add_parser(
        "pred", help="Predict structures for a list of sequences"
    )
    parser_pred.add_argument(
        "--name", type=str, default="console_input", dest="sequence_name", help="Sequence name (default: console_input)"
    )

    parser_pred.add_argument(
        "--pred_file",
        type=str,
        help="Dataset to predict. It can be a csv file with 'id' and 'sequence' columns or a fasta file",
    )
    parser_pred.add_argument(
        "-o",
        type=str, dest="out_path", 
        help="Output path, it can be a .csv file or a directory to hold CT files or interaction score files  (default: pred.csv)",
    )

    parser_pred.add_argument(
        "-w", type=str, dest="model_weights", help="Trained model weights"
    )
    parser_pred.add_argument("--pred_config", type=str, help="Path to the pred config file")

    return parser


def get_parser_defaults():
    """
    Devuelve los valores por defecto definidos en el parser.
    """
    return vars(parser().parse_args([]))
    