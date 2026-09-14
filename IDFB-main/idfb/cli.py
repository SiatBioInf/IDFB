import argparse
import sys
from pathlib import Path

from idfb.config import LEARNING_RATE, N_EPOCHS, PSEUDO_NUM_SAMPLES
from idfb.demo import create_demo_raw_data, run_demo
from idfb.evaluate import run_evaluate
from idfb.integrate import run_integrate
from idfb.preprocess import run_preprocess
from idfb.pseudo import run_generate_pseudo
from idfb.tasks.cancertype import save_cancertype_processed
from idfb.tasks.lung_cancer import save_lung_cancer_processed
from idfb.tasks.msi import save_msi_processed
from idfb.tasks.survival import save_survival_processed
from idfb.train import run_train


def build_parser():
    parser = argparse.ArgumentParser(
        prog="idfb",
        description="IDFB: Integrate bulk RNA-seq across sequencing platforms",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="Run end-to-end demo with synthetic data")
    p_demo.add_argument("--epochs", type=int, default=5)
    p_demo.add_argument("--batch-size", type=int, default=16)
    p_demo.add_argument("--latent-dim", type=int, default=64)
    p_demo.add_argument("--n-genes", type=int, default=256)
    p_demo.add_argument("--data-only", action="store_true")

    p_pre = sub.add_parser(
        "preprocess", help="Preprocess real Dataset into processed_data.csv"
    )
    p_pre.add_argument(
        "--task",
        action="append",
        choices=[
            "MSI",
            "Cancertype",
            "Lung_cancer_subtybes",
            "Survival_analysis",
        ],
        help="Task to prepare; can repeat. Default: all tasks",
    )
    p_pre.add_argument("--skip-pseudo", action="store_true")
    p_pre.add_argument("--num-samples", type=int, default=PSEUDO_NUM_SAMPLES)
    p_pre.add_argument("--seed", type=int, default=42)

    p_pseudo = sub.add_parser(
        "generate-pseudo", help="Generate pseudo training samples"
    )
    p_pseudo.add_argument("--num-samples", type=int, default=PSEUDO_NUM_SAMPLES)
    p_pseudo.add_argument("--seed", type=int, default=42)

    p_train = sub.add_parser("train", help="Train GAN-VAE on pseudo data")
    p_train.add_argument("--batch-size", type=int, default=64)
    p_train.add_argument("--epochs", type=int, default=N_EPOCHS)
    p_train.add_argument("--lr", type=float, default=LEARNING_RATE)
    p_train.add_argument("--latent-dim", type=int, default=512)
    p_train.add_argument("--no-early-stopping", action="store_true")

    p_prep = sub.add_parser("prepare", help="Prepare a downstream task dataset")
    p_prep.add_argument(
        "--task",
        required=True,
        choices=[
            "MSI",
            "Cancertype",
            "Lung_cancer_subtybes",
            "Survival_analysis",
        ],
    )

    p_int = sub.add_parser("integrate", help="Generate platform-corrected data")
    p_int.add_argument("--task", type=str, default="MSI")
    p_int.add_argument("--model", type=str, default=None)

    p_eval = sub.add_parser("evaluate", help="Evaluate integration quality")
    p_eval.add_argument("--task", type=str, default="MSI")

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "demo":
        if args.data_only:
            create_demo_raw_data(n_genes=args.n_genes)
        else:
            run_demo(
                epochs=args.epochs,
                batch_size=args.batch_size,
                latent_dim=args.latent_dim,
                n_genes=args.n_genes,
            )
    elif args.command == "preprocess":
        run_preprocess(
            tasks=args.task,
            with_pseudo=not args.skip_pseudo,
            num_samples=args.num_samples,
            seed=args.seed,
        )
    elif args.command == "generate-pseudo":
        run_generate_pseudo(num_samples=args.num_samples, seed=args.seed)
    elif args.command == "train":
        run_train(
            batch_size=args.batch_size,
            n_epochs=args.epochs,
            lr=args.lr,
            latent_dim=args.latent_dim,
            early_stopping=not args.no_early_stopping,
        )
    elif args.command == "prepare":
        {
            "MSI": save_msi_processed,
            "Cancertype": save_cancertype_processed,
            "Lung_cancer_subtybes": save_lung_cancer_processed,
            "Survival_analysis": save_survival_processed,
        }[args.task]()
    elif args.command == "integrate":
        model = Path(args.model) if args.model else None
        run_integrate(args.task, model)
    elif args.command == "evaluate":
        run_evaluate(args.task)
    else:
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
