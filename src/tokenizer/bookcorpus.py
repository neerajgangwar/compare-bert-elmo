import os
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer


def createDirectories(projdir):
    if not os.path.exists(f"{projdir}/output"):
        os.mkdir(f"{projdir}/output")

    if not os.path.exists(f"{projdir}/output/tokenizers"):
        os.mkdir(f"{projdir}/output/tokenizers")


def trainTokenizer(projdir):
    inputdir = f"{projdir}/data/bookcorpus"
    outputdir = f"{projdir}/output/tokenizers"

    paths = [str(x) for x in Path(inputdir).glob("*.txt")]
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=paths,
        vocab_size=52_000,
        min_frequency=2,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    )
    tokenizer.save_model(f"{outputdir}", "bookcorpus")


if __name__ == "__main__":
    projdir = f"{os.path.dirname(os.path.realpath(__file__))}/../.."
    createDirectories(projdir)
    trainTokenizer(projdir)
