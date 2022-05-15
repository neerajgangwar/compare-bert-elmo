import os
from pathlib import Path
from tokenizers import BertWordPieceTokenizer


def createDirectories(projdir):
    if not os.path.exists(f"{projdir}/output"):
        os.mkdir(f"{projdir}/output")

    if not os.path.exists(f"{projdir}/output/tokenizers"):
        os.mkdir(f"{projdir}/output/tokenizers")


def trainTokenizer(projdir):
    inputdir = f"{projdir}/data/wikitext"
    outputdir = f"{projdir}/output/tokenizers"

    paths = [str(x) for x in Path(inputdir).glob("wikitext_train.txt")]
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=False
    )
    tokenizer.train(
        files=paths,
        vocab_size=30_000,
        min_frequency=2,
        limit_alphabet=1000,
        wordpieces_prefix="##",
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )
    tokenizer.save_model(f"{outputdir}", "wikitext_wordpiece")


if __name__ == "__main__":
    projdir = f"{os.path.dirname(os.path.realpath(__file__))}/../.."
    createDirectories(projdir)
    trainTokenizer(projdir)
