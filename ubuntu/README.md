# Analysis of the Ubuntu Dialogue Corpus v2.0


# Setup

Download Ubuntu Dialogue Corpus v2.0 using the scripts:

git clone https://github.com/rkadlec/ubuntu-ranking-dataset-creator.git
cd ubuntu-ranking-dataset-creator
pip install -r requirements.txt
cd src
./generate.sh -t -s -l


# Content

1,852,869 dialogues in TSV format, one dialogue per file.
Dialog line format: [0] timestamp [1] sender [2] recepeint [3] utterance [4] named entities for [3] annotated with DBpedia Spotlight.



