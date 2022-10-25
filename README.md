# 3Q 研究プロジェクト A (秋山研究室)

Tajimi らの論文 を元に、化合物の構造式から血漿タンパク質との結合率（Plasma protein binding, PPB）を予測する回帰モデルを構築したプログラム

## environment

以下の環境で動作することを確認しています

```
macOS 11.6.8 (Intel)
Python 3.10.8
```

## install

```sh
$ pip install -r requirements.txt
```

## usage

プロジェクトルートで、以下のコマンドを実行することにより動かすことができます

```sh
$ python exercise_*.py
```

ファイルの出力先は、`output`ディレクトリになります

## 引用論文

Takashi Tajimi, Naoki Wakui, Keisuke Yanagisawa, Yasushi Yoshikawa, Masahito Ohue, and Yutaka Akiyama. "Computational prediction of plasma protein binding of cyclic peptides from small molecule experimental data using sparse modeling techniques", BMC Bioinformatics 19(Suppl 19): 527, 2018. doi: [10.1186/s12859-018-2529-z](https://doi.org/10.1186/s12859-018-2529-z)
