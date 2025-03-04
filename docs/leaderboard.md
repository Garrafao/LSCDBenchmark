# Leaderboard

The leaderboard for each task, max performance, paper, datasets.

| Task | Dataset       | Preprocessing  | Spelling Normalization | Evaluation | Model                              | Similarity Metric | Performance (Pearson) | Performance (Spearman) |
|------|---------------|----------------|------------------------|------------|------------------------------------|-------------------|-----------------------|------------------------|
| **German** ||||||||
| wic  | dwug_de_300   | raw            | german                 | wic        | XLMR-Large                         | Cosine            | 0.14                  | 0.16                   |
| wic  | dwug_de_300   | raw            | none                   | wic        | XLMR-Large                         | Cosine            | 0.14                  | 0.16                   |
| wic  | dwug_de_300   | normalization  | german                 | wic        | XLMR-Large                         | Cosine            | 0.13                  | 0.16                   |
| wic  | dwug_de_300   | normalization  | none                   | wic        | XLMR-Large                         | Cosine            | 0.13                  | 0.16                   |
| wic  | dwug_de_300   | raw            | german                 | wic        | XL-Lexeme                          | Cosine            | 0.60                  | 0.56                   |
| wic  | dwug_de_300   | raw            | none                   | wic        | XL-Lexeme                          | Cosine            | 0.60                  | 0.56                   |
| wic  | dwug_de_300   | normalization  | german                 | wic        | XL-Lexeme                          | Cosine            | 0.60                  | 0.56                   |
| wic  | dwug_de_300   | normalization  | none                   | wic        | XL-Lexeme                          | Cosine            | 0.60                  | 0.56                   |
| wic  | dwug_de_300   | raw            | german                 | wic        | deepmistake/WIC_DWUG+XLWSD         | Cosine            | 0.64                  | 0.59                   |
| wic  | dwug_de_300   | raw            | german                 | wic        | deepmistake/WIC+RSS+DWUG+XLWSD     | Cosine            | 0.64                  | 0.60                   |
| wic  | dwug_de_300   | raw            | none                   | wic        | deepmistake/WIC_DWUG+XLWSD         | Cosine            | 0.64                  | 0.59                   |
| wic  | dwug_de_300   | raw            | none                   | wic        | deepmistake/WIC+RSS+DWUG+XLWSD     | Cosine            | 0.64                  | 0.60                   |
| wic  | dwug_de_300   | normalization  | german                 | wic        | deepmistake/WIC_DWUG+XLWSD         | Cosine            | 0.64                  | 0.59                   |
| wic  | dwug_de_300   | normalization  | german                 | wic        | deepmistake/WIC+RSS+DWUG+XLWSD     | Cosine            | 0.64                  | 0.60                   |
| wic  | dwug_de_300   | normalization  | none                   | wic        | deepmistake/WIC_DWUG+XLWSD         | Cosine            | 0.65                  | 0.59                   |
| wic  | dwug_de_300   | normalization  | none                   | wic        | deepmistake/WIC+RSS+DWUG+XLWSD     | Cosine            | 0.64                  | 0.59                   |
| **English** ||||||||
| wic  | dwug_en_300   | raw            | none                   | wic        | deepmistake/WIC_DWUG+XLWSD         | Cosine            | 0.65                  | 0.65                   |
| wic  | dwug_en_300   | raw            | none                   | wic        | deepmistake/WIC+RSS+DWUG+XLWSD     | Cosine            | 0.65                  | 0.63                   |
| wic  | dwug_en_300   | raw            | english                | wic        | deepmistake/WIC_DWUG+XLWSD         | Cosine            | 0.63                  | 0.65                   |
| wic  | dwug_en_300   | raw            | english                | wic        | deepmistake/WIC+RSS+DWUG+XLWSD     | Cosine            | 0.63                  | 0.65                   |
| wic  | dwug_en_300   | normalization  | none                   | wic        | deepmistake/WIC_DWUG+XLWSD         | Cosine            | 0.65                  | 0.65                   |
| wic  | dwug_en_300   | normalization  | none                   | wic        | deepmistake/WIC+RSS+DWUG+XLWSD     | Cosine            | 0.63                  | 0.63                   |
| wic  | dwug_en_300   | normalization  | english                | wic        | deepmistake/WIC_DWUG+XLWSD         | Cosine            | 0.65                  | 0.65                   |
| wic  | dwug_en_300   | normalization  | english                | wic        | deepmistake/WIC+RSS+DWUG+XLWSD     | Cosine            | 0.63                  | 0.63                   |
| **Swedish** ||||||||
| wic  | dwug_sv_300   | raw            | none                   | wic        | deepmistake/WIC_DWUG+XLWSD         | Cosine            | 0.67                  | 0.64                   |
| wic  | dwug_sv_300   | raw            | none                   | wic        | deepmistake/WIC+RSS+DWUG+XLWSD     | Cosine            | 0.66                  | 0.65                   |
| wic  | dwug_sv_300   | raw            | swedish                | wic        | deepmistake/WIC_DWUG+XLWSD         | Cosine            | 0.67                  | 0.64                   |
| wic  | dwug_sv_300   | raw            | swedish                | wic        | deepmistake/WIC+RSS+DWUG+XLWSD     | Cosine            | 0.66                  | 0.65                   |
| wic  | dwug_sv_300   | normalization  | none                   | wic        | deepmistake/WIC_DWUG+XLWSD         | Cosine            | 0.67                  | 0.64                   |
| wic  | dwug_sv_300   | normalization  | none                   | wic        | deepmistake/WIC+RSS+DWUG+XLWSD     | Cosine            | 0.66                  | 0.65                   |
| wic  | dwug_sv_300   | normalization  | swedish                | wic        | deepmistake/WIC_DWUG+XLWSD         | Cosine            | 0.67                  | 0.64                   |
| wic  | dwug_sv_300   | normalization  | swedish                | wic        | deepmistake/WIC+RSS+DWUG+XLWSD     | Cosine            | 0.66                  | 0.65                   |